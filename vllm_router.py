import os
import requests
import math
import time
import json
from typing import List, Optional, Dict, Any, Tuple
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoTokenizer 

SYSTEM_PROMPT = (
    "You are a reasoning assistant. You must solve the problem step by step. "
    "CRITICAL REQUIREMENT: After completing EACH logical step, you MUST output the tag '</step>' on a new line. "
    "Do not output the final answer until you have gone through all necessary steps. "
    "Example format:\n<think>\nStep 1 analysis...\n</step>\nStep 2 calculation...\n</step>\n...<think>\nFinal Answer."
)

@register_model("vllm_router")
class VLLMRouter(LM):
    def __init__(
        self,
        small_url: str,
        large_url: str,
        small_model_name: str,
        large_model_name: str,
        confidence_threshold: float = 0.6,
        max_step_tokens: int = 2048,
        max_gen_tokens: int = 16384,
        step_tag: str = "</step>",
        **kwargs
    ):
        super().__init__()
        self.small_url = small_url
        self.large_url = large_url
        self.small_model_name = small_model_name
        self.large_model_name = large_model_name

        try:
            self.confidence_threshold = float(confidence_threshold)
        except (ValueError, TypeError):
            self.confidence_threshold = 0.6

        self.max_step_tokens = max_step_tokens
        self.max_gen_tokens = max_gen_tokens
        self.step_tag = step_tag
        self.system_prompt = SYSTEM_PROMPT

        # 加载 Tokenizer
        print(f"Loading Small Model Tokenizer: {small_model_name}...")
        self.tokenizer_small = AutoTokenizer.from_pretrained(small_model_name, trust_remote_code=True)
        if self.tokenizer_small.pad_token is None:
            self.tokenizer_small.pad_token = self.tokenizer_small.eos_token

        print(f"Loading Large Model Tokenizer: {large_model_name}...")
        self.tokenizer_large = AutoTokenizer.from_pretrained(large_model_name, trust_remote_code=True)
        if self.tokenizer_large.pad_token is None:
            self.tokenizer_large.pad_token = self.tokenizer_large.eos_token
            
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
        self.stats = {
            "total_requests": 0,
            "fallback_to_large": 0,
            "tokens_generated_by_small": 0,
            "tokens_generated_by_large": 0,
            "prefill_tokens_count": 0
        }

    @property
    def tokenizer_name(self) -> str:
        return self.large_model_name

    def _call_vllm_api(
        self, 
        url: str, 
        model_name: str,
        prompt_text: str,
        max_tokens: int = 256, 
        stop_sequences: List[str] = None,
        get_logprobs: bool = True,
        temperature: float = 0.7,
        top_k_logprobs: int = 5 
    ) -> Optional[Dict]:
        
        endpoint = f"{url.rstrip('/')}/v1/completions"

        payload = {
            "model": model_name,
            "prompt": prompt_text,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "n": 1
        }

        if get_logprobs:
            payload["logprobs"] = top_k_logprobs
        
        if stop_sequences:
            payload["stop"] = stop_sequences

        try:
            response = self.session.post(endpoint, json=payload, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                if "choices" in data:
                    return data
                elif "outputs" in data:
                    return data
                return {"choices": [{"text": "", "logprobs": None}], "raw": data}
                
            else:
                # 仅在非 200 时打印错误，减少噪音
                if response.status_code == 400:
                    err_text = response.text[:150]
                    if "logprobs" in err_text and "max_logprobs" in err_text:
                        print(f"⚠️ API Error 400: logprobs limit exceeded. Try reducing top_k_logprobs.")
                    else:
                        print(f"⚠️ API Error 400: {err_text}")
                else:
                    print(f"⚠️ API Error {response.status_code}: {response.text[:100]}")
                return None

        except Exception as e:
            print(f"⚠️ Network Error: {type(e).__name__} - {e}")
            return None

    def _calculate_confidence(self, logprobs_data) -> float:
        if not logprobs_data:
            return 0.0
        
        probs = []
        # 适配 vLLM 返回的字典格式 {'token_logprobs': [...]}
        if isinstance(logprobs_data, dict):
            token_logprobs_list = logprobs_data.get("token_logprobs", [])
            if token_logprobs_list and all(isinstance(x, (int, float)) for x in token_logprobs_list):
                probs = [math.exp(float(x)) for x in token_logprobs_list if x is not None]
        
        return sum(probs) / len(probs) if probs else 0.0

    def _run_speculative_routing(self, messages: List[Dict[str, str]], stop_sequences: List[str]) -> str:
        """
        核心路由逻辑。
        messages: 纯文本内容的对话列表 [{'role': 'user', 'content': '...'}, ...]
        在此函数内部统一应用模板，避免外部重复处理。
        """
        conversation_history = list(messages)
        all_generated_segments = [] 
        
        max_steps = 20
        step_count = 0
        force_large_model_this_step = False 

        while step_count < max_steps:
            current_step = step_count + 1
            use_large_model = force_large_model_this_step
            force_large_model_this_step = False
            
            # 【关键】在此处统一应用模板，将 messages 转换为 prompt 字符串
            # 无论是小模型还是大模型，都使用大模型的 tokenizer 来构建 prompt (通常小模型兼容大模型格式，或者两者格式一致)
            # 如果小模型格式完全不同，需要在这里加判断，但通常 Speculative Decoding 要求格式一致
            current_prompt_text = self._apply_template_safe(conversation_history)
            
            generated_text = ""
            new_token_ids = []

            # ================= 分支 A: 小模型执行 =================
            if not use_large_model:
                small_res = self._call_vllm_api(
                    url=self.small_url,
                    model_name=self.small_model_name,
                    prompt_text=current_prompt_text,
                    max_tokens=self.max_step_tokens,
                    stop_sequences=stop_sequences,
                    get_logprobs=True,
                    temperature=0.7
                )
                
                if small_res and small_res.get("choices"):
                    choice = small_res["choices"][0]
                    generated_text = choice.get("text", "")
                    
                    # 获取 token_ids (优先API返回，否则本地encode)
                    new_token_ids = choice.get("token_ids")
                    if not new_token_ids and generated_text:
                        new_token_ids = self.tokenizer_small.encode(generated_text, add_special_tokens=False)
                    
                    logprobs_raw = choice.get("logprobs", [])
                    confidence = self._calculate_confidence(logprobs_raw)
                    
                    if new_token_ids:
                        all_generated_segments.append({
                            "ids": new_token_ids,
                            "source": 0, # 0: small
                            "text": generated_text
                        })
                        self.stats["tokens_generated_by_small"] += len(new_token_ids)
                        
                        # 仅当置信度低时标记切换
                        if confidence < self.confidence_threshold:
                            force_large_model_this_step = True
                    else:
                        force_large_model_this_step = True
                else:
                    force_large_model_this_step = True

            # ================= 分支 B: 大模型执行 =================
            else:
                self.stats["fallback_to_large"] += 1
                # 统计 Prefill 长度 (用于分析加速效果)
                ctx_ids = self.tokenizer_large.apply_chat_template(conversation_history, tokenize=True, add_generation_prompt=False)
                self.stats["prefill_tokens_count"] += len(ctx_ids)
                
                large_res = self._call_vllm_api(
                    url=self.large_url,
                    model_name=self.large_model_name,
                    prompt_text=current_prompt_text,
                    max_tokens=self.max_step_tokens,
                    stop_sequences=stop_sequences,
                    get_logprobs=False,
                    temperature=0.7
                )
                
                if large_res and large_res.get("choices"):
                    choice = large_res["choices"][0]
                    generated_text = choice.get("text", "")
                    
                    new_token_ids = choice.get("token_ids")
                    if not new_token_ids and generated_text:
                        new_token_ids = self.tokenizer_large.encode(generated_text, add_special_tokens=False)
                    
                    if new_token_ids:
                        all_generated_segments.append({
                            "ids": new_token_ids,
                            "source": 1, # 1: large
                            "text": generated_text
                        })
                        self.stats["tokens_generated_by_large"] += len(new_token_ids)
                    else:
                        break # 大模型也没生成，退出
                else:
                    break # 大模型调用失败，退出

            # ================= 更新对话历史 & 检查结束条件 =================
            if generated_text:
                conversation_history.append({"role": "assistant", "content": generated_text})
                
                # 检查 step 标签
                if self.step_tag in generated_text:
                    step_count += 1
                
                # 检查自然结束 (没有 step 标签且有内容，或者包含答案关键词)
                is_natural_end = (self.step_tag not in generated_text) and len(generated_text.strip()) > 0
                has_answer = "Answer:" in generated_text or "final answer" in generated_text.lower()
                
                if has_answer or (is_natural_end and not force_large_model_this_step):
                    break
            else:
                break 

        return self._decode_mixed_segments(all_generated_segments)

    def _apply_template_safe(self, messages: List[Dict[str, str]]) -> str:
        """安全地应用模板，防止多重模板"""
        # 始终使用大模型的 tokenizer 来渲染 prompt，保证格式一致性
        # 假设输入 messages 是纯文本内容，不包含特殊 token
        try:
            prompt = self.tokenizer_large.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            return prompt
        except Exception as e:
            # 降级处理：手动拼接 (仅作为最后手段)
            text = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                text += f"{role}: {content}\n"
            text += "assistant: "
            return text

    def apply_chat_template(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        lm_eval 框架调用的标准接口，用于将消息列表转换为带模板的 Prompt。
        这对于构建 Few-Shot 示例至关重要。
        """
        # 使用大模型的 tokenizer 进行模板化，保证格式统一
        # 注意：fewshot 构建时通常不需要 add_generation_prompt=True，除非特定任务要求
        # 这里默认 False 以兼容 fewshot 上下文拼接，生成时由 _run_speculative_routing 控制
        add_generation_prompt = kwargs.get('add_generation_prompt', False)
        
        try:
            return self.tokenizer_large.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=add_generation_prompt
            )
        except Exception as e:
            # 降级处理：如果 tokenizer 报错，手动拼接
            print(f"⚠️ apply_chat_template 失败: {e}，使用手动拼接")
            text = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                # 简单的通用格式，具体可根据模型调整
                if role == "system":
                    text += f"System: {content}\n"
                elif role == "user":
                    text += f"User: {content}\n"
                elif role == "assistant":
                    text += f"Assistant: {content}\n"
            if add_generation_prompt:
                text += "Assistant: "
            return text

    def _decode_mixed_segments(self, segments: List[Dict]) -> str:
        if not segments:
            return ""
            
        final_text_parts = []
        for seg in segments:
            source = seg["source"]
            ids = seg["ids"]
            # 注意：这里解码时 skip_special_tokens=False 以保留可能的格式，
            # 但如果只需要纯文本回答，可以设为 True。根据 step_tag 需求，建议 False 或手动处理
            if source == 0:
                text = self.tokenizer_small.decode(ids, skip_special_tokens=False)
            else:
                text = self.tokenizer_large.decode(ids, skip_special_tokens=False)
            final_text_parts.append(text)
            
        return "".join(final_text_parts)

    def generate_until(self, requests: List) -> List[str]:
        results = []
        pbar = tqdm(requests, desc="Speculative Routing")

        for req in pbar:
            # 1. 解析请求
            if isinstance(req, Instance):
                context, gen_kwargs = req.args
            else:
                try:
                    context, gen_kwargs = req
                except ValueError:
                    raise TypeError(f"Invalid request format: {type(req)}")

            self.stats["total_requests"] += 1
            
            # 2. 处理停止符
            stop_sequences = gen_kwargs.get("until", []) 
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]
            
            if not stop_sequences:
                stop_sequences = [self.step_tag]
            elif self.step_tag not in stop_sequences:
                stop_sequences.append(self.step_tag)

            # 3. 【修复点】不再进行去模板化操作
            # lm_eval 传入的 context 通常已经是纯文本问题。
            # 我们直接将其放入 user message 中，由 _run_speculative_routing 内部统一加模板。
            clean_context = context
            
            # 构建 Messages 结构
            final_messages = []
            if self.system_prompt:
                final_messages.append({"role": "system", "content": self.system_prompt})
            
            final_messages.append({"role": "user", "content": clean_context})

            # 4. 调用路由
            try:
                generated_text = self._run_speculative_routing(final_messages, stop_sequences)
                results.append(generated_text)
            except Exception as e:
                print(f"\n❌ Error processing request: {e}")
                results.append("")

        pbar.close()
        
        # 5. 打印最终统计
        total = self.stats["total_requests"]
        if total > 0:
            print("\n" + "="*40)
            print("Routing Statistics:")
            print(f"   Total Requests: {total}")
            print(f"   Fallbacks to Large: {self.stats['fallback_to_large']}")
            
            small_tokens = self.stats['tokens_generated_by_small']
            large_tokens = self.stats['tokens_generated_by_large']
            total_gen = small_tokens + large_tokens
            
            print(f"   Small Model Tokens: {small_tokens}")
            print(f"   Large Model Tokens: {large_tokens}")
            if total_gen > 0:
                print(f"   Small Model Ratio: {small_tokens/total_gen:.2%}")
            print("="*40)
            
        return results

    def loglikelihood(self, requests: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Speculative Routing supports 'generate_until' only.")

    def loglikelihood_rolling(self, requests: List[Tuple[str, str]]) -> List[float]:
        raise NotImplementedError("Not implemented.")