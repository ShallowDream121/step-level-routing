import os
import requests
import math
import time
import json
from typing import List, Optional, Dict, Any, Tuple
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoTokenizer 

SYSTEM_PROMPT = (
    "You are a reasoning assistant. You must solve the problem step by step. "
    "CRITICAL REQUIREMENT: After completing EACH logical step, you MUST output the tag '</step>' on a new line. "
    "Do not output the final answer until you have gone through all necessary steps. "
    "CONTINUATION RULE: If the context ends mid-sentence (no '</step>' at the end): "
    "1. DO NOT restart, summarize, or say 'continuing'. "
    "2. IMMEDIATELY complete the sentence/formula from the exact cutoff point. "
    "3. Only output '</step>' once the step is fully finished."
    "Example format:\n<think>\\nStep 1 analysis...\\n</step>\\nStep 2 calculation...\\n</step>\\n...\\nTherefore, the answer is \\boxed{204}."
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
        self.confidence_threshold = confidence_threshold
        self.max_step_tokens = max_step_tokens
        self.max_gen_tokens = max_gen_tokens
        self.step_tag = step_tag

        self.system_prompt = SYSTEM_PROMPT

        print(f"🚀 正在加载小模型 Tokenizer ({small_model_name})...")
        self.tokenizer_small = AutoTokenizer.from_pretrained(
            small_model_name,
            trust_remote_code=True 
        )
        if self.tokenizer_small.pad_token is None:
            self.tokenizer_small.pad_token = self.tokenizer_small.eos_token

        print(f"🚀 正在加载大模型 Tokenizer ({large_model_name})...")
        self.tokenizer_large = AutoTokenizer.from_pretrained(
            large_model_name,
            trust_remote_code=True
        )
        if self.tokenizer_large.pad_token is None:
            self.tokenizer_large.pad_token = self.tokenizer_large.eos_token
            
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
        self.stats = {
            "total_requests": 0,
            "small_only_success": 0,
            "fallback_to_large": 0,
            "small_failures": 0,
            "tokens_generated_by_small": 0,
            "tokens_generated_by_large": 0,
            "prefill_tokens_count": 0
        }

    def _call_vllm_api(
        self, 
        url: str, 
        prompt_text: Optional[str] = None,
        prompt_token_ids: Optional[List[int]] = None,
        max_tokens: int = 256, 
        stop_sequences: List[str] = None,
        get_logprobs: bool = True,
        temperature: float = 0.7
    ) -> Optional[Dict]:

        payload = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "stop": stop_sequences,
            "stream": False
        }
        
        # 优先发送 token_ids 以避免二次编码误差，如果没有则发送 text
        if prompt_token_ids is not None:
            payload["prompt_token_ids"] = prompt_token_ids
            payload["prompt"] = "" 
        elif prompt_text is not None:
            payload["prompt"] = prompt_text
        else:
            raise ValueError("Must provide either prompt_text or prompt_token_ids")

        if get_logprobs:
            payload["logprobs"] = 5
        
        try:
            response = self.session.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ API 请求失败 ({url}): {e}")
            return None

    def _calculate_confidence(self, logprobs_data: List[Dict]) -> float:
        if not logprobs_data:
            print("DEBUG: NO DATA!!!",flush = True)
            return 0.0
        
        probs = []
        for token_logprobs in logprobs_data:
            if not token_logprobs:
                continue
            if isinstance(token_logprobs, dict):
                info = next(iter(token_logprobs.values()), None) if token_logprobs else None
                if info and isinstance(info, dict) and "logprob" in info:
                    logprob = info["logprob"]
                    if logprob is not None:
                        probs.append(math.exp(logprob))
                elif isinstance(token_logprobs, (int, float)):
                    probs.append(math.exp(token_logprobs))

        if not probs:
            print("DEBUG: NO PROBS!!!",flush = True)
            return 0.0
        print("sum = {sum(probs)}",flush = True)
        return sum(probs) / len(probs)
    
    def _get_initial_prompt_text(self, messages: List[Dict[str, str]], model_type: str) -> str:
        """
        仅在第一步应用聊天模板，生成初始的纯文本前缀。
        """
        tokenizer = self.tokenizer_small if model_type == "small" else self.tokenizer_large
        try:
            # apply_chat_template 返回的是包含 System/User/Assistant 标记的完整字符串
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True # 第一步需要生成提示符 (如 <|assistant|>)
            )
            # 重要：去除末尾可能的空白，但不要去除助手起始标记
            return prompt.rstrip()
        except Exception as e:
            print(f"⚠️ 应用聊天模板失败: {e}。回退到纯文本拼接。")
            text = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                text += f"{role}: {content}\n"
            text += "assistant: "
            return text

    def _decode_tokens(self, token_ids: List[int], model_type: str = "small", skip_special_tokens: bool = True) -> str:
        tokenizer = self.tokenizer_small if model_type == "small" else self.tokenizer_large
        if not token_ids:
            return ""
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def _run_speculative_routing(self, messages: List[Dict[str, str]], stop_sequences: List[str]) -> str:
        # 【核心修改】不再使用 conversation_history 字典列表
        # 而是维护一个不断增长的纯文本字符串 current_raw_text
        
        # 1. 初始化：仅对第一步应用 Chat Template
        # 假设 messages 包含 system 和 user
        initial_model_type = "small" # 默认用小模型渲染初始 prompt，或者根据配置
        current_raw_text = self._get_initial_prompt_text(messages, initial_model_type)
        
        all_generated_segments = [] 
        max_steps = int(self.max_gen_tokens / self.max_step_tokens) + 5 # 增加一点缓冲
        step_count = 0
        force_large_model_this_step = False

        print(f"📝 初始 Prompt 长度: {len(current_raw_text)} chars")

        while step_count < max_steps:
            current_step = step_count + 1
            use_large_model = force_large_model_this_step
            force_large_model_this_step = False
            current_model_type = "large" if use_large_model else "small"
            
            tokenizer_used = self.tokenizer_large if use_large_model else self.tokenizer_small
            url_used = self.large_url if use_large_model else self.small_url

            # 2. 编码当前的纯文本前缀
            # 注意：这里直接 encode 整个 current_raw_text，不再套模板
            input_ids = tokenizer_used.encode(current_raw_text, add_special_tokens=False)
            
            if use_large_model:
                self.stats["prefill_tokens_count"] += len(input_ids)

            generated_text = ""
            new_token_ids = []

            # ================= 分支 A: 小模型执行 =================
            if not use_large_model:
                small_res = self._call_vllm_api(
                    url=url_used,
                    prompt_token_ids=input_ids, # 直接发送 IDs
                    max_tokens=self.max_step_tokens,
                    stop_sequences=stop_sequences,
                    get_logprobs=True,
                    temperature=0.7
                )
                
                if small_res:
                    output_data = small_res.get("outputs", [{}])[0] if "outputs" in small_res else small_res
                    generated_text = output_data.get("text", "")
                    
                    new_token_ids = output_data.get("token_ids", None)
                    # 如果 API 没返回 token_ids，尝试本地编码 (不推荐，可能有差异，但在 vLLM 中通常会返回)
                    if new_token_ids is None and generated_text:
                        new_token_ids = tokenizer_used.encode(generated_text, add_special_tokens=False)
                    
                    logprobs_raw = output_data.get("logprobs", [])
                    confidence = self._calculate_confidence(logprobs_raw)
                    
                    if new_token_ids:
                        all_generated_segments.append({
                            "ids": new_token_ids,
                            "source": 0,
                            "text": generated_text
                        })
                        self.stats["tokens_generated_by_small"] += len(new_token_ids)
                        print(f"   [Step {current_step}] 🟢 小模型生成 (置信度: {confidence:.2f}) -> {len(new_token_ids)} tokens")

                        if confidence < self.confidence_threshold:
                            print(f"   ⚠️ 置信度低，标记 *下一步* 切换到大模型。")
                            force_large_model_this_step = True
                    else:
                         print(f"   ❌ 小模型未生成有效内容。")
                         force_large_model_this_step = True
                else:
                    print(f"   ❌ 小模型调用失败。")
                    force_large_model_this_step = True

            # ================= 分支 B: 大模型执行 =================
            else:
                self.stats["fallback_to_large"] += 1
                
                large_res = self._call_vllm_api(
                    url=url_used,
                    prompt_token_ids=input_ids,
                    max_tokens=self.max_step_tokens,
                    stop_sequences=stop_sequences,
                    get_logprobs=True,
                    temperature=0.7
                )
                
                if large_res:
                    output_data = large_res.get("outputs", [{}])[0] if "outputs" in large_res else large_res
                    generated_text = output_data.get("text", "")
                    
                    new_token_ids = output_data.get("token_ids", None)
                    if new_token_ids is None and generated_text:
                        new_token_ids = tokenizer_used.encode(generated_text, add_special_tokens=False)
                    
                    logprobs_raw = output_data.get("logprobs", [])
                    confidence = self._calculate_confidence(logprobs_raw)

                    if new_token_ids:
                        all_generated_segments.append({
                            "ids": new_token_ids,
                            "source": 1,
                            "text": generated_text
                        })
                        self.stats["tokens_generated_by_large"] += len(new_token_ids)
                        print(f"   [Step {current_step}] 🟢 大模型生成 (置信度: {confidence:.2f}) -> {len(new_token_ids)} tokens")

                        if confidence < self.confidence_threshold:
                            print(f"   ⚠️ 置信度低，标记 *下一步* 继续大模型。")
                            force_large_model_this_step = True
                        
                    else:
                        print("   ❌ 大模型未生成有效内容。")
                        break
                else:
                    print("   ❌ 大模型也失败了，终止。")
                    break

            # ================= 更新纯文本上下文 (关键逻辑) =================
            if new_token_ids:
                # 1. 将新生成的 tokens 解码为文本
                # 注意：必须使用与生成时相同的 tokenizer，且 skip_special_tokens=False 
                # 以保留必要的格式字符（如果有的话），但通常生成内容不含特殊符
                decoded_segment = tokenizer_used.decode(new_token_ids, skip_special_tokens=True)
                
                # 2. 拼接到全局纯文本上下文中
                # 这样下一次循环时，input_ids 就是 "上文 + 本步生成" 的完整编码
                current_raw_text += decoded_segment
                
                step_count += 1
                
                # 检查是否自然结束或包含答案
                has_answer = "//boxed{" in decoded_segment or "\\boxed{" in decoded_segment
                # 检查是否包含了 step_tag，如果包含且是完整步骤，可以选择继续或停止
                # 这里逻辑保持原样：只要有输出就继续，直到达到最大步数或出现答案
                if has_answer:
                    print("   ✅ 检测到最终答案，停止生成。")
                    break
            else:
                # 没有生成任何 token，强制退出
                break 

        return self._decode_mixed_segments(all_generated_segments)

    def _decode_mixed_segments(self, segments: List[Dict]) -> str:
        if not segments:
            return ""
            
        final_text_parts = []
        for seg in segments:
            source = seg["source"]
            ids = seg["ids"]
            
            if source == 0:
                text = self.tokenizer_small.decode(ids, skip_special_tokens=True)
            else:
                text = self.tokenizer_large.decode(ids, skip_special_tokens=True)
            
            final_text_parts.append(text)
            
        return "".join(final_text_parts)

    def generate_until(self, requests: List[Tuple[str, dict]]) -> List[str]:
        results = []
        for req in tqdm(requests, desc="Speculative Routing (Pure Text Mode)"):
            context, gen_kwargs = req
            self.stats["total_requests"] += 1
            
            stop_sequences = gen_kwargs.get("stop", [self.step_tag])
            if self.step_tag not in stop_sequences:
                stop_sequences.append(self.step_tag)
            
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            
            messages.append({"role": "user", "content": context})
            
            generated_text = self._run_speculative_routing(messages, stop_sequences)
            results.append(generated_text)
            
        total = self.stats["total_requests"]
        if total > 0:
            print("\n" + "="*40)
            print("路由统计:")
            print(f"   总请求数: {total}")
            print(f"   切换到大模型次数: {self.stats['fallback_to_large']}")
            print(f"   累计 Prefill 加速 Token 数: {self.stats['prefill_tokens_count']}")
            if self.stats['fallback_to_large'] > 0:
                avg_ctx_len = self.stats['prefill_tokens_count'] / self.stats['fallback_to_large']
                print(f"   平均每次切换的上下文长度: {avg_ctx_len:.1f} tokens")

            small_tokens = self.stats['tokens_generated_by_small']
            large_tokens = self.stats['tokens_generated_by_large']
            total_gen_tokens = small_tokens + large_tokens
            print(f"   小模型生成 Token 数: {small_tokens}")
            print(f"   大模型生成 Token 数: {large_tokens}")
            if total_gen_tokens > 0:
                print(f"   小模型承担比例: {small_tokens/total_gen_tokens:.2%}")
            print("="*40)
            
        return results

    def loglikelihood(self, requests: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Speculative Routing with Prefill currently supports 'generate_until' tasks only.")

    def loglikelihood_rolling(self, requests: List[Tuple[str, str]]) -> List[float]:
        raise NotImplementedError("Not implemented.")