import os
import requests
import math
import time
import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoTokenizer 
import re
from collections import Counter

# ================= 优化后的 System Prompt =================
SYSTEM_PROMPT = (
    "You are an expert mathematician solving a problem step-by-step.\n"
    "CRITICAL RULES:\n"
    "1. NEVER restart the solution. NEVER say 'Let's solve', 'Okay', 'Alright', or restate the problem statement in subsequent steps.\n"
    "2. If you see a '</step>' tag or a system instruction to continue, you MUST pick up exactly where the previous thought left off.\n"
    "3. Your output should be a continuous flow of logic. Step 1 starts the logic, Step 2 continues it, etc.\n"
    "4. Only output the final answer in \\boxed{} format when the solution is fully derived.\n"
    "5. If interrupted, assume the user wants you to continue the calculation, not start over."
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@register_model("vllm_router")
class VLLMRouter(LM):
    def __init__(
        self,
        small_url: str,
        large_url: str,
        small_model_name: str,
        large_model_name: str,
        confidence_threshold: float = 0.85,
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

        self.max_steps = max_gen_tokens // max_step_tokens + 1
        try:
            self.confidence_threshold = float(confidence_threshold)
        except (ValueError, TypeError):
            self.confidence_threshold = 0.85

        self.max_step_tokens = max_step_tokens
        self.max_gen_tokens = max_gen_tokens
        self.step_tag = step_tag
        self.system_prompt = SYSTEM_PROMPT

        # 定义常见的 EOS 标记列表，用于清洗
        self.eos_tokens = [
            "<｜end▁of▁sentence｜>", # Qwen / DeepSeek 常见
            "</s>",                 # Llama / Mistral 常见
            "<|endoftext|>",        # GPT-2 / Others
            "[END]",                # 某些自定义模型
            "<eos>"                 # 通用
        ]

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

    def _clean_eos(self, text: str) -> str:
        """
        强力清洗文本末尾的所有 EOS 标记。
        """
        cleaned_text = text
        for eos in self.eos_tokens:
            # 使用 while 确保移除所有连续出现的 EOS (虽然通常只有一个)
            while cleaned_text.rstrip().endswith(eos):
                cleaned_text = cleaned_text.rstrip()[:-len(eos)]
        return cleaned_text

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
        top_k_logprobs: int = 5,
        repetition_penalty: float = 1.15 # 新增：默认开启重复惩罚
    ) -> Optional[Dict]:
        
        endpoint = f"{url.rstrip('/')}/v1/completions"

        payload = {
            "model": model_name,
            "prompt": prompt_text,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty, # 应用重复惩罚
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
                if response.status_code == 400:
                    err_text = response.text[:150]
                    if "logprobs" in err_text and "max_logprobs" in err_text:
                        logger.warning(f"⚠️ API Error 400: logprobs limit exceeded.")
                    else:
                        logger.warning(f"⚠️ API Error 400: {err_text}")
                else:
                    logger.warning(f"⚠️ API Error {response.status_code}: {response.text[:100]}")
                return None

        except Exception as e:
            logger.error(f"⚠️ Network Error: {type(e).__name__} - {e}")
            return None

    def _calculate_confidence(self, logprobs_data) -> float:
        if not logprobs_data:
            return 0.0
        
        probs = []
        if isinstance(logprobs_data, dict):
            token_logprobs_list = logprobs_data.get("token_logprobs", [])
            if token_logprobs_list and all(isinstance(x, (int, float)) for x in token_logprobs_list):
                valid_logs = [x for x in token_logprobs_list if x is not None]
                if valid_logs:
                    probs = [math.exp(float(x)) for x in valid_logs]
        
        return sum(probs) / len(probs) if probs else 0.0

    def _check_repetition(self, text: str, threshold: float = 0.5) -> bool:
        """检测文本是否陷入死循环复读"""
        sentences = re.split(r'[。\n]', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        if not sentences:
            return False
        counter = Counter(sentences)
        most_common_count = counter.most_common(1)[0][1]
        if most_common_count / len(sentences) > threshold:
            logger.warning(f"🔴 检测到复读！最高频句子重复 {most_common_count} 次")
            return True
        return False

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
        
    def _run_speculative_routing(self, messages: List[Dict[str, str]], stop_sequences: List[str]) -> str:
        # 【核心策略】维护单一纯文本流 current_raw_text
        # 1. 初始化：仅对第一步应用 Chat Template
        # 注意：即使第一步用小模型推理，我们也建议用大模型的 tokenizer 来构建初始 Prompt
        # 这样可以保证后续无论切换哪个模型，编码基准一致。
        # 如果大小模型 vocab 完全不同，此方案需调整为：每次发送前用对应模型 tokenizer 重新 encode(当前文本)
        
        print(f"🚀 [Routing] 启动 | 最大步数: {self.max_steps} | 阈值: {self.confidence_threshold}", flush=True)
        
        # 使用大模型 tokenizer 渲染初始模板 (作为 Ground Truth)
        try:
            initial_model_type = "large" # 默认用大模型渲染初始 prompt，或者根据配置
            current_raw_text = self._get_initial_prompt_text(messages, initial_model_type)
        except Exception as e:
            print(f"❌ 模板渲染失败: {e}", flush=True)
            # Fallback
            current_raw_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant: "

        current_raw_text = self._clean_eos(current_raw_text)
        
        all_generated_segments = [] 
        step_count = 0
        force_large_model_this_step = False
        
        print(f"📝 初始 Prompt 长度: {len(current_raw_text)} chars", flush=True)

        while step_count < self.max_steps:
            current_step = step_count + 1
            use_large_model = force_large_model_this_step
            force_large_model_this_step = False
            
            # 决定当前使用的模型配置
            tokenizer_used = self.tokenizer_large if use_large_model else self.tokenizer_small
            url_used = self.large_url if use_large_model else self.small_url
            model_name = self.large_model_name if use_large_model else self.small_model_name

            # 【关键】编码输入
            # 无论用哪个模型推理，我们都基于当前的 current_raw_text 编码
            # 如果大小模型 vocab 不共享，这里需要：
            # temp_text = self.tokenizer_large.decode(self.tokenizer_large.encode(current_raw_text)) 
            # 然后由小模型 encode。这里假设 vocab 兼容或直接传文本给 API (如果 API 支持 prompt 字符串)
            # 你的 _call_vllm_api 似乎支持 prompt_token_ids。
            
            # 为了绝对安全，我们总是用当前推理模型的 tokenizer 来编码当前文本
            # 这样可以避免跨 tokenizer 的分词误差
            input_ids = tokenizer_used.encode(current_raw_text, add_special_tokens=False)
            
            if len(input_ids) > 32000: # 简单保护，防止超长
                print(f"⚠️ 上下文过长 ({len(input_ids)})，截断尾部...", flush=True)
                input_ids = input_ids[-30000:] # 简单截断策略，实际应更智能

            if use_large_model:
                self.stats["prefill_tokens_count"] += len(input_ids)
                print(f"🔸 Step {current_step} [大]: 输入 Tokens={len(input_ids)}", flush=True)
            else:
                print(f"🔹 Step {current_step} [小]: 输入 Tokens={len(input_ids)}", flush=True)

            generated_text = ""
            new_token_ids = []
            finish_reason = ""

            # ================= 分支 A: 小模型执行 =================
            if not use_large_model:
                small_res = self._call_vllm_api(
                    url=url_used,
                    model_name=model_name,
                    prompt_token_ids=input_ids,
                    max_tokens=self.max_step_tokens,
                    stop_sequences=stop_sequences,
                    get_logprobs=True,
                    temperature=0.7,
                    repetition_penalty=1.15
                )
                
                if small_res and "choices" in small_res:
                    choice = small_res["choices"][0]
                    generated_text = choice.get("text", "")
                    finish_reason = choice.get("finish_reason", "")
                    new_token_ids = choice.get("token_ids", [])
                    
                    # 容错：如果 API 没返回 token_ids，本地补编码
                    if not new_token_ids and generated_text:
                        new_token_ids = tokenizer_used.encode(generated_text, add_special_tokens=False)
                    
                    logprobs_raw = choice.get("logprobs", None)
                    confidence = self._calculate_confidence(logprobs_raw)
                    
                    print(f"   🟢 小模型完成 | 置信度: {confidence:.3f} | 原因: {finish_reason or 'EOS'}", flush=True)

                    if not new_token_ids:
                        print("   ❌ 小模型未生成有效 tokens，切换大模型。", flush=True)
                        force_large_model_this_step = True

                else:
                    print("   ❌ 小模型调用失败或返回格式错误，切换大模型。", flush=True)
                    force_large_model_this_step = True

            # ================= 分支 B: 大模型执行 =================
            else:
                self.stats["fallback_to_large"] += 1
                
                large_res = self._call_vllm_api(
                    url=url_used,
                    model_name=model_name,
                    prompt_token_ids=input_ids,
                    max_tokens=self.max_step_tokens,
                    stop_sequences=stop_sequences,
                    get_logprobs=True,
                    temperature=0.7,
                    repetition_penalty=1.15
                )
                
                if large_res and "choices" in large_res:
                    choice = large_res["choices"][0]
                    generated_text = choice.get("text", "")
                    finish_reason = choice.get("finish_reason", "")
                    new_token_ids = choice.get("token_ids", [])

                    if not new_token_ids and generated_text:
                        new_token_ids = tokenizer_used.encode(generated_text, add_special_tokens=False)
                    
                    logprobs_raw = choice.get("logprobs", None)
                    confidence = self._calculate_confidence(logprobs_raw)

                    print(f"   🟢 大模型完成 | 置信度: {confidence:.3f} | 原因: {finish_reason or 'EOS'}", flush=True)

                    if not new_token_ids:
                        print("   ❌ 大模型未生成有效 tokens，终止。", flush=True)
                        break
                else:
                    print("   ❌ 大模型调用失败，终止。", flush=True)
                    break

            # ================= 【核心逻辑】更新纯文本上下文 =================
            if new_token_ids:
                # 1. 记录生成片段
                all_generated_segments.append({
                    "ids": new_token_ids,
                    "source": 1 if use_large_model else 0,
                    "text": generated_text
                })
                
                if use_large_model:
                    self.stats["tokens_generated_by_large"] += len(new_token_ids)
                else:
                    self.stats["tokens_generated_by_small"] += len(new_token_ids)

                # 2. 解码新生成的内容 (用于拼接)
                # 必须使用生成时的 tokenizer 解码，以保证 round-trip 一致性
                decoded_segment = tokenizer_used.decode(new_token_ids, skip_special_tokens=True)
                
                # 3. 【重要】清洗 decoded_segment 末尾的 EOS 标记
                # 防止下次 prompt 里出现奇怪的空白或截断符
                clean_segment = self._clean_eos(decoded_segment)
                
                # 4. 拼接到全局上下文
                current_raw_text += clean_segment
                
                step_count += 1
                print(f"   🔄 已拼接文本 (当前总长: {len(current_raw_text)} chars)", flush=True)

                # 5. 检查终止条件
                # 检查答案
                if "\\boxed{" in clean_segment or "oxed{" in clean_segment: # 兼容可能的转义问题
                    print("   ✅ 检测到最终答案 (\\boxed)，任务完成。", flush=True)
                    break
                
                # 检查是否触发了 step_tag (虽然 API 会截断，但文本里可能没有，靠 finish_reason 判断)
                # 如果 finish_reason 是 stop 且 stop 序列里包含 step_tag，说明一个逻辑步结束了
                # 这里我们依靠外层循环继续，只要没答案就继续生成
                if finish_reason == "length":
                    print("   ⏱️ 达到单步最大长度，自动继续下一步。", flush=True)
                
                # 决策下一步模型
                if not use_large_model and confidence < self.confidence_threshold:
                    print(f"   ⚠️ 置信度 ({confidence:.3f}) 低于阈值，*下一步* 切换大模型。", flush=True)
                    force_large_model_this_step = True
                elif use_large_model and confidence < self.confidence_threshold:
                    print(f"   ⚠️ 大模型置信度 ({confidence:.3f}) 仍低，*下一步* 继续大模型。", flush=True)
                    force_large_model_this_step = True
                else:
                    # 默认回小模型 (如果上一步是大模型且置信度高)
                    if use_large_model:
                        print("   ⏩ 下一步恢复小模型推理。", flush=True)
                    else:
                        print("   ⏩ 下一步继续小模型推理。", flush=True)

            else:
                # 没有生成任何 token
                print("   ⛔ 无生成内容，循环终止。", flush=True)
                break

        print(f"✅ [Routing] 结束。总步数: {step_count}", flush=True)
        return self._decode_mixed_segments(all_generated_segments)

    def _apply_template_safe(self, messages: List[Dict[str, str]], is_first_step: bool = False) -> str:
        """
        应用模板，并确保彻底清除末尾的 EOS 标记。
        """
        try:
            # 1. 渲染历史消息 (不加 generation prompt)
            prompt = self.tokenizer_large.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            # 2. 【强力清洗】移除末尾所有可能的 EOS 标记
            # 无论 tokenizer 是否自动添加了 EOS，这里都强制移除
            prompt = self._clean_eos(prompt)
            
            # 3. 动态添加后缀
            if is_first_step:
                # === 第一步：冷启动 ===
                prompt += "<｜Assistant｜> <think>\n"
                logger.debug("🔹 [Template] Step 1: 添加冷启动标签")
            else:
                # === 第二步及以后：热续写 ===
                # 只加一个换行，确保模型认为自己在续写同一句话
                # 绝对不要加 <｜Assistant｜> 或 <think>
                prompt += "\n"
                logger.debug("🔗 [Template] Step N: 热续写模式 (无新标签)")
            
            return prompt
            
        except Exception as e:
            logger.error(f"Template error: {e}")
            # Fallback
            text = ""
            for msg in messages:
                content = self._clean_eos(msg['content']) # Fallback 中也清洗
                text += f"{msg['role']}: {content}\n"
            
            if is_first_step:
                return text + "assistant: <think>\n"
            else:
                return text + "\n"

    def apply_chat_template(self, messages: List[Dict[str, str]], **kwargs) -> str:
        add_generation_prompt = kwargs.get('add_generation_prompt', False)
        try:
            return self.tokenizer_large.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=add_generation_prompt
            )
        except Exception:
            return "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    def _decode_mixed_segments(self, segments: List[Dict]) -> str:
        if not segments:
            return ""
        final_text_parts = []
        for seg in segments:
            source = seg["source"]
            ids = seg["ids"]
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
            if isinstance(req, Instance):
                context, gen_kwargs = req.args
            else:
                context, gen_kwargs = req

            self.stats["total_requests"] += 1
            
            stop_sequences = gen_kwargs.get("until", []) 
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]
            
            if self.step_tag not in stop_sequences:
                stop_sequences.append(self.step_tag)
            
            clean_context = context
            
            final_messages = []
            if self.system_prompt:
                final_messages.append({"role": "system", "content": self.system_prompt})
            
            format_instruction = (
                f"\n\nIMPORTANT: Solve this step-by-step. "
                f"After EACH logical step, output '{self.step_tag}' on a new line. "
                f"Do not output the final answer until the end. "
                f"Start thinking now:"
            )
            final_messages.append({"role": "user", "content": clean_context + format_instruction})

            try:
                generated_text = self._run_speculative_routing(final_messages, stop_sequences)
                results.append(generated_text)
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                results.append("")

        pbar.close()
        
        total = self.stats["total_requests"]
        if total > 0:
            print("\n" + "="*40)
            print("Routing Statistics:")
            print(f"   Total Requests: {total}")
            print(f"   Fallbacks: {self.stats['fallback_to_large']}")
            s_tok = self.stats['tokens_generated_by_small']
            l_tok = self.stats['tokens_generated_by_large']
            print(f"   Tokens (Small/Large): {s_tok} / {l_tok}")
            if s_tok + l_tok > 0:
                print(f"   Small Ratio: {s_tok/(s_tok+l_tok):.2%}")
            print("="*40)
            
        return results

    def loglikelihood(self, requests):
        raise NotImplementedError("Speculative Routing supports 'generate_until' only.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("Not implemented.")