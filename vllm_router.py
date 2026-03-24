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
        confidence_threshold: float = 0.6,
        max_step_tokens: int = 1024,
        max_gen_tokens: int = 16384,
        step_tag: str = "</step>",
        **kwargs
    ):
        super().__init__()
        self.small_url = small_url
        self.large_url = large_url
        self.small_model_name = small_model_name
        self.large_model_name = large_model_name

        self.max_steps = 20
        try:
            self.confidence_threshold = float(confidence_threshold)
        except (ValueError, TypeError):
            self.confidence_threshold = 0.6

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

    def _run_speculative_routing(self, messages: List[Dict[str, str]], stop_sequences: List[str]) -> str:
        logger.info(f"🚀 [Routing] 开始执行 (动态截断+人工补全模式), 最大步数: {self.max_steps}, 阈值: {self.confidence_threshold}")
        
        conversation_history = list(messages)
        all_generated_segments = [] 
        
        step_count = 0
        force_large_model_this_step = False 
        current_step = 0
        
        last_generated_prefix = "" 
        repeat_counter = 0

        while step_count < self.max_steps:
            current_step = step_count + 1
            use_large_model = force_large_model_this_step
            force_large_model_this_step = False
            
            # 构建 Prompt
            current_prompt_text = self._apply_template_safe(conversation_history, current_step == 1)
    
            if current_step <= 3:
                logger.info(f"🔍 [DEBUG] Step {current_step} 发送给模型的完整 Prompt (最后 1000 字符):")
                logger.info("-" * 50)
                debug_tail = current_prompt_text[-1000:] if len(current_prompt_text) > 1000 else current_prompt_text
                # 检查末尾是否有 EOS
                if any(debug_tail.endswith(eos) for eos in self.eos_tokens):
                    logger.warning("⚠️ [DEBUG] 警告：Prompt 末尾检测到 EOS 标记！这会导致思维中断。")
                logger.info(debug_tail.replace("\n", "\\n"))
                logger.info("-" * 50)
                logger.info(f"🔍 [DEBUG] conversation_history 长度: {len(conversation_history)} 条消息")
                if len(conversation_history) > 2:
                    logger.info(f"🔍 [DEBUG] 倒数第二条消息 (Assistant): {conversation_history[-2]['content'][:100]}...")
            
            generated_text = ""
            new_token_ids = []
            finish_reason = ""

            if not use_large_model:
                logger.debug(f"🔹 Step {current_step}: 调用小模型 (MaxTokens={self.max_step_tokens})...")
                
                small_res = self._call_vllm_api(
                    url=self.small_url,
                    model_name=self.small_model_name,
                    prompt_text=current_prompt_text,
                    max_tokens=self.max_step_tokens,
                    stop_sequences=stop_sequences,
                    get_logprobs=True,
                    temperature=0.7,
                    repetition_penalty=1.15
                )
                
                if small_res and small_res.get("choices"):
                    choice = small_res["choices"][0]
                    generated_text = choice.get("text", "")
                    finish_reason = choice.get("finish_reason", "")
                    
                    new_token_ids = choice.get("token_ids")
                    if not new_token_ids and generated_text:
                        new_token_ids = self.tokenizer_small.encode(generated_text, add_special_tokens=False)
                    
                    logprobs_raw = choice.get("logprobs", [])
                    confidence = self._calculate_confidence(logprobs_raw)
                    
                    preview_text = generated_text[:60].replace("\n", "\\n") if generated_text else ""
                    logger.info(f"🔹 Step {current_step} [小]: 置信度={confidence:.3f} | 结束原因: {finish_reason or 'EOS'} | 预览: '{preview_text}...'")

                    if new_token_ids:
                        all_generated_segments.append({
                            "ids": new_token_ids, "source": 0, "text": generated_text
                        })
                        self.stats["tokens_generated_by_small"] += len(new_token_ids)
                        
                        if confidence < self.confidence_threshold:
                            logger.warning(f"⚠️ Step {current_step}: 置信度低 ({confidence:.3f})，下一步切换大模型。")
                            force_large_model_this_step = True
                    else:
                        logger.warning(f"⚠️ Step {current_step}: 小模型无 token_ids，切换大模型。")
                        force_large_model_this_step = True
                else:
                    logger.error(f"❌ Step {current_step}: 小模型调用失败，切换大模型。")
                    force_large_model_this_step = True

            else:
                self.stats["fallback_to_large"] += 1
                logger.warning(f"🔸 Step {current_step}: 🔴 切换到大模型...")
                
                try:
                    ctx_ids = self.tokenizer_large.apply_chat_template(conversation_history, tokenize=True, add_generation_prompt=False)
                    self.stats["prefill_tokens_count"] += len(ctx_ids)
                except:
                    pass
                
                large_res = self._call_vllm_api(
                    url=self.large_url,
                    model_name=self.large_model_name,
                    prompt_text=current_prompt_text,
                    max_tokens=self.max_step_tokens,
                    stop_sequences=stop_sequences,
                    get_logprobs=False,
                    temperature=0.7,
                    repetition_penalty=1.15
                )
                
                if large_res and large_res.get("choices"):
                    choice = large_res["choices"][0]
                    generated_text = choice.get("text", "")
                    finish_reason = choice.get("finish_reason", "")
                    
                    new_token_ids = choice.get("token_ids")
                    if not new_token_ids and generated_text:
                        new_token_ids = self.tokenizer_large.encode(generated_text, add_special_tokens=False)

                    if new_token_ids:
                        preview_text = generated_text[:60].replace("\n", "\\n") if generated_text else ""
                        logprobs_raw = choice.get("logprobs", [])
                        confidence = self._calculate_confidence(logprobs_raw)

                        logger.info(f"🔸 Step {current_step} [大]: 置信度={confidence:.3f}, 结束原因: {finish_reason or 'EOS'} | 预览: '{preview_text}...'")
                        
                        all_generated_segments.append({
                            "ids": new_token_ids, "source": 1, "text": generated_text
                        })

                        self.stats["tokens_generated_by_large"] += len(new_token_ids)
                        if confidence < self.confidence_threshold:
                            logger.warning(f"⚠️ Step {current_step}: 置信度低 ({confidence:.3f})，下一步继续大模型。")
                            force_large_model_this_step = True
                    else:
                        logger.warning(f"⚠️ Step {current_step}: 大模型无 token_ids，终止。")
                        break 
                else:
                    logger.error(f"❌ Step {current_step}: 大模型调用失败，终止。")
                    break 

            # ================= 【核心修复】更新历史 & 检查结束条件 =================
            if generated_text:
                # 1. 检查复读
                current_prefix = generated_text[:30]
                if current_prefix == last_generated_prefix and len(generated_text) > 50:
                    repeat_counter += 1
                    if repeat_counter >= 2:
                        logger.error("❌ 检测到重复生成循环，强制终止。")
                        break
                else:
                    repeat_counter = 0
                    last_generated_prefix = current_prefix

                # 2. 【关键】清洗生成文本中的 EOS 标记，防止污染历史
                clean_generated_text = self._clean_eos(generated_text)
                
                # 3. 检查是否需要人工注入 step_tag
                history_content = clean_generated_text
                step_tag_found = self.step_tag in clean_generated_text
                
                if finish_reason == "length":
                    if not step_tag_found:
                        history_content += f"\n{self.step_tag}\n"
                        logger.debug(f"✂️ Step {current_step}: 被长度截断，人工注入 {self.step_tag} 强制闭合步骤。")
                elif not step_tag_found:
                    if "\\boxed{" not in clean_generated_text:
                        history_content += f"\n{self.step_tag}\n"
                        logger.debug(f"🏷️ Step {current_step}: 自然结束但无标签，人工补全 {self.step_tag}。")
                    else:
                        logger.debug(f"✅ Step {current_step}: 检测到答案，无需补标签。")

                # 4. 将【清洗并处理后的文本】加入历史
                conversation_history.append({"role": "assistant", "content": history_content})
                
                step_count += 1
                logger.debug(f"🔄 Step {current_step} 完成，累计步数: {step_count}")
                
                has_answer = "\\boxed{" in clean_generated_text
                if has_answer:
                    logger.info(f"🎯 Step {current_step}: 检测到最终答案，任务完成。")
                    break
                
                if force_large_model_this_step:
                    logger.debug(f"⏭️ 准备在下一轮 (Step {step_count+1}) 切换大模型...")
                else:
                    logger.debug(f"⏩ 继续下一轮 (Step {step_count+1}) 小模型推理...")
                    
            else:
                logger.warning("⚠️ 生成的文本为空，终止循环。")
                break 
        
        total_tokens = self.stats["tokens_generated_by_small"] + self.stats["tokens_generated_by_large"]
        logger.info(f"✅ [Routing] 完成。总步数: {step_count}, 总Token: {total_tokens}")

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