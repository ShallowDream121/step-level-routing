import os
import requests
import math
import time
import json
import csv
import logging
from typing import List, Optional, Dict, Any, Tuple
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoTokenizer 
import re
from collections import Counter
from distutils.util import strtobool

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
        confidence_threshold: float = 0.7,
        max_step_tokens: int = 512,
        max_gen_tokens: int = 16384,
        step_plan: str = "",
        debug: bool = False,
        dynamic_block: bool = True,
        block_tiers: str = "64,128,256,512,1024,2048",
        upgrade_steps: int = 2,
        **kwargs
    ):
        super().__init__()
        self.small_url = small_url
        self.large_url = large_url
        self.small_model_name = small_model_name
        self.large_model_name = large_model_name
        self.debug = bool(strtobool(str(debug)))

        try:
            self.confidence_threshold = float(confidence_threshold)
        except (ValueError, TypeError):
            self.confidence_threshold = 0.7

        self.max_step_tokens = max_step_tokens
        self.max_gen_tokens = max_gen_tokens

        self.dynamic_block = dynamic_block
        self.upgrade_steps = upgrade_steps
        self.block_tiers = [int(x.strip()) for x in block_tiers.split(",")]
        self.block_tiers.sort()

        if self.dynamic_block:
            initial_tokens = int(max_step_tokens)
            best_idx = 0
            best_dist = abs(self.block_tiers[0] - initial_tokens)
            for i, t in enumerate(self.block_tiers):
                dist = abs(t - initial_tokens)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            self._initial_block_idx = best_idx
        else:
            self._initial_block_idx = 0

        self.log_file_path = "./confidence/confidence_log_blocks_512_gpqa.csv"

        self.eos_tokens = [
            "<｜end▁of▁sentence｜>",
            "</s>",        
            "<|endoftext|>",   
            "[END]",   
            "<eos>"              
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
        }

    def _clean_eos(self, text: str) -> str:
        cleaned_text = text
        for eos in self.eos_tokens:
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
        prompt_token_ids: Optional[List[int]] = None,
        max_tokens: int = 256, 
        stop_sequences: List[str] = None,
        get_logprobs: bool = True,
        temperature: float = 0.7,
        top_k_logprobs: int = 5,
        repetition_penalty: float = 1.15
    ) -> Optional[Dict]:
        
        endpoint = f"{url.rstrip('/')}/v1/completions"

        payload = {
            "model": model_name,
            "prompt": prompt_text,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "repetition_penalty": repetition_penalty, 
            "stream": False,
            "n": 1
        }

        if get_logprobs:
            payload["logprobs"] = top_k_logprobs
        
        if stop_sequences:
            payload["stop"] = stop_sequences

        try:
            response = self.session.post(endpoint, json=payload, timeout=None)
            
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
                        logger.warning(f"Error 400: logprobs limit exceeded.")
                    else:
                        logger.warning(f"Error 400: {err_text}")
                else:
                    logger.warning(f"Error {response.status_code}: {response.text[:100]}")
                return None

        except Exception as e:
            logger.error(f"Network Error: {type(e).__name__} - {e}")
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

    def _get_initial_prompt_text(self, messages: List[Dict[str, str]], model_type: str) -> str:
        tokenizer = self.tokenizer_small if model_type == "small" else self.tokenizer_large
        try:
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            return prompt.rstrip()
        except Exception as e:
            logger.warning(f"应用聊天模板失败: {e}。回退到纯文本拼接。")
            text = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                text += f"{role}: {content}\n"
            text += "assistant: "
            return text
        
    def _run_speculative_routing(self, messages: List[Dict[str, str]], stop_sequences: List[str]) -> str:        
        if self.dynamic_block:
            logger.info(f"[Routing] 启动 | 阈值: {self.confidence_threshold} | 动态Block: {self.block_tiers}")
        else:
            logger.info(f"[Routing] 启动 | 阈值: {self.confidence_threshold}")
        
        try:
            initial_model_type = "small"
            current_raw_text = self._get_initial_prompt_text(messages, initial_model_type)
        except Exception as e:
            logger.warning(f"模板渲染失败: {e}")
            current_raw_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant: "
        
        all_generated_segments = []
        step_count = 0
        force_large_model_this_step = False

        current_block_idx = self._initial_block_idx
        consecutive_good = 0

        if self.dynamic_block:
            logger.info(f"  初始Block: {self.block_tiers[current_block_idx]} tokens | 升级需连续 {self.upgrade_steps} 步高置信度")

        if self.debug:
            print(f"初始 Prompt 长度: {len(current_raw_text)} chars", flush=True)
            self.print_preview(current_raw_text, 200)

        while True:
            current_step = step_count + 1
            use_large_model = force_large_model_this_step
            force_large_model_this_step = False

            current_block_tokens = self.block_tiers[current_block_idx] if self.dynamic_block else self.max_step_tokens

            tokenizer_used = self.tokenizer_large if use_large_model else self.tokenizer_small
            url_used = self.large_url if use_large_model else self.small_url
            model_name = self.large_model_name if use_large_model else self.small_model_name

            input_ids = tokenizer_used.encode(current_raw_text, add_special_tokens=False)

            gen_tokens = 0

            if len(input_ids) > 28000:
                logger.warning(f"上下文过长 ({len(input_ids)})，截断尾部...")
                input_ids = input_ids[-27000:]

            model_label = "大" if use_large_model else "小"
            logger.info(f"Step {current_step} [{model_label}]: 输入 Tokens={len(input_ids)}")

            generated_text = ""
            new_token_ids = []
            finish_reason = ""

            if use_large_model:
                self.stats["fallback_to_large"] += 1

            res = self._call_vllm_api(
                url=url_used,
                model_name=model_name,
                prompt_text=current_raw_text,
                prompt_token_ids=input_ids,
                max_tokens=current_block_tokens,
                stop_sequences=stop_sequences,
                get_logprobs=True,
                temperature=0.7,
                repetition_penalty=1.15
            )

            if res and "choices" in res:
                choice = res["choices"][0]
                generated_text = choice.get("text", "")
                finish_reason = choice.get("finish_reason", "")
                new_token_ids = choice.get("token_ids", [])

                if not new_token_ids and generated_text:
                    new_token_ids = tokenizer_used.encode(generated_text, add_special_tokens=False)

                logprobs_raw = choice.get("logprobs", None)
                confidence = self._calculate_confidence(logprobs_raw)

                logger.info(f" {model_label}模型完成 | 置信度: {confidence:.3f} | 原因: {finish_reason or 'EOS'}")

                if not new_token_ids:
                    logger.error(f"   {model_label}模型未生成有效 tokens。")
                    break

            else:
                logger.error(f"  {model_label}模型调用失败或返回格式错误")
                break

            # ================= 更新上下文 =================
            source = 1 if use_large_model else 0
            if new_token_ids:
                all_generated_segments.append({
                    "ids": new_token_ids,
                    "source": source,
                    "text": generated_text
                })
                
                if use_large_model:
                    self.stats["tokens_generated_by_large"] += len(new_token_ids)
                else:
                    self.stats["tokens_generated_by_small"] += len(new_token_ids)
                
                gen_tokens += len(new_token_ids)

                decoded_segment = tokenizer_used.decode(new_token_ids, skip_special_tokens=True)
                
                clean_segment = self._clean_eos(decoded_segment)
                current_raw_text += clean_segment
                
                #with open(self.log_file_path, mode='a', newline='', encoding='utf-8') as csvfile:
                #    writer = csv.writer(csvfile)
                #    writer.writerow([current_step, f"{confidence:.5f}", source, len(new_token_ids)])

                step_count += 1
                if self.debug:
                    print(f"  已拼接文本 (当前总长: {len(current_raw_text)} chars)", flush=True)

                if "</think>" in clean_segment or finish_reason == "stop":
                    logger.info("   检测到推理结束 (</think>)，任务完成。")
                    break
                elif "\\boxed{" in clean_segment:
                    logger.info("   检测到最终答案 (\\boxed)，任务完成。")
                    break
                
                if gen_tokens > int(self.max_gen_tokens):
                    logger.info("达到最大tokens生成上限，停止推理。")
                    break

                if self.dynamic_block:
                    if confidence > self.confidence_threshold:
                        consecutive_good += 1
                        if consecutive_good >= self.upgrade_steps and current_block_idx < len(self.block_tiers) - 1:
                            current_block_idx += 1
                            consecutive_good = 0
                            logger.info(f"   Block ↑ 升至 {self.block_tiers[current_block_idx]} tokens")
                    else:
                        consecutive_good = 0
                        if current_block_idx > 0:
                            current_block_idx //=2
                            logger.info(f"   Block ↓ 降至 {self.block_tiers[current_block_idx]} tokens")

                if not use_large_model and confidence < self.confidence_threshold:
                    logger.info(f"   置信度 ({confidence:.3f}) 低于阈值，*下一步* 切换大模型。")
                    force_large_model_this_step = True
                elif use_large_model and confidence < self.confidence_threshold:
                    logger.info(f"   大模型置信度 ({confidence:.3f}) 仍低，*下一步* 继续大模型。")
                    force_large_model_this_step = True
                else:
                    if use_large_model:
                        logger.info("   下一步恢复小模型推理。")
                    else:
                        logger.info("   下一步继续小模型推理。")

            else:
                logger.error("   无生成内容，循环终止。")
                break

        logger.info(f"[Routing] 结束。总步数: {step_count}")
        return self._decode_mixed_segments(all_generated_segments)

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
            
            clean_context = context
            
            final_messages = []
            final_messages.append({"role": "user", "content": clean_context})

            try:
                generated_text = self._run_speculative_routing(final_messages, stop_sequences)
                results.append(generated_text)
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                results.append("")

        pbar.close()
        
        with open("./log_traces/logs/routing_block_dynamic3.log", "a", encoding="utf-8") as f:
            total = self.stats["total_requests"]
            if total > 0:
                f.write("\n" + "="*40 + "\n")
                f.write("Routing Statistics:\n")
                f.write(f"   Total Requests: {total}\n")
                f.write(f"   Fallbacks: {self.stats['fallback_to_large']}\n")
                
                s_tok = self.stats['tokens_generated_by_small']
                l_tok = self.stats['tokens_generated_by_large']
                
                f.write(f"   Tokens (Small/Large): {s_tok} / {l_tok}\n")
                
                if s_tok + l_tok > 0:
                    ratio = s_tok/(s_tok+l_tok)
                    f.write(f"   Small Ratio: {ratio:.2%}\n")
                    
                f.write("="*40 + "\n")
            
        return results

    def loglikelihood(self, requests):
        raise NotImplementedError("Speculative Routing supports 'generate_until' only.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("Not implemented.")
    
    # for debug
    def print_preview(self, text, length=100):
        if not text:
            print("[空文本]")
            return

        if len(text) <= length * 2:
            print(f"全文内容:\n{text}")
        else:
            head = text[:length]
            tail = text[-length:]
            print(f"文本预览 (前{length}字 & 后{length}字):\n{head}\n\n...... (中间省略 {len(text) - length * 2} 字) ...... \n\n{tail}")