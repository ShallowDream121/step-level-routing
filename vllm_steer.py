import os
import requests
import math
import time
import json
import logging
import csv
from typing import List, Optional, Dict, Any, Tuple
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoTokenizer 
import re
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@register_model("vllm_steer")
class VLLMSteer(LM):
    def __init__(
        self,
        small_url: str,
        large_url: str,
        small_model_name: str,
        large_model_name: str,
        confidence_threshold: float = 0.7,
        max_step_tokens: int = 2048,
        max_gen_tokens: int = 16384,
        step_tag: str = "/n/n",
        debug: bool = False,
        **kwargs
    ):
        super().__init__()
        self.small_url = small_url
        self.large_url = large_url
        self.small_model_name = small_model_name
        self.large_model_name = large_model_name
        self.debug = debug

        try:
            self.confidence_threshold = float(confidence_threshold)
        except (ValueError, TypeError):
            self.confidence_threshold = 0.7

        self.max_step_tokens = max_step_tokens
        self.max_gen_tokens = int(max_gen_tokens)
        self.step_tag = step_tag

        self.log_file_path = "confidence_log.csv"

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

        
    def _run_speculative_routing(self, messages: List[Dict[str, str]], stop_sequences: List[str]) -> str:        
        logger.info(f"[Routing] 启动 | 阈值: {self.confidence_threshold}")
        
        try:
            current_raw_text = self.tokenizer_small.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
        except Exception as e:
            logger.warning(f"模板渲染失败: {e}")
            current_raw_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant: "
        
        all_generated_segments = []
        force_large_model_this_step = False
        step_count = 0
        gen_tokens = 0

        while True:
            current_step = step_count + 1
            use_large_model = force_large_model_this_step
            force_large_model_this_step = False
            
            tokenizer_used = self.tokenizer_large if use_large_model else self.tokenizer_small
            url_used = self.large_url if use_large_model else self.small_url
            model_name = self.large_model_name if use_large_model else self.small_model_name

            input_ids = tokenizer_used.encode(current_raw_text, add_special_tokens=False)
            
            if len(input_ids) > 32000:
                logger.warning(f"上下文过长 ({len(input_ids)})，截断尾部...")
                input_ids = input_ids[-30000:] 
            if use_large_model:
                self.stats["prefill_tokens_count"] += len(input_ids)
                logger.info(f"Step {current_step} [大]: 输入 Tokens={len(input_ids)}")
            else:
                logger.info(f"Step {current_step} [小]: 输入 Tokens={len(input_ids)}")

            generated_text = ""
            new_token_ids = []
            
            step_res = self._call_vllm_api(
                url=url_used,
                model_name=model_name,
                prompt_text=current_raw_text,
                max_tokens=self.max_step_tokens,
                stop_sequences=["\n\n"],
                get_logprobs=True
            )

            if step_res and "choices" in step_res:
                choice = step_res["choices"][0]
                generated_text = choice.get("text", "")
                
                new_token_ids = tokenizer_used.encode(generated_text, add_special_tokens=False)
                
                logprobs_raw = choice.get("logprobs", None)
                confidence = self._calculate_confidence(logprobs_raw)

                source = 1 if use_large_model else 0
                all_generated_segments.append({
                    "ids": new_token_ids,
                    "source": source
                })
                #with open(self.log_file_path, mode='a', newline='', encoding='utf-8') as csvfile:
                #    writer = csv.writer(csvfile)
                #    writer.writerow([current_step, f"{confidence:.5f}", source, len(new_token_ids)])

                if use_large_model:
                    self.stats["tokens_generated_by_large"] += len(new_token_ids)
                else:
                    self.stats["tokens_generated_by_small"] += len(new_token_ids)

                if not use_large_model and confidence < self.confidence_threshold and step_count == 0:
                    logger.info(f"Step {current_step} 首步小模型置信度低 ({confidence:.3f})，直接切换到大模型")
                    force_large_model_this_step = True
                    continue

                if not use_large_model and confidence < self.confidence_threshold:
                    logger.info(f"Step {current_step} Confidence Low ({confidence:.3f}), switch to Large next")
                    force_large_model_this_step = True
                else:
                    logger.info(f"Step {current_step} Confidence OK ({confidence:.3f})")
                    force_large_model_this_step = False

                gen_tokens += len(new_token_ids)
                generated_text += "\n\n"
                current_raw_text += generated_text

                if "\\boxed{" in generated_text or "</think>" in generated_text:
                    logger.info("检测到最终答案或结束标记，停止生成。")
                    break
                elif gen_tokens>=self.max_gen_tokens:
                    logger.warning(f"超出最大token限制:{self.max_gen_tokens}")
                    break

                step_count += 1
            else:
                logger.error("Step execution failed")
                break
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
            
            if self.step_tag not in stop_sequences:
                stop_sequences.append(self.step_tag)
            
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