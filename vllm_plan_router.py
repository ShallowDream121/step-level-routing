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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@register_model("vllm_plan_router")
class VLLMPlanRouter(LM):
    def __init__(self,
                 small_url: str,
                 large_url: str,
                 small_model_name: str,
                 large_model_name: str,
                 confidence_threshold: float = 0.7,
                 max_step_tokens: int = 2048,
                 max_gen_tokens: int = 16384,
                 step_plan: str = "",
                 debug: bool = False,
                 **kwargs):
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
        self.step_plan = step_plan

        self.system_prompt = (
            "You are a Step-by-Step Reasoning Engine. You are provided with a 'Step-by-Step Plan' as your roadmap.\n"
            "### ROLE ###\n"
            "1. **The Plan is a Map**: The list of steps provided in the context serves as a reference for the overall strategy. Do NOT attempt to execute the entire list at once.\n"
            "2. **Focus on the Current Step**: You must realise the step currently assigned to you and execute ONLY that step.\n"
            "3. **Context Awareness**: Use the 'Previous Reasoning' as the factual basis for your current calculation. Do not repeat previous work.\n"
            "\n"
            "### CRITICAL CONSTRAINTS ###\n"
            "- **No Anticipation**: Even if you see future steps in the plan, DO NOT solve them yet. Stop immediately after finishing the current step.\n"
            "- **No Repetition**: Do not re-list the plan or repeat the step description. Go straight to the solution.\n"
            "\n"
            "### OUTPUT FORMAT ###\n"
            "- Provide clear, logical reasoning in English.\n"
            "- If the step produces a result, enclose it in \\boxed{{}}.\n"
            "- End your response immediately with the tag '</think>'."
        )
        
        self.eos_tokens = ["<｜end▁of▁sentence｜>", "</s>", "</refer>final<|im_message|>", "[END]", "<eos>"]

        logger.info(f"Loading Small Model Tokenizer: {small_model_name}...")
        self.tokenizer_small = AutoTokenizer.from_pretrained(small_model_name, trust_remote_code=True)
        if self.tokenizer_small.pad_token is None:
            self.tokenizer_small.pad_token = self.tokenizer_small.eos_token

        logger.info(f"Loading Large Model Tokenizer: {large_model_name}...")
        self.tokenizer_large = AutoTokenizer.from_pretrained(large_model_name, trust_remote_code=True)
        if self.tokenizer_large.pad_token is None:
            self.tokenizer_large.pad_token = self.tokenizer_large.eos_token
            
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
        self.stats = {
            "total_requests": 0,
            "fallback_to_large": 0,
            "plan_tokens": 0,
            "tokens_generated_by_small": 0,
            "tokens_generated_by_large": 0,
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

    def _run_routing(self, messages: List[Dict[str, str]], stop_sequences: List[str], steps: List[str]) -> str:
        logger.info(f"[Text-Concatenation Routing] 启动 | 阈值: {self.confidence_threshold}")
        full_context_str = ""
        
        system_content = self.tokenizer_large.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
        full_context_str += system_content + "\n\n"

        generated_context = ""
        
        all_generated_segments = []
        force_large_model_this_step = False
        gen_tokens = 0

        if len(steps) == 0:
            steps.append("Calculate.")

        for step_idx, step_desc in enumerate(steps):
            current_step = step_idx + 1
            use_large_model = force_large_model_this_step
            tokenizer_used = self.tokenizer_large if use_large_model else self.tokenizer_small
            url_used = self.large_url if use_large_model else self.small_url
            model_name = self.large_model_name if use_large_model else self.small_model_name

            current_prompt = full_context_str 
            current_messages = messages.copy()
            if generated_context != "":
                current_messages.append({"role":"assistant", "content":generated_context})

            newsys_prompt = "You must realise the step currently assigned to you and execute ONLY that step.\n"
            if step_idx == len(steps)-1 and len(steps)>1:
                newsys_prompt += "this step produces the final result, enclose it in \\boxed{{}}\n"

            newsys_prompt += "CURRENT STEP:"
            newsys_prompt += step_desc + "\n"

            newsys_prompt += "Please finish the step.\n"

            current_messages.append({"role":"user","content":newsys_prompt})

            current_prompt = self.tokenizer_large.apply_chat_template(
                current_messages, 
                tokenize=False, 
                add_generation_prompt=False
            )

            try:
                prompt_tokens = tokenizer_used.encode(current_prompt, add_special_tokens=False)
            except Exception as e:
                logger.error(f"Tokenization Error: {e}")
                prompt_tokens = []
                
            input_len = len(prompt_tokens)
            
            if input_len > 28000: 
                logger.warning(f"Context too long ({input_len} tokens). Truncating...")
                system_tokens = tokenizer_used.encode(system_content + "\n\n", add_special_tokens=False)
                prompt_tokens = system_tokens + prompt_tokens[-2048:]
                current_prompt = tokenizer_used.decode(prompt_tokens, skip_special_tokens=True)

            step_res = self._call_vllm_api(
                url=url_used,
                model_name=model_name,
                prompt_text=current_prompt,
                max_tokens=self.max_step_tokens,
                stop_sequences=stop_sequences,
                get_logprobs=True,
                temperature=0.7
            )

            if step_res and "choices" in step_res:
                choice = step_res["choices"][0]
                generated_text = choice.get("text", "")
                logprobs_raw = choice.get("logprobs", None)
                confidence = self._calculate_confidence(logprobs_raw)

                new_token_ids = tokenizer_used.encode(generated_text, add_special_tokens=False)
                source = 1 if use_large_model else 0
                all_generated_segments.append({
                    "ids": new_token_ids,
                    "source": source
                })
                
                if use_large_model:
                    self.stats["tokens_generated_by_large"] += len(new_token_ids)
                    self.stats["fallback_to_large"] += 1
                else:
                    self.stats["tokens_generated_by_small"] += len(new_token_ids)
                
                gen_tokens += len(new_token_ids)

                full_context_str += step_desc
                full_context_str +=": \n"
                full_context_str += generated_text + "\n\n"

                generated_context += generated_text

                self.print_preview(generated_text,300)

                if not use_large_model and confidence < self.confidence_threshold:
                    logger.info(f"Step {current_step} Confidence Low ({confidence:.3f}), forcing Large model NEXT")
                    force_large_model_this_step = True
                elif use_large_model and confidence < self.confidence_threshold:
                    logger.info(f"Step {current_step} Confidence Low ({confidence:.3f}), continue Large model")
                    force_large_model_this_step = True
                else:
                    logger.info(f"Step {current_step} Confidence OK ({confidence:.3f})")
                    force_large_model_this_step = False

                if gen_tokens >= self.max_gen_tokens:
                    logger.warning(f"超出最大token限制: {self.max_gen_tokens}")
                    break
            else:
                logger.error("Step execution failed")
                full_context_str += "[ERROR: Step Failed]\n\n"
                force_large_model_this_step = True

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
        with open(self.step_plan, 'r', encoding='utf-8') as f:
            plans = json.load(f)

        idx = 0

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
            final_messages=[{"role": "system", "content": self.system_prompt}]
            final_messages.append({"role": "user", "content": clean_context})
            steps = plans[idx]['steps']
            idx+=1

            plan_txt = '\n'.join(steps)
            final_messages.append({"role": "assistant", "content": plan_txt})

            try:
                generated_text = self._run_routing(final_messages, stop_sequences, steps)
                results.append(generated_text)
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                results.append("")

        pbar.close()

        with open("./log_traces/logs/routing_plan.log", "a", encoding="utf-8") as f:
            total = self.stats["total_requests"]
            if total > 0:
                f.write("\n" + "="*40 + "\n")
                f.write("Routing Statistics:\n")
                f.write(f"   Total Requests: {total}\n")
                f.write(f"   Fallbacks: {self.stats['fallback_to_large']}\n")
                f.write(f"   Tokens for Plan: {self.stats['plan_tokens']}\n")
                
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