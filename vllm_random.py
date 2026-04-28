import requests
import logging
import random
from typing import List, Optional, Dict
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoTokenizer 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
ratio = 9


@register_model("vllm_random")
class VLLMRANDOM(LM):
    def __init__(
        self,
        small_url: str,
        large_url: str,
        small_model_name: str,
        large_model_name: str,
        confidence_threshold: float = 0.7,
        max_step_tokens: int = 2048,
        max_gen_tokens: int = 16384,
        step_plan: str = "",
        debug: bool = False,
        **kwargs
    ):
        super().__init__()
        self.small_url = small_url
        self.small_model_name = small_model_name
        self.large_url = large_url
        self.large_model_name = large_model_name
        self.debug = debug

        self.max_gen_tokens = int(max_gen_tokens)

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
            "tokens_generated_by_small": 0,
            "tokens_generated_by_large": 0,
            "generate_by_large": 0
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
        temperature: float = 0.7,
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
                    logger.warning(f"Error 400: {err_text}")
                else:
                    logger.warning(f"Error {response.status_code}: {response.text[:100]}")
                return None

        except Exception as e:
            logger.error(f"Network Error: {type(e).__name__} - {e}")
            return None
    
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

    def _run_speculative_routing(self, messages: List[Dict[str, str]], stop_sequences: List[str]) -> str:  
        logger.info(f"[Routing] 启动， 随机模式")

        random_ratio = 1/(ratio+1)
        if_use_large_model = True if random.random()<random_ratio else False

        if if_use_large_model:
            self.stats["generate_by_large"] += 1

        tokenizer = self.tokenizer_large if if_use_large_model else self.tokenizer_small
        model_name = self.large_model_name if if_use_large_model else self.small_model_name
        url = self.large_url if if_use_large_model else self.small_url

        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, 
                add_generation_prompt=False
            ).rstrip()

        if self.debug:
            print(f"初始 Prompt 长度: {len(prompt_text)} chars", flush=True)
            self.print_preview(prompt_text, 200)

        generated_text = ""
        finish_reason = ""
        res = self._call_vllm_api(
            url=url,
            model_name=model_name,
            prompt_text = prompt_text,
            max_tokens=self.max_gen_tokens,
            stop_sequences=stop_sequences,
            temperature=0.7,
        )
        
        if res and "choices" in res:
            choice = res["choices"][0]
            generated_text = choice.get("text", "")
            finish_reason = choice.get("finish_reason", "")
            new_token_ids = choice.get("token_ids", [])
                    
            if not new_token_ids and generated_text:
                new_token_ids = tokenizer.encode(generated_text, add_special_tokens=False)
            
            logger.info(f" 模型完成 | 原因: {finish_reason or 'EOS'}")
            if not new_token_ids:
                logger.error("   模型未生成有效 tokens。")
            else :
                if self.debug:
                    print(f"   模型生成 tokens: {len(new_token_ids)}", flush=True)
                if if_use_large_model:
                    self.stats["tokens_generated_by_large"] += len(new_token_ids)
                else:
                    self.stats["tokens_generated_by_small"] += len(new_token_ids)

        else:
            logger.error("  模型调用失败或返回格式错误")

        return generated_text
    
    def generate_until(self, requests: List) -> List[str]:
        results = []
        pbar = tqdm(requests, desc="small model")

        for req in pbar:
            if isinstance(req, Instance):
                context, gen_kwargs = req.args
            else:
                context, gen_kwargs = req

            self.stats["total_requests"] += 1
            
            stop_sequences = gen_kwargs.get("until", []) 
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]
            
            final_messages = []
            final_messages.append({"role": "user", "content": context})

            try:
                generated_text = self._run_speculative_routing(final_messages, stop_sequences)
                results.append(generated_text)
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                results.append("")

        pbar.close()
        
        with open("./log_traces/logs/routing_random.log", "a", encoding="utf-8") as f:
            total = self.stats["total_requests"]
            if total > 0:
                f.write("\n" + "="*40 + "\n")
                f.write("Routing Statistics:\n")
                f.write(f"   Total Requests: {total}\n")
                f.write(f"   Fallbacks: {self.stats['generate_by_large']}\n")
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