import requests
import logging
from typing import List, Optional, Dict
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoTokenizer 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@register_model("vllm_big")
class VLLMBIG(LM):
    def __init__(
        self,
        small_url: str,
        large_url: str,
        small_model_name: str,
        large_model_name: str,
        confidence_threshold: float = 0.7,
        max_step_tokens: int = 2048,
        max_gen_tokens: int = 16384,
        step_tag: str = "</step>",
        debug: bool = False,
        **kwargs
    ):
        super().__init__()
        self.large_url = large_url
        self.large_model_name = large_model_name
        self.debug = debug

        self.max_gen_tokens = int(max_gen_tokens)

        self.step_tag = step_tag

        logger.info(f"Loading Large Model Tokenizer: {large_model_name}...")
        self.tokenizer_large = AutoTokenizer.from_pretrained(large_model_name, trust_remote_code=True)
        if self.tokenizer_large.pad_token is None:
            self.tokenizer_large.pad_token = self.tokenizer_large.eos_token

        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
        self.stats = {
            "total_requests": 0,
            "tokens_generated": 0,
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


    def _get_initial_prompt_text(self, messages: List[Dict[str, str]]) -> str:
        try:
            prompt = self.tokenizer_large.apply_chat_template(
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
        logger.info(f"[Routing] 启动 , 使用大模型:{self.large_model_name}")

        prompt_text = self._get_initial_prompt_text(messages)

        if self.debug:
            print(f"初始 Prompt 长度: {len(prompt_text)} chars", flush=True)
            self.print_preview(prompt_text, 200)

        generated_text = ""
        finish_reason = ""
        large_res = self._call_vllm_api(
            url=self.large_url,
            model_name=self.large_model_name,
            prompt_text = prompt_text,
            max_tokens=self.max_gen_tokens,
            stop_sequences=stop_sequences,
            temperature=0.7,
        )
        
        if large_res and "choices" in large_res:
            choice = large_res["choices"][0]
            generated_text = choice.get("text", "")
            finish_reason = choice.get("finish_reason", "")
            new_token_ids = choice.get("token_ids", [])
                    
            if not new_token_ids and generated_text:
                new_token_ids = self.tokenizer_large.encode(generated_text, add_special_tokens=False)
            
            logger.info(f" 大模型完成 | 原因: {finish_reason or 'EOS'}")
            if not new_token_ids:
                logger.error("   大模型未生成有效 tokens。")
            else :
                if self.debug:
                    print(f"   大模型生成 tokens: {len(new_token_ids)}", flush=True)
                self.stats["tokens_generated"] += len(new_token_ids)

        else:
            logger.error("  大模型调用失败或返回格式错误")

        return generated_text
    
    def generate_until(self, requests: List) -> List[str]:
        results = []
        pbar = tqdm(requests, desc="large model")

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
        
        total = self.stats["total_requests"]
        if total > 0:
            print("\n" + "="*40)
            print("Routing Statistics:")
            print(f"   Total Requests: {total}")
            s_tok = self.stats['tokens_generated']
            print(f"   Tokens generated (Large): {s_tok}")
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
