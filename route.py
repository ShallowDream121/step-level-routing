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

        print(f"正在加载小模型 Tokenizer...")
        self.tokenizer_small = AutoTokenizer.from_pretrained(
            small_model_name,
            trust_remote_code=True 
        )

        if self.tokenizer_small.pad_token is None:
            self.tokenizer_small.pad_token = self.tokenizer_small.eos_token

        print(f"正在加载大模型 Tokenizer...")
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
            return 0.0
        
        probs = []
        for token_logprobs in logprobs_data:
            if not token_logprobs:
                continue
            if isinstance(token_logprobs, dict):
                info = next(iter(token_logprobs.values()), None)
                if info and isinstance(info, dict) and "logprob" in info:
                    logprob = info["logprob"]
                    if logprob is not None:
                        probs.append(math.exp(logprob))

        if not probs:
            return 0.0
        
        # 置信度定义为所有 token 中最大概率的平均值
        return sum(probs) / len(probs)

    def _decode_tokens(self, token_ids: List[int], model_type: str = "small", skip_special_tokens: bool = True) -> str:
        if model_type == "small":
            tokenizer = self.tokenizer_small
        elif model_type == "large":
            tokenizer = self.tokenizer_large
        else:
            raise ValueError(f"无效的 model_type: '{model_type}'。必须是 'small' 或 'large'。")
        
        if not token_ids:
            return ""
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def _run_speculative_routing(self, prompt: str) -> str:
        """
        核心逻辑：
        1. 小模型生成 -> 无论置信度高低，都将生成的 Token 拼接到 current_token_ids。
        2. 若置信度低 -> 标记 next_step_use_large = True。
        3. 下一轮循环 -> 检测到标记，直接调用大模型进行 Prefill + 生成。
        4. 统计：所有小模型生成的都计入 small_tokens，所有大模型生成的都计入 large_tokens。
        """

        input_ids = self.tokenizer_small.encode(prompt, add_special_tokens=False)
        
        current_token_ids = list(input_ids)
        source_flags = [0] * len(input_ids)  # 0: Small, 1: Large
        
        max_steps = 20
        step_count = 0
        global_stop_sequences = [self.step_tag] + ["</s>", "<|eot_id|>", "\n\nUser:", "User:"]
        step_stops = [self.step_tag] + global_stop_sequences
        
        force_large_model_this_step = False 

        while step_count < max_steps:
            current_step = step_count + 1
            
            use_large_model = force_large_model_this_step
            force_large_model_this_step = False

            generated_segment_ids = []
            generated_text = ""
            current_source_flag = 0 # 默认小模型

            # ================= 分支 A: 小模型执行 =================
            if not use_large_model:
                current_source_flag = 0
                len_before = len(current_token_ids)
                
                small_res = self._call_vllm_api(
                    url=self.small_url,
                    prompt_token_ids=current_token_ids, # vLLM 内部会用 small_model 的 tokenizer encode
                    max_tokens=self.max_step_tokens,
                    stop_sequences=step_stops,
                    get_logprobs=True,
                    temperature=0.7
                )
                
                if small_res:
                    output_data = small_res.get("outputs", [{}])[0] if "outputs" in small_res else small_res
                    generated_text = output_data.get("text", "")
                    
                    # 【关键】获取小模型生成的 Token IDs
                    new_token_ids = output_data.get("token_ids", None)
                    if new_token_ids is None:
                        # 兜底：如果没有返回 IDs，必须用【小模型】tokenizer 编码
                        new_token_ids = self.tokenizer_small.encode(generated_text, add_special_tokens=False)
                    
                    logprobs_raw = output_data.get("logprobs", [])
                    confidence = self._calculate_confidence(logprobs_raw)
                    generated_segment_ids = new_token_ids
                    
                    # 更新并行数组
                    current_token_ids.extend(generated_segment_ids)
                    source_flags.extend([0] * len(generated_segment_ids)) # 标记为 0
                    
                    self.stats["tokens_generated_by_small"] += len(generated_segment_ids)
                    
                    print(f"   [Step {current_step}] 🟢 小模型生成 (置信度: {confidence:.2f}) -> 采纳 {len(generated_segment_ids)} tokens")

                    if confidence < self.confidence_threshold:
                        print(f"   ⚠️ 置信度低，标记 *下一步* 切换到大模型。")
                        force_large_model_this_step = True
                        
                else:
                    print(f"   ❌ 小模型调用失败，标记 *下一步* 切换到大模型。")
                    force_large_model_this_step = True

            # ================= 分支 B: 大模型执行 =================
            else:
                current_source_flag = 1
                self.stats["fallback_to_large"] += 1
                print(f"   [Step {current_step}] 🔴 触发大模型 (原因: 上一步低置信度/失败)")
                
                large_res = self._call_vllm_api(
                    url=self.large_url,
                    prompt_token_ids=current_token_ids, # vLLM 内部会用 large_model 的 tokenizer encode
                    max_tokens=self.max_step_tokens,
                    stop_sequences=step_stops,
                    get_logprobs=False, # 大模型通常不需要 logprobs 除非做验证
                    temperature=0.7
                )
                
                if large_res:
                    output_data = large_res.get("outputs", [{}])[0] if "outputs" in large_res else large_res
                    generated_text = output_data.get("text", "")
                    
                    # 【关键】获取大模型生成的 Token IDs
                    # 注意：这里的 ID 是基于大模型词表的！
                    new_token_ids = output_data.get("token_ids", None)
                    if new_token_ids is None:
                        # 兜底：必须用【大模型】tokenizer 编码，绝不能用 small!
                        if hasattr(self, 'tokenizer_large'):
                            new_token_ids = self.tokenizer_large.encode(generated_text, add_special_tokens=False)
                        else:
                            # 如果没加载大模型 tokenizer，这是一个严重配置错误，尝试回退或报错
                            raise RuntimeError("Large model returned no token_ids and large tokenizer is not loaded.")
                    
                    # 更新并行数组
                    current_token_ids.extend(new_token_ids)
                    source_flags.extend([1] * len(new_token_ids)) # 标记为 1
                    
                    # 计算置信度 (如果有 logprobs)
                    logprobs_raw = output_data.get("logprobs", [])
                    if logprobs_raw:
                        confidence = self._calculate_confidence(logprobs_raw)
                        if confidence < self.confidence_threshold:
                            print(f"   ⚠️ 大模型置信度低，继续由大模型续写。")
                            force_large_model_this_step = True
                    
                    self.stats["tokens_generated_by_large"] += len(new_token_ids)
                    
                    history_len = len(current_token_ids) - len(new_token_ids)
                    self.stats["prefill_tokens_count"] += history_len
                    
                else:
                    print("   ❌ 大模型也失败了，终止。")
                    break

            # 步骤计数与停止检查逻辑 (保持原样)
            if self.step_tag in generated_text:
                step_count += 1
            
            is_natural_end = (self.step_tag not in generated_text) and (len(generated_text.strip()) > 0)
            has_answer = "Answer:" in generated_text
            
            if has_answer or (is_natural_end and not force_large_model_this_step):
                break

        # ================= 高效混合解码 =================
        return self._decode_mixed_tokens_fast(current_token_ids, source_flags, len(input_ids))

    def _decode_mixed_tokens_fast(self, all_ids: List[int], all_flags: List[int], prompt_len: int) -> str:
        if len(all_ids) <= prompt_len:
            return ""
            
        gen_ids = all_ids[prompt_len:]
        gen_flags = all_flags[prompt_len:]
        
        if not gen_ids:
            return ""
            
        final_text_parts = []
        n = len(gen_ids)
        i = 0
        
        if not hasattr(self, 'tokenizer_large') or self.tokenizer_large is None:
            if 1 in gen_flags:
                raise RuntimeError("Detected large model tokens but large tokenizer is not initialized.")
            return self._decode_tokens(gen_ids)

        while i < n:
            current_flag = gen_flags[i]
            start_idx = i
            while i < n and gen_flags[i] == current_flag:
                i += 1
            end_idx = i
            segment_ids = gen_ids[start_idx:end_idx]

            if current_flag == 0:
                text = self.tokenizer_small.decode(segment_ids, skip_special_tokens=False)
            else:
                text = self.tokenizer_large.decode(segment_ids, skip_special_tokens=False)
            
            final_text_parts.append(text)
            
        return "".join(final_text_parts)

    def generate_until(self, requests: List[Tuple[str, dict]]) -> List[str]:
        results = []
        for req in tqdm(requests, desc="Speculative Routing (Prefill Mode)"):
            context, gen_kwargs = req
            self.stats["total_requests"] += 1
            generated_text = self._run_speculative_routing(context)
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
                print(f"   💡 若无 Prefill，这些切换将产生约 {avg_ctx_len * 0.05:.2f}s 的额外延迟/次 (估算)")

            small_tokens = self.stats['tokens_generated_by_small']
            large_tokens = self.stats['tokens_generated_by_large']
            total_gen_tokens = small_tokens + large_tokens
            print(f"   小模型生成 Token 数: {small_tokens}")
            print(f"   大模型生成 Token 数: {large_tokens}")
            print(f"   小模型承担比例: {small_tokens/total_gen_tokens:.2%}" if total_gen_tokens > 0 else "   无生成数据")
            print("="*40)
            
        return results

    def loglikelihood(self, requests: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Speculative Routing with Prefill currently supports 'generate_until' tasks only.")

    def loglikelihood_rolling(self, requests: List[Tuple[str, str]]) -> List[float]:
        raise NotImplementedError("Not implemented.")