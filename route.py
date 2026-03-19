import os
import requests
import math
import time
import json
from typing import List, Optional, Dict, Any, Tuple
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
# 需要安装 transformers 来获取 tokenizer，用于字符串<->ID 转换
from transformers import AutoTokenizer 

@register_model("vllm_spec_router_prefill")
class VLLMSpeculativeRouterPrefill(LM):
    def __init__(
        self,
        small_model_url: str,
        large_model_url: str,
        # 新增：需要知道模型的 tokenizer 路径，以便在本地进行 ID 管理
        # 通常小模型和大模型如果是同系列 (如 Opt-125m 和 Opt-6.7b)，tokenizer 是通用的
        # 如果不同，建议以小模型的 tokenizer 为准，或者分别初始化
        tokenizer_name: str = "facebook/opt-125m", 
        confidence_threshold: float = 0.6,
        max_step_tokens: int = 256,
        max_gen_tokens: int = 2048,
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
        
        # 初始化 Tokenizer (本地运行，用于 ID <-> Text 转换)
        print(f"🔄 正在加载 Tokenizer: {tokenizer_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
        self.stats = {
            "total_requests": 0,
            "small_only_success": 0,
            "fallback_to_large": 0,
            "small_failures": 0,
            "prefill_tokens_count": 0 # 统计通过 Prefill 加速处理的 token 总数
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
        """
        发送请求到 vLLM API。
        支持直接传入 prompt_token_ids 以触发 Prefill 优化。
        """
        payload = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "stop": stop_sequences,
            "stream": False
        }
        
        # 优先使用 token_ids，如果没有则使用 text
        if prompt_token_ids is not None:
            payload["prompt_token_ids"] = prompt_token_ids
            # 当使用 token_ids 时，通常不需要 prompt 文本，但某些 vLLM 版本可能需要占位符
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
            # 获取实际生成 token 的 logprob
            # vLLM 返回格式通常是: [ { "token": "...", "logprob": -0.1, "top_logprobs": [...] }, ... ]
            # 注意：lm_eval 或不同 vLLM 版本返回格式可能微调，这里做兼容处理
            if isinstance(token_logprobs, dict):
                # 兼容旧格式或特定封装
                best_token_info = list(token_logprobs.values())[0] if token_logprobs else None
                if best_token_info and "logprob" in best_token_info:
                    probs.append(math.exp(best_token_info["logprob"]))
            elif isinstance(token_logprobs, list) or (isinstance(token_logprobs, dict) and "logprob" in token_logprobs):
                 # 标准 vLLM 返回结构通常是列表中的字典，或者直接是包含 logprob 的字典
                 # 假设这里拿到的是单个 token 的详细信息字典
                 logprob = token_logprobs.get("logprob")
                 if logprob is not None:
                     probs.append(math.exp(logprob))
            else:
                # 尝试从 top_logprobs 提取第一个 (rank 1)
                if "top_logprobs" in token_logprobs and len(token_logprobs["top_logprobs"]) > 0:
                    logprob = token_logprobs["top_logprobs"][0].get("logprob")
                    if logprob is not None:
                        probs.append(math.exp(logprob))

        if not probs:
            return 0.0
            
        return sum(probs) / len(probs)

    def _decode_tokens(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """本地将 token IDs 解码为字符串"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def _run_speculative_routing(self, prompt: str) -> str:
        """
        【方案 B: Batch Prefill 加速版】
        核心逻辑：
        1. 维护全局 current_token_ids。
        2. 小模型生成 -> 获取新生成的 token IDs -> 拼接到全局列表。
        3. 若置信度低 -> 丢弃小模型最后生成的那部分 (可选) 或 直接基于当前全局列表调用大模型。
        4. 大模型调用 -> 传入完整的 current_token_ids -> vLLM 自动执行 Prefill -> 生成新 token -> 拼接。
        """
        # 1. 初始化：将 Prompt 转换为 Token IDs
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        current_token_ids = list(input_ids)
        
        max_steps = 20
        step_count = 0
        global_stop_sequences = ["</s>", "<|eot_id|>", "\n\nUser:", "User:"]
        
        # 预编码 stop token 以便检测
        step_tag_id = self.tokenizer.encode(self.step_tag, add_special_tokens=False)
        # 简单处理：假设 step_tag 是单个 token 或多个，这里主要靠文本检测辅助
        
        print(f"🚀 开始推理，初始 Prompt 长度: {len(current_token_ids)} tokens")

        while step_count < max_steps:
            # --- 步骤 1: 小模型试探 ---
            # 构造停止条件：遇到 step_tag 或全局停止符
            # 注意：vLLM API 的 stop 参数接受字符串，它会在解码后匹配
            step_stops = [self.step_tag] + global_stop_sequences
            
            # 记录当前长度，以便计算小模型生成了多少
            len_before_small = len(current_token_ids)
            
            small_res = self._call_vllm_api(
                url=self.small_url,
                prompt_token_ids=current_token_ids, # 【关键】直接传 ID，避免重复分词
                max_tokens=self.max_step_tokens,
                stop_sequences=step_stops,
                get_logprobs=True,
                temperature=0.7
            )
            
            use_large_model = False
            generated_segment_ids = []
            confidence = 0.0
            
            if small_res:
                # 解析小模型结果
                # vLLM 返回结构: { "text": "...", "tokens": [...], "logprobs": [...] }
                # 注意：不同 vLLM 版本返回字段略有不同，这里假设返回了 tokens 或可以通过 text 反推
                # 最稳妥的方式：API 返回新生成的 token IDs。如果 API 没返回 IDs，只能用 text 再 encode (有损耗风险)
                # 假设 vLLM 返回了 "tokens" 字段 (列表 of token IDs) 或者 "output"->"token_ids"
                
                output_data = small_res.get("outputs", [{}])[0] if "outputs" in small_res else small_res
                
                generated_text = output_data.get("text", "")
                # 尝试获取 token IDs (vLLM 较新版本支持返回 token_ids)
                new_token_ids = output_data.get("token_ids", None)
                
                # 如果 API 没返回 IDs，我们只能本地重新 encode (注意：这可能导致细微的分词对齐问题，但在同一 tokenizer 下通常安全)
                if new_token_ids is None:
                    # 为了精确，我们 encode 整个返回文本，然后减去输入？
                    # 更好的方法：直接 encode 生成的文本片段
                    # 但要注意 special tokens。这里简化处理：
                    new_token_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)
                
                logprobs_raw = output_data.get("logprobs", [])
                
                # 计算置信度
                confidence = self._calculate_confidence(logprobs_raw)
                
                generated_segment_ids = new_token_ids
                
                # 检查是否自然结束 (没有 step_tag 且长度很短，或者包含 Answer)
                is_natural_end = (self.step_tag not in generated_text) and (len(generated_text.strip()) > 0)
                has_answer = "Answer:" in generated_text
                
                if confidence < self.confidence_threshold:
                    use_large_model = True
                    print(f"   [Step {step_count+1}] 小模型置信度 ({confidence:.2f}) 低 -> 准备切换 (已生成 {len(generated_segment_ids)} tokens)")
                    # 策略：丢弃小模型这次生成的低质量 token，让大模型基于之前的上下文重新生成
                    # 如果你希望保留小模型的前半部分，可以在这里切片，但为了逻辑简单，我们整步回退
                    generated_segment_ids = [] 
                else:
                    # 小模型合格，采纳
                    current_token_ids.extend(generated_segment_ids)
                    print(f"   [Step {step_count+1}] 小模型置信度 ({confidence:.2f}) 高 -> 采纳 (累计 tokens: {len(current_token_ids)})")
                    
                    if self.step_tag in generated_text:
                        step_count += 1
                    
                    if is_natural_end or has_answer:
                        break
                    continue # 进入下一步

            else:
                # 小模型失败
                use_large_model = True
                print(f"   [Step {step_count+1}] 小模型调用失败 -> 切换")
                generated_segment_ids = []

            # --- 步骤 2: 大模型接管 (Batch Prefill 核心) ---
            if use_large_model:
                self.stats["fallback_to_large"] += 1
                
                # 【关键加速点】
                # 我们将 current_token_ids (包含 prompt + 之前所有正确的 steps) 
                # 一次性传给大模型。
                # vLLM 会将这 N 个 token 视为 Prompt，使用 Prefill 阶段并行计算 KV Cache。
                # 然后从第 N+1 个位置开始 Decode 生成新内容。
                
                print(f"   [Step {step_count+1}] 🚄 触发大模型 Prefill (上下文长度: {len(current_token_ids)} tokens)...")
                start_time = time.time()
                
                large_res = self._call_vllm_api(
                    url=self.large_url,
                    prompt_token_ids=current_token_ids, # 【核心】传入完整历史 ID
                    max_tokens=self.max_step_tokens, # 大模型也只生成一步，保持粒度一致
                    stop_sequences=step_stops,
                    get_logprobs=False, # 大模型生成时通常不需要 logprobs，除非你要二次校验
                    temperature=0.7
                )
                
                elapsed = time.time() - start_time
                print(f"   ⏱️ 大模型 Prefill + Decode 耗时: {elapsed:.3f}s")
                
                if large_res:
                    output_data = large_res.get("outputs", [{}])[0] if "outputs" in large_res else large_res
                    generated_text = output_data.get("text", "")
                    new_token_ids = output_data.get("token_ids", None)
                    
                    if new_token_ids is None:
                        new_token_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)
                    
                    # 将大模型生成的内容加入历史
                    current_token_ids.extend(new_token_ids)
                    self.stats["prefill_tokens_count"] += len(current_token_ids) # 记录加速的 token 数
                    
                    print(f"   [Step {step_count+1}] 大模型生成完毕 (新增 {len(new_token_ids)} tokens)")
                    
                    if self.step_tag in generated_text:
                        step_count += 1
                    
                    if "Answer:" in generated_text or (len(new_token_ids) < 5 and self.step_tag not in generated_text):
                        break
                else:
                    print("   ❌ 大模型也失败了，终止。")
                    break
        
        # 最终解码
        final_text = self.tokenizer.decode(current_token_ids[len(input_ids):], skip_special_tokens=True)
        return final_text

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
            print("📊 路由统计 (Prefill 加速版):")
            print(f"   总请求数: {total}")
            print(f"   切换到大模型次数: {self.stats['fallback_to_large']}")
            print(f"   累计 Prefill 加速 Token 数: {self.stats['prefill_tokens_count']}")
            if self.stats['fallback_to_large'] > 0:
                avg_ctx_len = self.stats['prefill_tokens_count'] / self.stats['fallback_to_large']
                print(f"   平均每次切换的上下文长度: {avg_ctx_len:.1f} tokens")
                print(f"   💡 若无 Prefill，这些切换将产生约 {avg_ctx_len * 0.05:.2f}s 的额外延迟/次 (估算)")
            print("="*40)
            
        return results

    def loglikelihood(self, requests: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Speculative Routing with Prefill currently supports 'generate_until' tasks only.")

    def loglikelihood_rolling(self, requests: List[Tuple[str, str]]) -> List[float]:
        raise NotImplementedError("Not implemented.")