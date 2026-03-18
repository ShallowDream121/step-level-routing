import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import re

class CollaborativeReasoner:
    def __init__(self, small_model_name, large_model_name, threshold=0.60, device="cpu"):
        """
        初始化协同推理引擎
        :param small_model_name: 小模型路径 (如 "Qwen/Qwen2.5-0.5B-Instruct")
        :param large_model_name: 大模型路径 (如 "Qwen/Qwen2.5-1.5B-Instruct")
        :param threshold: 置信度阈值 (0.0 - 1.0)
        :param device: 运行设备 ("cpu" 或 "cuda")
        """
        self.device = device
        self.threshold = threshold
        self.step_tag = "</step>"
        
        print(f"⏳ 正在加载小模型: {small_model_name} ...")
        self.small_tokenizer = AutoTokenizer.from_pretrained(small_model_name, trust_remote_code=True)
        self.small_model = AutoModelForCausalLM.from_pretrained(
            small_model_name, 
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        
        print(f"⏳ 正在加载大模型: {large_model_name} ...")
        self.large_tokenizer = AutoTokenizer.from_pretrained(large_model_name, trust_remote_code=True)
        self.large_model = AutoModelForCausalLM.from_pretrained(
            large_model_name, 
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        
        # 统一使用大模型的 tokenizer 进行编码（假设词汇表兼容或大模型包含小模型词汇）
        # 实际生产中建议确保两个模型 tokenizer 一致，或者做 ID 映射
        self.tokenizer = self.large_tokenizer 
        self.current_model = self.small_model
        self.current_model_name = "Small"

    def _get_next_token(self, input_ids):
        """
        使用当前模型生成下一个 token，并返回 token_id 和 logits
        """
        with torch.no_grad():
            outputs = self.current_model.generate(
                input_ids=input_ids,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        new_token_id = outputs.sequences[0][-1].item()
        # scores 是一个 tuple，取第一个元素的第一个 batch 的数据
        logits = outputs.scores[0][0] 
        return new_token_id, logits

    def _calculate_step_confidence(self, logits_list):
        """
        计算一个 Step 的置信度：取该步骤内所有 token 的最大概率的最小值
        """
        if not logits_list:
            return 1.0
        
        min_max_prob = 1.0
        
        for logits in logits_list:
            probs = torch.softmax(logits, dim=-1)
            max_prob = torch.max(probs).item()
            if max_prob < min_max_prob:
                min_max_prob = max_prob
                
        return min_max_prob

    def generate(self, prompt, max_steps=10):
        """
        执行协同推理
        """
        # 1. 构建 Prompt (强制要求分步)
        system_prompt = "You are a reasoning assistant. Solve the problem step by step. End each step with </step>."
        full_prompt = f"{system_prompt}\nUser: {prompt}\nAssistant:"
        
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
        
        generated_text = ""
        current_step_logits = []
        current_step_text = ""
        step_count = 0
        
        print("\n🚀 开始推理...\n")
        print(f"[系统] 初始模型: {self.current_model_name} (阈值: {self.threshold})")

        while step_count < max_steps:
            # 生成单个 Token
            new_token_id, logits = self._get_next_token(input_ids)
            new_token_text = self.tokenizer.decode([new_token_id])
            
            # 更新状态
            generated_text += new_token_text
            current_step_text += new_token_text
            current_step_logits.append(logits)
            
            # 打印流式输出 (不换行)
            print(new_token_text, end="", flush=True)
            
            # 更新输入 IDs 供下一次生成
            input_ids = torch.cat([input_ids, torch.tensor([[new_token_id]]).to(self.device)], dim=1)
            
            # 检查是否结束
            if new_token_id == self.tokenizer.eos_token_id:
                print("\n\n[系统] 生成结束 (EOS)")
                break

            # 检查是否遇到 Step 分隔符
            if self.step_tag in current_step_text:
                # 提取完整的 step 文本 (去除标签后的部分留到下一步)
                # 简单处理：假设标签在末尾
                if current_step_text.strip().endswith(self.step_tag):
                    step_count += 1
                    
                    # --- 核心逻辑：计算置信度并路由 ---
                    confidence = self._calculate_step_confidence(current_step_logits)
                    
                    print(f"\n\n[步骤 {step_count} 完成]")
                    print(f"📊 置信度得分: {confidence:.4f}")
                    
                    if confidence < self.threshold:
                        print(f"⚠️ 置信度低 (< {self.threshold})! 触发路由切换...")
                        if self.current_model_name == "Small":
                            self.current_model = self.large_model
                            self.current_model_name = "Large"
                            print(f"🔄 已切换到模型: {self.current_model_name}")
                        else:
                            print(f"ℹ️ 已是大模型，继续生成...")
                    else:
                        print(f"✅ 置信度高，保持模型: {self.current_model_name}")
                    
                    # 重置 Step 缓存
                    current_step_logits = []
                    current_step_text = ""
                    
                    # 可选：如果切换了模型，这里可以做一个简单的“重新评估”逻辑
                    # 但为了简单，我们直接让新模型基于当前上下文继续生成
                    
        return generated_text

# ==========================================
# 运行示例
# ==========================================
if __name__ == "__main__":
    # 配置模型路径 (为了演示速度，这里用了两个很小的模型)
    # 在实际使用中，请替换为真正的 "Small" (e.g., Qwen2.5-0.5B) 和 "Large" (e.g., Qwen2.5-7B)
    SMALL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    LARGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct" 
    
    # 如果你的显存够大，可以试试：
    # SMALL_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
    # LARGE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

    try:
        engine = CollaborativeReasoner(
            small_model_name=SMALL_MODEL,
            large_model_name=LARGE_MODEL,
            threshold=0.65,  # 设置阈值，低于此值切换大模型
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        query = "If I have 3 apples and buy 5 more, then give away 2, how many do I have? Think step by step."
        result = engine.generate(query)
        
        print("\n" + "="*30)
        print("最终完整回答:")
        print("="*30)
        print(result)
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        print("提示：如果是显存不足 (OOM)，请尝试更小的模型或增加 device='cpu'。")