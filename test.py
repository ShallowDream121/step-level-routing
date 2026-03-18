import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class CollaborativeReasonerFixed:
    def __init__(self, small_model_name, large_model_name, threshold=0.50, device="cuda"):
        self.device = device
        self.threshold = threshold
        self.large_model_name = large_model_name
        
        # ✅【关键修复 1】定义 step_tag，防止 AttributeError
        self.step_tag = "</step>"
        
        # --- 1. 加载小模型及其专属 Tokenizer ---
        print(f"⏳ 加载小模型: {small_model_name} ...")
        self.small_tokenizer = AutoTokenizer.from_pretrained(small_model_name, trust_remote_code=True)
        # 重要：设置 pad_token 以避免警告 (不同模型默认值不同)
        if self.small_tokenizer.pad_token is None:
            self.small_tokenizer.pad_token = self.small_tokenizer.eos_token
            
        self.small_model = AutoModelForCausalLM.from_pretrained(
            small_model_name, 
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device,
            trust_remote_code=True
        )
        
        # --- 2. 预加载大模型 Tokenizer (权重延迟加载以节省显存) ---
        print(f"⏳ 预加载大模型 Tokenizer: {large_model_name} ...")
        self.large_tokenizer = AutoTokenizer.from_pretrained(large_model_name, trust_remote_code=True)
        if self.large_tokenizer.pad_token is None:
            self.large_tokenizer.pad_token = self.large_tokenizer.eos_token
            
        self.large_model = None  # 权重初始化为 None
        
        # --- 3. 初始化当前状态为小模型 ---
        self.current_model = self.small_model
        self.current_tokenizer = self.small_tokenizer
        self.current_model_name = "Small"

    def _get_next_token(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.current_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.current_tokenizer.eos_token_id
            )
            
        new_token_id = outputs.sequences[0][-1].item()
        logits = outputs.scores[0][0] 
        
        # 更新 attention_mask (新 token 有效，设为 1)
        new_attention_mask = torch.cat([
            attention_mask, 
            torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        ], dim=1)
        
        return new_token_id, logits, outputs.sequences[0], new_attention_mask

    def _calculate_step_confidence(self, logits_list):
        """计算步骤置信度：取该步骤所有 token 中最低的最大概率"""
        if not logits_list: 
            return 1.0
        
        min_max_prob = 1.0
        for logits in logits_list:
            probs = torch.softmax(logits, dim=-1)
            max_prob = torch.max(probs).item()
            if max_prob < min_max_prob:
                min_max_prob = max_prob
        return min_max_prob

    def _switch_to_large(self):
        """切换到大型模型：同时切换权重和分词器"""
        if self.current_model_name == "Large":
            return 

        print("🔄 正在切换到大模型环境...")
        
        # 1. 如果大模型权重还没加载，现在加载
        if self.large_model is None:
            print(f"⏳ 正在加载大模型权重: {self.large_model_name} (这可能需要几秒)...")
            self.large_model = AutoModelForCausalLM.from_pretrained(
                self.large_model_name, 
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )
            print("✅ 大模型权重加载完成")
        
        # 2. 【关键修复 2】同时切换 模型 和 分词器
        # 必须使用与大模型匹配的 tokenizer，否则会出现乱码和无空格
        self.current_model = self.large_model
        self.current_tokenizer = self.large_tokenizer
        self.current_model_name = "Large"
        
        print(f"✨ 切换成功！当前分词器类型: {type(self.current_tokenizer).__name__}")

    def generate(self, prompt, max_steps=10):
        system_prompt = (
            "You are a reasoning assistant. Solve the problem step by step.\n"
            "Rules:\n"
            "1. Each step must contain actual reasoning or calculation.\n"
            "2. Do NOT output empty steps like '<step></step>'.\n"
            "3. End each step strictly with </step>.\n"
            "Example: <step>First, calculate 2+3=5.</step>"
        )
        full_prompt_text = f"{system_prompt}\nUser: {prompt}\nAssistant:"
        
        history_text = full_prompt_text
        current_step_logits = []
        current_step_text = ""
        step_count = 0
        
        print("\n🚀 开始推理...\n")
        print(f"[策略] 阈值: {self.threshold} | 低分->大模型 | 高分->小模型")
        print(f"[系统] 初始模型: {self.current_model_name}")

        while step_count < max_steps:
            # 重新编码历史文本 (确保使用当前模型的 tokenizer)
            inputs = self.current_tokenizer.encode(history_text, return_tensors="pt").to(self.device)
            attention_mask = torch.ones_like(inputs).to(self.device)
            
            # 生成单个 Token
            new_token_id, logits, new_seq, new_attn_mask = self._get_next_token(inputs, attention_mask)
            new_token_text = self.current_tokenizer.decode([new_token_id])
            
            history_text += new_token_text
            current_step_text += new_token_text
            current_step_logits.append(logits)
            
            print(new_token_text, end="", flush=True)
            
            if new_token_id == self.current_tokenizer.eos_token_id:
                print("\n\n[系统] 生成结束 (EOS)")
                break

            # --- 核心路由逻辑 ---
            # 检查是否包含结束标签，且以结束标签结尾
            if self.step_tag in current_step_text and current_step_text.strip().endswith(self.step_tag):
                
                # 1. 提取纯内容 (去除标签)
                raw_content = current_step_text.replace(self.step_tag, "").strip()
                
                # 2. 【关键修复 3】检查是否为空步骤或无意义步骤
                # 如果内容太短 (例如少于 5 个字符)，视为无效生成，强制低分
                is_meaningful = len(raw_content) > 5 
                
                if not is_meaningful:
                    print(f"\n⚠️ [检测] 步骤内容为空或过短 ('{raw_content}')，强制判定为低置信度!")
                    confidence = 0.0  # 强制给 0 分，触发切换
                else:
                    confidence = self._calculate_step_confidence(current_step_logits)
                
                step_count += 1
                print(f"\n\n[步骤 {step_count} 完成] 置信度: {confidence:.4f} (阈值: {self.threshold})")
                
                # 3. 执行路由决策
                if confidence < self.threshold:
                    # --- 低置信度 -> 切换到大模型 ---
                    if self.current_model_name == "Small":
                        print(f"🔄 [路由] 置信度低! 切换到 **Large** 模型接管...")
                        self._switch_to_large()
                    else:
                        print(f"ℹ️ [路由] 已是大模型，继续生成...")
                else:
                    # --- 高置信度 -> 保持当前模型 ---
                    print(f"✅ [路由] 置信度高，保持 **{self.current_model_name}** 模型")
                
                # 重置当前步骤缓存
                current_step_logits = []
                current_step_text = ""

        return history_text

# 运行配置
if __name__ == "__main__":
    # 请确保路径正确
    SMALL_MODEL = "/home/lcy/models/DeepSeek-R1-Distill-Qwen-1.5B"
    LARGE_MODEL = "/home/lcy/models/DeepSeek-R1-Distill-Llama-8B"
    
    engine = CollaborativeReasonerFixed(
        small_model_name=SMALL_MODEL,
        large_model_name=LARGE_MODEL,
        threshold=0.5,
        device="cuda"
    )

    query = "What's 2+3?"
    result = engine.generate(query)
    print("\n" + "="*30)
    print("最终结果预览:", result[-500:]) # 只打印最后500字避免刷屏