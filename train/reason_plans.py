import os
import json
import re 
import torch
from typing import List, Dict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import dashscope

dashscope.api_key = "sk-74e6c42f8639444eacb98a241bdd09e9"

SYSTEM_PROMPT_aime = """
You are an expert Task Decomposition Engine specialized in training small AI models (1.5B parameters). Your goal is to break down problems into a sequence of extremely simple, atomic, and executable steps.

Core Principles:
1.  **Atomicity:** Each step must perform exactly ONE logical operation or calculation.
    -   Bad: "Calculate the total and print the answer." (Two actions)
    -   Good: "Calculate the total." -> "State the final answer."
2.  **Explicitness:** Do not skip intermediate calculations. Even simple arithmetic should be a step if it's part of the logic flow.
3.  **Clarity:** Use concise language. Prefer mathematical expressions over long descriptions where possible.
4.  **No Redundancy:** Do not repeat the result in the instruction (e.g., instead of "Calculate 2+2 which is 4", say "Calculate 2+2").

### Output Format:
Strictly wrap the steps between `START` and `END`. Do not output any conversational text.

START
1. [Identify variable A from text]
2. [Identify variable B from text]
3. [Calculate intermediate value C using A and B]
4. [Final calculation]
END

### Examples:

**Input:** Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether?
**Output:**
START
1. Identify the number of clips sold in April: 48
2. Calculate the number of clips sold in May: 48 / 2
3. Calculate the total number of clips: 48 + (result of step 2)
4. The answer is the result of step 3.
END

**Input:** Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
**Output:**
START
1. Identify the hourly rate: 12
2. Identify the time worked in minutes: 50
3. Convert time to hours: 50 / 60
4. Calculate earnings: 12 * (result of step 3)
5. The answer is the result of step 4.
END
"""

SYSTEM_PROMPT_aime_new = """
Decompose the math problem into simple steps wrapped in START and END.

Example:
Input: Natalia sold 48 clips in April and half as many in May. Total?
Output:
START
1. Identify April sales: 48
2. Calculate May sales: 48 / 2
3. Calculate total: 48 + May sales
END
"""

class AIMEInference:
    def __init__(self, base_model_path: str, lora_weights_path: str, output_file: str = "results_structured.json"):
        self.output_file = output_file
        self.device = "cuda"
        
        print(f"正在加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(self.model, lora_weights_path)
        self.model.eval()
        print("模型加载完成！")

    def format_prompt(self, mode: int ,doc: Dict) -> str:
        prompt = ""
        if mode == 1:
            problem = doc.get("problem")
            prompt = f"{SYSTEM_PROMPT_aime_new}\nQuestion:{problem}\nOutput:\n"
        elif mode == 2:
            problem = doc.get("Question")
            prompt = f"{SYSTEM_PROMPT_aime_new}\nQuestion:{problem}\nOutput:\n"
        else:
            prompt = doc.get("problem")
        return prompt

    def generate(self, prompt: str, max_new_tokens: int = 2048) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response

    def parse_steps(self, text: str) -> List[str]:
        steps = []
        match = re.search(r"START\s*(.*?)\s*END", text, re.DOTALL)
        
        if match:
            content = match.group(1)
            steps = [line.strip() for line in content.split('\n') if line.strip()]
        return steps

    def run(self):
        print("正在加载数据集...")
        try:
            full_dataset = load_dataset("simplescaling/aime24_nofigures", None, split="train")
            dataset = full_dataset
        except Exception as e:
            print(f"数据集加载失败: {e}")
            return

        results = []
        
        print("开始处理...")
        for item in tqdm(dataset):
            problem_id = item.get('id', 'unknown')
            
            prompt = self.format_prompt(1, item)
            raw_response = self.generate(prompt)
            
            parsed_steps = self.parse_steps(raw_response)
            
            result_item = {
                "id": problem_id,
                "problem": item.get("problem"),
                "raw_response": raw_response,   
                "steps": parsed_steps   
            }
            results.append(result_item)

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"处理完成！结果已保存至 {self.output_file}")

if __name__ == "__main__":
    BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    LORA_PATH = "./saves/deepseek-qwen-distill-7b-gsm8k-lora-v3/" 
    OUTPUT_FILE = "../plans/aime_plans_deepseek_7B.json"
    
    if os.path.exists(LORA_PATH):
        runner = AIMEInference(BASE_MODEL, LORA_PATH, OUTPUT_FILE)
        runner.run()
    else:
        print(f"错误：找不到 LoRA 权重路径: {LORA_PATH}")