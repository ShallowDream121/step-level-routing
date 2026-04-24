import os
import json
import re
from typing import List, Dict
from tqdm import tqdm
from datasets import load_dataset
import dashscope

# 设置你的 API Key
dashscope.api_key = "sk-74e6c42f8639444eacb98a241bdd09e9"

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
    def __init__(self, output_file: str = "results_structured.json"):
        self.output_file = output_file
        print("DashScope 客户端初始化完成！")

    def format_prompt(self, mode: int, doc: Dict) -> str:
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

    def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        """
        使用 DashScope 的 Qwen-Max-0919 模型进行推理
        """
        try:
            response = dashscope.Generation.call(
                model="qwen-max-0919",
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=max_tokens,
                temperature=0.0,  # 保持确定性输出，类似 do_sample=False
                top_p=1.0,
                result_format='message'
            )

            if response.status_code == 200:
                # 提取回复内容
                content = response.output.choices[0].message.content
                return content
            else:
                print(f"API 调用失败: {response.code} - {response.message}")
                return ""
        except Exception as e:
            print(f"调用 DashScope 出错: {e}")
            return ""

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
    OUTPUT_FILE = "../plans/aime_plans_qwen_max.json"
    
    runner = AIMEInference(OUTPUT_FILE)
    runner.run()