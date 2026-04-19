import json
import os
import time
import dashscope
from datasets import load_dataset

dashscope.api_key = "sk-74e6c42f8639444eacb98a241bdd09e9"

SYSTEM_PROMPT = """
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


def generate_reasoning_data(question, max_retries=3):
    full_prompt = f"{SYSTEM_PROMPT}\nQuestion: {question}"
    
    for attempt in range(max_retries):
        try:
            response = dashscope.Generation.call( 
                model="qwen-max-0919", 
                messages=[{'role': 'user', 'content': full_prompt}]
            )
                        
            if response.status_code == 200:
                return response.output.text
            else:
                print(f"  ❌ API Error: {response.code} - {response.message}")
                time.sleep(2)
                
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            time.sleep(2)
            
    return None

# ================= 主程序 =================

print("正在加载 MATH 数据集...")
try:
    raw_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test") 
    math_train_data = raw_dataset.to_list()
    print(f"数据集加载成功，共 {len(math_train_data)} 条数据。")
except Exception as e:
    print(f"数据集加载失败: {e}")
    exit()

processed_data = []
output_file = "train_data_math.json"

target_data = math_train_data

print(f"开始处理数据...")

for i, item in enumerate(target_data):
    q = item['problem']
    print(f"[{i+1}/{len(target_data)}] 处理中: {q[:40]}...")

    formatted_output = generate_reasoning_data(q)
    
    if formatted_output:
        data_entry = {
            "instruction": SYSTEM_PROMPT.strip(), 
            "input": q,
            "output": formatted_output
        }
        processed_data.append(data_entry)

        if (i + 1) % 10 == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            print(f"已保存 {len(processed_data)} 条数据到 {output_file}")

        time.sleep(0.5) 
    else:
        print(f" 第 {i+1} 条数据处理失败，跳过。")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=2)

print(f"全部完成！共生成 {len(processed_data)} 条有效数据。")
print(f"文件已保存为: {output_file}")