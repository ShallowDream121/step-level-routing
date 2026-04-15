import os
import re
import datasets
import random
from typing import Dict, List, Union
from decimal import Decimal, InvalidOperation

if os.getenv("USE_TEMPLATE", "0") == "1":
    QUERY_TEMPLATE = 'You are a reasoning assistant. You must solve the problem step by step. CRITICAL REQUIREMENT:  '\
    '1.You must use English.' \
    '2.Do not output the final answer until you have gone through all necessary steps.'\
    '3.Choose the best answer from the four choices marked A, B, C, and D.' \
    'Put your final answer within \\boxed{{}}. '\
    'Example format:\n<think>\\n...\\nTherefore, the answer is \\boxed{{A}}. </think> '\
    '{question}\n' \
    'Choice A: {choice1}\n' \
    'Choice B: {choice2}\n' \
    'Choice C: {choice3}\n' \
    'Choice D: {choice4}\n' \
    'Answer:'
else:
    QUERY_TEMPLATE = '{question}'

print("QUERY_TEMPLATE: ", QUERY_TEMPLATE)

def doc_to_text(doc: dict) -> str:
    return QUERY_TEMPLATE.format(question=doc["Question"], choice1=doc["choice1"], choice2=doc["choice2"], choice3=doc["choice3"], choice4=doc["choice4"])

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [
            doc["Incorrect Answer 1"].strip(),
            doc["Incorrect Answer 2"].strip(),
            doc["Incorrect Answer 3"].strip(),
            doc["Correct Answer"].strip(),
        ]

        random.shuffle(choices)
        correct_answer_index = choices.index(doc["Correct Answer"].strip())

        out_doc = {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": f"{chr(65 + correct_answer_index)}",
        }
        return out_doc

    return dataset.map(_process_doc)

def process_results(doc: Dict, results: List[str]):
    ground_truth = str(doc.get("answer", "")).strip().upper()
    raw_pred = results[0] if isinstance(results, list) else results
    extracted_answer = None

    boxed_match = re.search(r"\\boxed\{([^}]+)\}", raw_pred)
    
    if boxed_match:
        content = boxed_match.group(1).strip()
        letter_match = re.search(r'[A-D]', content)
        if letter_match:
            extracted_answer = letter_match.group(0)
        else:
            extracted_answer = content
            
    else:
        answer_patterns = [
            r"(?:final\s+)?(?:the\s+)?answer(?:\s+is)?[:\s]+([A-D])(?:\.|$)",
            r"(?:therefore|thus|hence|so),?\s+(?:the\s+)?answer(?:\s+is)?[:\s]+([A-D])"
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, raw_pred, re.IGNORECASE)
            if match:
                extracted_answer = match.group(1).strip().upper()
                break

    if extracted_answer is None:
        first_line = raw_pred.strip().split('\n')[0].strip()
        if re.match(r'^[A-D]\.?$', first_line):
             extracted_answer = first_line.replace('.', '')
        else:
             extracted_answer = ""

    pred_normalized = extracted_answer.upper()

    is_correct = int(pred_normalized == ground_truth)
    
    return {
        "exact_match": is_correct,
        "extracted_answer!": extracted_answer, 
    }
