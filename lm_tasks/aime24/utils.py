import os
import re
from typing import Dict, List, Union
from decimal import Decimal, InvalidOperation

if os.getenv("SMALL_ONLY", "0") == "1" or os.getenv("LARGE_ONLY", "0") == "1":
    QUERY_TEMPLATE = 'You are a reasoning assistant. You must solve the problem step by step. CRITICAL REQUIREMENT:  '\
    '1.You must use English.' \
    '2.Do not output the final answer until you have gone through all necessary steps.'\
    'Put your final answer within \\boxed{{}}. '\
    'Example format:\n<think>\\n...\\nTherefore, the answer is \\boxed{{204}}. </think> '\
    '{question}' \
    'Answer:'
else:
    QUERY_TEMPLATE = '{question}'

print("QUERY_TEMPLATE: ", QUERY_TEMPLATE)

def doc_to_text(doc: dict) -> str:
    return QUERY_TEMPLATE.format(question=doc.get("problem"))

def _normalize_answer(ans: str) -> str:
    ans = ans.strip().rstrip('.')
    frac_match = re.match(r"^\\frac\{([^}]+)\}\{([^}]+)\}$", ans)
    if frac_match:
        numerator = int(frac_match.group(1))
        denominator = int(frac_match.group(2))
        if denominator != 0:
            ans = str(numerator / denominator)

    try:
        val = Decimal(ans)

        if val == val.to_integral_value():
            return str(int(val))
        else:
            return str(val.normalize())
    except (ValueError, InvalidOperation):
        return ans.lower().replace(" ", "")

def process_results(doc: Dict, results: List[str]):
    ground_truth = _normalize_answer(str(doc.get("answer", "")))
    
    raw_pred = results[0] if isinstance(results, list) else results
    
    extracted_answer = None
    
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", raw_pred)
    if boxed_match:
        extracted_answer = boxed_match.group(1).strip()

    else:
        answer_patterns = [
            r"(?:final\s+)?(?:the\s+)?answer(?:\s+is)?[:\s]+(.+?)(?:\.|$)",

            r"(?:therefore|thus|hence|so),?\s+(?:the\s+)?answer(?:\s+is)?[:\s]+(.+?)(?:\.|$)"
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, raw_pred, re.IGNORECASE)
            if match:
                extracted_answer = match.group(1).strip()
                if extracted_answer.endswith('.'):
                    extracted_answer = extracted_answer[:-1]
                break
    

    if extracted_answer is None:
        extracted_answer = raw_pred

    pred_normalized = _normalize_answer(extracted_answer)

    is_correct = int(pred_normalized == ground_truth)
    
    return {
        "exact_match": is_correct,
        "extracted_answer!": extracted_answer
    }