import os
import re
import sys
import json
import subprocess
import tempfile
import logging
from typing import Dict, List, Tuple

from datasets import Dataset

logger = logging.getLogger(__name__)


# ============================================================
# Prompt Template
# ============================================================

if os.getenv("USE_TEMPLATE", "0") == "1":
    QUERY_TEMPLATE = (
        "You are a coding expert. Solve the following competitive programming problem.\n"
        "CRITICAL REQUIREMENT:\n"
        "1. Think step by step about the problem requirements and algorithm.\n"
        "2. Write the complete, runnable Python code as your solution.\n"
        "3. Put your final code inside \\boxed{{}} using ```python code blocks.\n"
        "4. End your response with </think>\n\n"
        "Example format:\n"
        "<think>\n"
        "[Step-by-step analysis]\n"
        "Therefore, the solution is \\boxed{{```python\n"
        "def task_func(arg):\n"
        "    # your implementation\n"
        "    return result\n"
        "```}} </think>\n\n"
        "Problem:\n{question}\n"
    )
else:
    QUERY_TEMPLATE = "{question}"

print("LIVECODEBENCH QUERY_TEMPLATE:", QUERY_TEMPLATE[:120])


def doc_to_text(doc: dict) -> str:
    question = doc.get("question_content", "")
    starter = doc.get("starter_code", "")

    if starter:
        question = f"{question}\n\nStarter code:\n```python\n{starter}\n```"

    return QUERY_TEMPLATE.format(question=question)


# ============================================================
# Hard-subset filter
# ============================================================

def process_docs(dataset: Dataset) -> Dataset:
    def _is_hard(doc):
        return doc.get("difficulty", "") == "hard"

    hard = dataset.filter(_is_hard)
    logger.info(
        f"LiveCodeBench-Hard: {len(hard)} problems filtered from {len(dataset)} total"
    )
    return hard


# ============================================================
# Code Extraction
# ============================================================

def extract_code(text: str) -> str:
    """
    Extract Python code from model output.
    Priority: \\boxed{...} > markdown code blocks > code heuristics > raw text.
    """
    # Priority 1: code inside \boxed{}
    boxed_contents = re.findall(
        r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}",
        text, re.DOTALL,
    )
    if boxed_contents:
        content = boxed_contents[-1]
        code_blocks = re.findall(
            r"```(?:python|py)?\s*\n(.*?)```", content, re.DOTALL
        )
        if code_blocks:
            return code_blocks[0].strip()
        stripped = content.strip()
        if stripped.startswith("python") and "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]
        return stripped

    # Priority 2: standalone markdown code blocks
    code_blocks = re.findall(
        r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL
    )
    if code_blocks:
        return code_blocks[-1].strip()

    # Priority 3: code-like content
    code_match = re.search(
        r"((?:def|class|import|from)\s+\w+[\s\S]*)",
        text,
    )
    if code_match:
        return code_match.group(1).strip()

    return text.strip()


# ============================================================
# Entry-point inference
# ============================================================

def _infer_entry(starter_code: str, meta_data) -> Tuple[bool, str, str]:
    """
    Infer how to invoke the solution.
    Returns (is_class: bool, class_or_func_name: str, method_name: str).
    """
    starter = starter_code or ""

    is_class = bool(re.search(r'^\s*class\s+\w+', starter))

    if is_class:
        class_match = re.search(r"class\s+(\w+)", starter)
        class_name = class_match.group(1) if class_match else "Solution"

        method_name = "solve"
        if meta_data:
            try:
                md = json.loads(meta_data) if isinstance(meta_data, str) else meta_data
                if isinstance(md, dict) and "func_name" in md:
                    method_name = md["func_name"]
            except (json.JSONDecodeError, TypeError):
                pass
        return True, class_name, method_name

    else:
        func_match = re.search(r"def\s+(\w+)", starter)
        func_name = func_match.group(1) if func_match else "task_func"
        return False, func_name, ""


# ============================================================
# Sandboxed Code Execution (JSON test cases)
# ============================================================

EXECUTE_TEMPLATE = """
import sys
import json
import io

sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

# --- Generated code ---
{code}

# --- Test harness ---
_test_cases = json.loads({test_cases_json!r})
_passed = 0

for _i, _tc in enumerate(_test_cases):
    try:
        _inp = _tc["input"]
        _expected = _tc["output"]
        _result = {call_expr}
        if _result == _expected:
            _passed += 1
    except Exception:
        pass

sys.__stdout__.write(f"__PASS__:{{_passed}}/{{len(_test_cases)}}")
sys.exit(0 if _passed == len(_test_cases) else 1)
"""


def execute_code(
    code: str,
    test_cases_json: str,
    starter_code: str,
    meta_data,
    timeout: int = 30,
) -> Tuple[bool, str, str]:
    """Run generated code against JSON test cases in a subprocess sandbox."""

    if not test_cases_json:
        return False, "", "No test cases provided"

    # Parse test cases to ensure valid JSON
    try:
        _ = json.loads(test_cases_json)
    except json.JSONDecodeError as e:
        return False, "", f"Invalid test cases JSON: {e}"

    is_class, name, method = _infer_entry(starter_code, meta_data)

    if is_class:
        call_expr = (
            f"{name}().{method}(*_inp) "
            "if isinstance(_inp, list) else "
            f"{name}().{method}(_inp)"
        )
    else:
        call_expr = (
            f"{name}(*_inp) "
            "if isinstance(_inp, list) else "
            f"{name}(_inp)"
        )

    script = EXECUTE_TEMPLATE.format(
        code=code,
        test_cases_json=test_cases_json,
        call_expr=call_expr,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
        passed = result.returncode == 0
        return passed, result.stdout or "", result.stderr or ""

    except subprocess.TimeoutExpired:
        return False, "", f"Timeout after {timeout}s"
    except Exception as e:
        return False, "", f"Execution error: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ============================================================
# Result Processing
# ============================================================

def process_results(doc: Dict, results: List[str]):
    raw_pred = results[0] if isinstance(results, list) else results

    code = extract_code(raw_pred)
    test_cases_json = doc.get("public_test_cases", "")
    starter_code = doc.get("starter_code", "")
    meta_data = doc.get("meta_data", None)

    if not code:
        return {
            "pass_at_1": 0,
            "extracted_code!": "",
        }

    passed, stdout, stderr = execute_code(
        code, test_cases_json, starter_code, meta_data, timeout=30
    )

    return {
        "pass_at_1": int(passed),
        "extracted_code!": code,
    }
