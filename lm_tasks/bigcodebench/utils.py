import os
import re
import sys
import subprocess
import tempfile
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# Prompt Template
# ============================================================

if os.getenv("USE_TEMPLATE", "0") == "1":
    QUERY_TEMPLATE = (
        "You are a coding expert. Complete the Python function described below.\n"
        "CRITICAL REQUIREMENT:\n"
        "1. Think step by step about what the function needs to do.\n"
        "2. Write the complete, runnable Python code including the function definition.\n"
        "3. Put your final code inside \\boxed{{}} using ```python code blocks.\n"
        "4. End your response with </think>\n\n"
        "Example format:\n"
        "<think>\n"
        "[Step-by-step analysis]\n"
        "Therefore, the solution code is \\boxed{{```python\n"
        "def task_func(args):\n"
        "    # your implementation here\n"
        "    return result\n"
        "```}} </think>\n\n"
        "Requirements: {libs_note}\n"
        "Question:\n{question}\n"
    )
else:
    QUERY_TEMPLATE = "{question}"

print("BIGCODEBENCH QUERY_TEMPLATE:", QUERY_TEMPLATE[:120])


def doc_to_text(doc: dict) -> str:
    question = doc.get("complete_prompt", "")
    libs = doc.get("libs", "")

    if libs:
        libs_note = f"You may use the following libraries: {libs}"
    else:
        libs_note = "Use only Python standard library unless specified otherwise."

    return QUERY_TEMPLATE.format(question=question, libs_note=libs_note)


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

    # Priority 3: extract code-like content starting from def/class/import
    code_match = re.search(
        r"((?:def|class|import|from)\s+\w+[\s\S]*)",
        text,
    )
    if code_match:
        return code_match.group(1).strip()

    # Fallback: entire output
    return text.strip()


# ============================================================
# Sandboxed Code Execution (unittest-based)
# ============================================================

EXECUTE_TEMPLATE = """
import sys
import unittest
import io

# Suppress all output during test execution
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

# --- Generated code ---
{code}

# --- Test harness ---
{test}

if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=0, exit=True)
"""


def execute_code(
    code: str,
    test: str,
    timeout: int = 30,
) -> Tuple[bool, str, str]:
    """Run generated code against unittest-based test harness."""

    if not test:
        return False, "", "No test code provided"

    script = EXECUTE_TEMPLATE.format(code=code, test=test)

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
    test_code = doc.get("test", "")

    if not code:
        return {
            "pass_at_1": 0,
            "extracted_code!": "",
        }

    passed, stdout, stderr = execute_code(code, test_code, timeout=30)

    return {
        "pass_at_1": int(passed),
        "extracted_code!": code,
    }
