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
        "You are a coding expert. Solve the following programming problem by writing correct Python code.\n"
        "CRITICAL REQUIREMENT:\n"
        "1. Think step by step about the problem requirements and approach.\n"
        "2. Write the complete, runnable Python code as your solution.\n"
        "3. Put your final code inside \\boxed{{}}.\n"
        "4. End your response with </think>\n\n"
        "Example format:\n"
        "<think>\n"
        "[Step-by-step analysis and reasoning]\n"
        "Therefore, the solution code is \\boxed{{```python\n"
        "def solution(arg):\n"
        "    # your implementation\n"
        "    return result\n"
        "```}} </think>\n\n"
        "Question: {question}\n"
    )
else:
    QUERY_TEMPLATE = "{question}"

print("BIGCODEBENCH QUERY_TEMPLATE:", QUERY_TEMPLATE[:120])


def doc_to_text(doc: dict) -> str:
    return QUERY_TEMPLATE.format(question=doc.get("prompt", ""))


# ============================================================
# Code Extraction
# ============================================================

def extract_code(text: str) -> str:
    """
    Extract Python code from model output.
    Priority: \\boxed{...} > markdown code blocks > raw text.
    """
    # Priority 1: code inside \boxed{}
    boxed_contents = re.findall(
        r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}",
        text, re.DOTALL,
    )
    if boxed_contents:
        content = boxed_contents[-1]  # last boxed match
        code_blocks = re.findall(
            r"```(?:python|py)?\s*\n(.*?)```", content, re.DOTALL
        )
        if code_blocks:
            return code_blocks[0].strip()
        # boxed but no code block - strip common wrappers and return
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

    # Priority 3: try to find code-like content (def/class/import)
    code_indicators = re.search(
        r"((?:def\s+\w+|class\s+\w+|import\s+\w+|from\s+\w+\s+import).*)",
        text, re.DOTALL,
    )
    if code_indicators:
        return code_indicators.group(1).strip()

    # Fallback: entire output
    return text.strip()


# ============================================================
# Sandboxed Code Execution
# ============================================================

SANDBOX_HEADER = """
import sys
import signal
import os

class _LimitedOut:
    def __init__(self, limit=100000):
        self.limit = limit
        self.written = 0
    def write(self, s):
        if self.written < self.limit:
            sys.__stdout__.write(s)
            self.written += len(s)
    def flush(self):
        sys.__stdout__.flush()

sys.stdout = _LimitedOut()
sys.stderr = _LimitedOut()
"""


def execute_code(
    code: str,
    test_code: str,
    timeout: int = 30,
) -> Tuple[bool, str, str]:
    """
    Run code + test in a subprocess sandbox.
    Returns (passed, stdout, stderr).
    """

    if not test_code:
        return False, "", "No test code provided"

    full_script = f"{SANDBOX_HEADER}\n\n{code}\n\n{test_code}\n"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(full_script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
        passed = result.returncode == 0 and "AssertionError" not in result.stdout and "AssertionError" not in result.stderr
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
