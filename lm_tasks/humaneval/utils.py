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
        "You are a coding expert. Complete the Python function below by writing only the function body.\n"
        "CRITICAL REQUIREMENT:\n"
        "1. Think step by step about the function logic.\n"
        "2. Write the complete function including its definition.\n"
        "3. Put your final code inside \\boxed{{}} using ```python code blocks.\n"
        "4. End your response with </think>\n\n"
        "Example format:\n"
        "<think>\n"
        "[Step-by-step analysis]\n"
        "Therefore, the solution is \\boxed{{```python\n"
        "def func_name(args):\n"
        "    # your implementation\n"
        "    return result\n"
        "```}} </think>\n\n"
        "Question: Complete the following function:\n{question}\n"
    )
else:
    QUERY_TEMPLATE = "{question}"

print("HUMANEVAL QUERY_TEMPLATE:", QUERY_TEMPLATE[:120])


def doc_to_text(doc: dict) -> str:
    return QUERY_TEMPLATE.format(question=doc.get("prompt", ""))


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
# Sandboxed Code Execution (assertion-based)
# ============================================================

EXECUTE_TEMPLATE = """
import sys
import io
import math
import itertools
import collections
import functools
import heapq
import re as _re_mod
from typing import (
    List, Tuple, Dict, Set, Optional, Union, Any,
    Callable, Iterator, Generator, Iterable, Sequence,
)

sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

# --- Generated code ---
{code}

# --- Test harness ---
{test}

check({entry_point})
sys.__stdout__.write("__PASS__")
"""


def execute_code(
    code: str,
    prompt_header: str,
    test: str,
    entry_point: str,
    timeout: int = 15,
) -> Tuple[bool, str, str]:
    """Run generated code against assertion-based test harness."""

    if not test:
        return False, "", "No test code provided"

    # If the extracted code is only a body (no def/class), prepend the prompt header
    if not re.match(r'^\s*(def|class)\s', code):
        code = prompt_header + "\n" + code

    script = EXECUTE_TEMPLATE.format(
        code=code,
        test=test,
        entry_point=entry_point,
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
        passed = result.returncode == 0 and "__PASS__" in (result.stdout or "")
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
    prompt_header = doc.get("prompt", "")
    entry_point = doc.get("entry_point", "solution")

    if not code:
        return {
            "pass_at_1": 0,
            "extracted_code!": "",
        }

    passed, stdout, stderr = execute_code(
        code, prompt_header, test_code, entry_point, timeout=15
    )

    return {
        "pass_at_1": int(passed),
        "extracted_code!": code,
    }
