"""
analyst_agent.py
-----------------

The orchestration loop: given a natural-language question and a real
DataFrame, ask an LLM to write Python analysis code, run it safely, and
if it fails (unsafe, malformed, or a runtime error), feed the error back
to the LLM and ask for a corrected version -- up to a fixed retry limit.
This retry loop is what makes the pattern genuinely "agentic" rather
than a single prompt-and-run call: the agent reacts to its own tool's
real output, not just to what the LLM said would happen.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.code_generator import build_codegen_prompt, extract_code
from src.llm_client import LLMClient
from src.sandbox_executor import execute_analysis_code

import pandas as pd

RETRY_PROMPT_TEMPLATE = """Your previous code failed with this error:

{error}

Here was your previous code:
```python
{previous_code}
```

Please write a corrected version. Respond with a single Python code \
block, nothing else.
"""


@dataclass
class AnalysisAttempt:
    code: str
    success: bool
    result: object
    error: str | None


@dataclass
class AnalysisRunResult:
    question: str
    attempts: list = field(default_factory=list)
    final_success: bool = False
    final_result: object = None

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)


def run_analysis(
    client: LLMClient,
    question: str,
    df: pd.DataFrame,
    max_attempts: int = 3,
    timeout_seconds: float = 5.0,
) -> AnalysisRunResult:
    """Runs the full generate -> execute -> (retry on failure) loop.
    Stops as soon as a generated script succeeds, or after max_attempts
    if every attempt fails -- either way, every attempt (successful or
    not) is recorded in the returned AnalysisRunResult.attempts list, so
    the caller can see exactly what the agent tried, not just the final
    outcome.
    """
    run_result = AnalysisRunResult(question=question)

    prompt = build_codegen_prompt(question, list(df.columns))
    system = "You are a careful, security-conscious data analyst."

    for attempt_num in range(1, max_attempts + 1):
        response = client.complete(prompt, system=system)
        code = extract_code(response.text)

        exec_result = execute_analysis_code(code, df, timeout_seconds=timeout_seconds)
        run_result.attempts.append(
            AnalysisAttempt(code=code, success=exec_result.success, result=exec_result.result, error=exec_result.error)
        )

        if exec_result.success:
            run_result.final_success = True
            run_result.final_result = exec_result.result
            return run_result

        if attempt_num < max_attempts:
            prompt = RETRY_PROMPT_TEMPLATE.format(error=exec_result.error, previous_code=code)

    return run_result
