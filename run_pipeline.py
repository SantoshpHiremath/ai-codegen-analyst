"""
run_pipeline.py
----------------

End-to-end demo: generate a synthetic claims dataset, ask the agent a
handful of natural-language questions, and show exactly what it did --
the code it generated, whether it ran successfully, and (for the
deliberately-broken question) the full retry-and-recovery sequence.

Run with:
    python3 run_pipeline.py
"""
from __future__ import annotations

import pandas as pd

from src.analyst_agent import run_analysis
from src.generate_data import generate_claims
from src.llm_client import MockLLMClient

QUESTIONS = [
    "What is the average claim_amount_eur?",
    "Count claims by claim_type",
    "What are the top longest days_open claims?",
    "What is the broken_syntax_error_test result?",
    "Tell me something totally unrelated to any pattern",
]


def _print_attempt(i: int, attempt) -> None:
    status = "OK" if attempt.success else "FAILED"
    print(f"  attempt {i}: {status}")
    print("  --- generated code ---")
    for line in attempt.code.splitlines():
        print(f"    {line}")
    if attempt.success:
        print(f"  --- result ---\n    {attempt.result}")
    else:
        print(f"  --- error ---\n    {attempt.error}")
    print()


def main() -> None:
    print("=" * 70)
    print("ai-codegen-analyst -- end-to-end demo")
    print("=" * 70)

    print("\nGenerating synthetic motor/personal-injury claims dataset...")
    rows = generate_claims(n=2000, seed=42)
    df = pd.DataFrame(rows)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")
    print("  (synthetic data -- NOT real Allianz or any other insurer's data)")

    client = MockLLMClient()

    for question in QUESTIONS:
        print("\n" + "-" * 70)
        print(f"Question: {question}")
        print("-" * 70)

        result = run_analysis(client, question, df, max_attempts=3)

        for i, attempt in enumerate(result.attempts):
            _print_attempt(i, attempt)

        if result.final_success:
            print(f"=> Final answer (after {result.attempt_count} attempt(s)): {result.final_result}")
        else:
            print(f"=> FAILED after {result.attempt_count} attempt(s), no answer produced.")

    print("\n" + "=" * 70)
    print("Notes:")
    print("- MockLLMClient is the only LLM backend actually exercised here")
    print("  (no live OPENAI_API_KEY / ANTHROPIC_API_KEY in this environment).")
    print("- The 'broken_syntax_error_test' question deliberately exercises")
    print("  the full retry loop: attempt 0 fails with a real SyntaxError from")
    print("  the sandbox, attempt 1 is a genuine corrected version, not a")
    print("  fallback that happens to succeed by a different route.")
    print("- Every generated script ran through the two-layer sandbox: a")
    print("  static AST safety check, then execution in a restricted-builtins")
    print("  subprocess under a wall-clock timeout. See README.md.")
    print("=" * 70)


if __name__ == "__main__":
    main()
