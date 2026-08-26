"""
code_generator.py
------------------

Builds the prompt that asks an LLM to write a Python analysis script for
a natural-language data question, and extracts runnable code from the
LLM's response -- real LLMs typically wrap code in markdown fences
(```python ... ```), so this has to handle that, not just assume the
raw response text is bare executable Python.
"""
from __future__ import annotations

import re

from src.llm_client import LLMClient

CODEGEN_PROMPT_TEMPLATE = """You are a data analyst writing a short Python \
script to answer a question about a pandas DataFrame called `df`.

The DataFrame has these columns: {columns}

Column notes:
{column_notes}

Question: {question}

Write ONLY the analysis code (no imports, no function definitions, no \
print statements) that computes the answer and assigns it to a variable \
called `result`. Assume `df` and `pandas as pd` already exist in scope. \
Respond with a single Python code block, nothing else.
"""

COLUMN_NOTES = """- claim_amount_eur may contain nulls and occasional negative \
values (data-entry sign errors) -- filter or clean before aggregating.
- region may have inconsistent casing (e.g. "bayern" vs "Bayern") from an \
upstream feed.
- customer_age of -1 is a sentinel value for "unknown," not a real age.
- Some claim_id rows may be exact duplicates (upstream retries).
"""


def build_codegen_prompt(question: str, columns: list) -> str:
    return CODEGEN_PROMPT_TEMPLATE.format(
        columns=", ".join(columns),
        column_notes=COLUMN_NOTES,
        question=question,
    )


def extract_code(llm_response_text: str) -> str:
    """Pulls Python code out of an LLM response. Handles three real
    response shapes: a fenced ```python block, a fenced ``` block with no
    language tag, or (as a fallback) treating the whole response as bare
    code -- since a smaller/less compliant model sometimes skips the
    fence entirely.
    """
    fenced = re.search(r"```(?:python)?\s*\n(.*?)```", llm_response_text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return llm_response_text.strip()


def generate_analysis_code(client: LLMClient, question: str, columns: list) -> str:
    prompt = build_codegen_prompt(question, columns)
    response = client.complete(prompt, system="You are a careful, security-conscious data analyst.")
    return extract_code(response.text)
