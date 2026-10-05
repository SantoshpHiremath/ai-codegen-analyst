"""
LLM client abstraction: a real client interface plus a tested mock.

Note: no OpenAI or Anthropic API key is configured here, so `RealOpenAIClient` and `RealAnthropicClient` below have
never actually been executed against a live API in this environment.
Their call shapes match the real SDKs as closely as possible without
being able to run them -- that is a different, weaker claim than "tested
against a live API," and it's stated directly rather than left implicit.

What IS real and fully tested: the `LLMClient` interface, `MockLLMClient`
(deterministic, used by every test and by run_pipeline.py), and the fact
that a real client is swappable in through the exact same interface with
zero changes to any calling code -- the evaluation/generation pipeline
depends only on `LLMClient`, never on a specific vendor SDK.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    model: str


class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, system: str = "") -> LLMResponse:
        ...


class MockLLMClient(LLMClient):
    """Deterministic mock backend: no network calls. Generates real,
    runnable Python analysis code for a small set of recognized question
    patterns (matched by keyword, similar to MockLLMClient's
    keyword-triggered response templates in llm-eval-pipeline), and a
    deliberately-broken script for one pattern so the sandbox executor
    and the agent's retry logic have a genuine failure to detect and
    recover from -- not just uniform success dressed up as a "codegen
    agent."

    This is the only LLMClient actually exercised by this project's test
    suite and run_pipeline.py, since no live API key exists here.
    """

    def complete(self, prompt: str, system: str = "") -> LLMResponse:
        lower = prompt.lower()
        is_retry = "your previous code failed" in lower

        if is_retry and "was never closed" in lower:
            # A genuine "fix" of the specific syntax error injected below
            # -- simulates a real model correcting itself after seeing
            # its own error message, closing the retry loop honestly
            # rather than accidentally falling through to a different
            # response path on retry.
            code = self._avg_claim_amount_code()
        elif "average claim_amount_eur" in lower or ("average" in lower and "claim_amount" in lower):
            code = self._avg_claim_amount_code()
        elif "count" in lower and "claim_type" in lower:
            code = self._count_by_claim_type_code()
        elif "days_open" in lower and ("longest" in lower or "top" in lower):
            code = self._top_longest_open_code()
        elif "broken" in lower or "syntax_error_test" in lower:
            # Deliberately invalid Python -- exercises the sandbox's real
            # error handling and the agent's retry-on-failure path.
            code = "result = df['claim_amount_eur'].mean(\n"  # missing closing paren
        else:
            code = self._generic_describe_code()

        return LLMResponse(text=code, model="mock-codegen-llm-v1")

    @staticmethod
    def _avg_claim_amount_code() -> str:
        return (
            "cleaned = df[df['claim_amount_eur'].notna() & (df['claim_amount_eur'] >= 0)]\n"
            "result = round(cleaned['claim_amount_eur'].mean(), 2)\n"
        )

    @staticmethod
    def _count_by_claim_type_code() -> str:
        return (
            "result = df.groupby('claim_type').size().sort_values(ascending=False).to_dict()\n"
        )

    @staticmethod
    def _top_longest_open_code() -> str:
        return (
            "top5 = df.nlargest(5, 'days_open')[['claim_id', 'claim_type', 'days_open']]\n"
            "result = top5.to_dict('records')\n"
        )

    @staticmethod
    def _generic_describe_code() -> str:
        return (
            "result = {\n"
            "    'row_count': len(df),\n"
            "    'columns': list(df.columns),\n"
            "}\n"
        )


class RealOpenAIClient(LLMClient):
    """Written to match the real openai>=1.0 SDK's call shape
    (`client.chat.completions.create(model=..., messages=[...])`).
    NEVER EXECUTED in this sandbox -- there is no OPENAI_API_KEY
    available here, and instantiating this class raises immediately
    rather than silently falling back to mock behavior.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        import os
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "RealOpenAIClient requires OPENAI_API_KEY. This class has "
                "never been executed in this project's development "
                "environment (no key was available) — its call shape "
                "matches the real openai>=1.0 SDK but has NOT been "
                "verified against a live API. See README."
            )
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "The `openai` package is not installed in this environment."
            ) from e
        self._client = openai.OpenAI(api_key=key)
        self._model = model

    def complete(self, prompt: str, system: str = "") -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(model=self._model, messages=messages)
        return LLMResponse(text=resp.choices[0].message.content, model=resp.model)


class RealAnthropicClient(LLMClient):
    """Written to match the real anthropic SDK's call shape
    (`client.messages.create(model=..., messages=[...])`). NEVER
    EXECUTED in this sandbox for the same reason as RealOpenAIClient --
    no ANTHROPIC_API_KEY available here.
    """

    def __init__(self, api_key: str | None = None, model: str = "claude-3-5-sonnet-latest"):
        import os
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "RealAnthropicClient requires ANTHROPIC_API_KEY. This "
                "class has never been executed in this project's "
                "development environment (no key was available) — its "
                "call shape matches the real anthropic SDK but has NOT "
                "been verified against a live API. See README."
            )
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "The `anthropic` package is not installed in this environment."
            ) from e
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model

    def complete(self, prompt: str, system: str = "") -> LLMResponse:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system or None,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in resp.content if hasattr(block, "text"))
        return LLMResponse(text=text, model=resp.model)
