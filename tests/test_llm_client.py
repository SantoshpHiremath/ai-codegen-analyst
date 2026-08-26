"""Tests for src/llm_client.py -- the LLMClient interface, MockLLMClient's
keyword-triggered codegen responses, and the RealOpenAIClient/
RealAnthropicClient fail-fast-without-a-key behaviour."""
from __future__ import annotations

import pytest

from src.llm_client import (
    LLMResponse,
    MockLLMClient,
    RealAnthropicClient,
    RealOpenAIClient,
)


class TestMockLLMClientRouting:
    def test_average_claim_amount_question_returns_mean_code(self):
        client = MockLLMClient()
        resp = client.complete("What is the average claim_amount_eur?")
        assert "mean" in resp.text

    def test_count_by_claim_type_question_returns_groupby_code(self):
        client = MockLLMClient()
        resp = client.complete("Count claims by claim_type")
        assert "groupby" in resp.text

    def test_top_longest_open_question_returns_nlargest_code(self):
        client = MockLLMClient()
        resp = client.complete("What are the top 5 longest days_open claims?")
        assert "nlargest" in resp.text

    def test_broken_keyword_returns_deliberately_invalid_code(self):
        client = MockLLMClient()
        resp = client.complete("What is the broken_syntax_error_test result?")
        assert resp.text.count("(") != resp.text.count(")")

    def test_unrecognized_question_returns_generic_describe_code(self):
        client = MockLLMClient()
        resp = client.complete("Tell me something totally unrelated to any pattern")
        assert "row_count" in resp.text

    def test_response_includes_model_name(self):
        client = MockLLMClient()
        resp = client.complete("What is the average claim_amount_eur?")
        assert resp.model == "mock-codegen-llm-v1"

    def test_returns_llm_response_instance(self):
        client = MockLLMClient()
        resp = client.complete("anything")
        assert isinstance(resp, LLMResponse)


class TestMockLLMClientRetryBehavior:
    def test_retry_after_broken_syntax_error_returns_the_genuine_fix(self):
        """The core behavior fixed this session: a retry prompt that
        references the specific injected syntax error ('was never
        closed') must deliberately route to the real average-calculation
        fix, not fall through by accident to the generic describe code
        path via keyword non-match.
        """
        client = MockLLMClient()
        retry_prompt = (
            "Your previous code failed with this error:\n\n"
            "SyntaxError: generated code does not parse ('(' was never closed (<unknown>, line 1))\n\n"
            "Here was your previous code:\n```python\nresult = df['claim_amount_eur'].mean(\n```\n\n"
            "Please write a corrected version. Respond with a single Python code block, nothing else.\n"
        )
        resp = client.complete(retry_prompt)
        assert "mean" in resp.text
        assert "row_count" not in resp.text

    def test_retry_fix_is_syntactically_valid_python(self):
        import ast

        client = MockLLMClient()
        retry_prompt = (
            "Your previous code failed with this error:\n\n"
            "SyntaxError: generated code does not parse ('(' was never closed (<unknown>, line 1))\n\n"
            "Here was your previous code:\n```python\nresult = df['claim_amount_eur'].mean(\n```\n"
        )
        resp = client.complete(retry_prompt)
        ast.parse(resp.text)  # should not raise

    def test_retry_prompt_for_a_different_error_does_not_trigger_the_syntax_fix(self):
        # A retry prompt whose error text doesn't mention "was never closed"
        # should NOT be special-cased -- it should fall through to normal
        # keyword routing (or the generic fallback), since is_retry alone
        # isn't a signal of *which* correction to make.
        client = MockLLMClient()
        retry_prompt = (
            "Your previous code failed with this error:\n\n"
            "KeyError: 'nonexistent_column'\n\n"
            "Here was your previous code:\n```python\nresult = df['nonexistent_column'].mean()\n```\n"
        )
        resp = client.complete(retry_prompt)
        assert "row_count" in resp.text


class TestRealOpenAIClientFailsFastWithoutKey:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            RealOpenAIClient(api_key=None)

    def test_error_message_discloses_never_executed(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="never been executed"):
            RealOpenAIClient(api_key=None)


class TestRealAnthropicClientFailsFastWithoutKey:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            RealAnthropicClient(api_key=None)

    def test_error_message_discloses_never_executed(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="never been executed"):
            RealAnthropicClient(api_key=None)
