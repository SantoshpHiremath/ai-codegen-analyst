"""Tests for src/analyst_agent.py -- the full generate -> execute ->
retry-on-failure orchestration loop. This is the "agentic" part: these
tests confirm the loop reacts to the sandbox's REAL execution output,
not to a scripted/assumed outcome."""
from __future__ import annotations

import pandas as pd
import pytest

from src.analyst_agent import run_analysis
from src.llm_client import MockLLMClient


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "claim_id": ["C1", "C2", "C3"],
            "claim_type": ["Kfz-Haftpflicht", "Personenschaden", "Kfz-Haftpflicht"],
            "claim_amount_eur": [1000.0, 5000.0, 2000.0],
            "days_open": [10, 40, 5],
        }
    )


class TestSuccessOnFirstAttempt:
    def test_recognized_question_succeeds_in_one_attempt(self, sample_df):
        client = MockLLMClient()
        result = run_analysis(client, "What is the average claim_amount_eur?", sample_df)
        assert result.final_success is True
        assert result.attempt_count == 1
        assert result.attempts[0].success is True

    def test_final_result_is_numerically_correct(self, sample_df):
        client = MockLLMClient()
        result = run_analysis(client, "What is the average claim_amount_eur?", sample_df)
        assert result.final_result == pytest.approx((1000.0 + 5000.0 + 2000.0) / 3, rel=1e-3)


class TestRetryOnFailure:
    def test_broken_syntax_question_retries_and_recovers(self, sample_df):
        client = MockLLMClient()
        result = run_analysis(client, "What is the broken_syntax_error_test result?", sample_df, max_attempts=3)
        assert result.attempt_count == 2
        assert result.attempts[0].success is False
        assert "SyntaxError" in result.attempts[0].error
        assert result.attempts[1].success is True
        assert result.final_success is True

    def test_retry_produces_a_genuine_correction_not_a_fallback(self, sample_df):
        # The specific bug fixed this session: attempt 1 must be the real
        # average-calculation fix (matching the question's original intent
        # closely enough), not the generic row_count/columns fallback that
        # would succeed "by accident" via keyword non-match.
        client = MockLLMClient()
        result = run_analysis(client, "What is the broken_syntax_error_test result?", sample_df, max_attempts=3)
        assert "mean" in result.attempts[1].code
        assert "row_count" not in result.attempts[1].code

    def test_all_attempts_are_recorded_not_just_the_final_one(self, sample_df):
        client = MockLLMClient()
        result = run_analysis(client, "What is the broken_syntax_error_test result?", sample_df, max_attempts=3)
        assert len(result.attempts) == result.attempt_count
        assert all(hasattr(a, "code") and hasattr(a, "success") for a in result.attempts)


class TestMaxAttemptsExhausted:
    def test_stops_at_max_attempts_if_every_attempt_fails(self, sample_df):
        # Force every attempt to fail by using an unsafe question that maps
        # to no recognized pattern except always failing -- we simulate
        # this directly via a client subclass that always returns unsafe code.
        class AlwaysUnsafeClient(MockLLMClient):
            def complete(self, prompt, system=""):
                from src.llm_client import LLMResponse
                return LLMResponse(text="import os\nresult = os.listdir('.')\n", model="always-unsafe")

        client = AlwaysUnsafeClient()
        result = run_analysis(client, "anything", sample_df, max_attempts=3)
        assert result.final_success is False
        assert result.attempt_count == 3
        assert all(not a.success for a in result.attempts)

    def test_final_result_is_none_when_all_attempts_fail(self, sample_df):
        class AlwaysUnsafeClient(MockLLMClient):
            def complete(self, prompt, system=""):
                from src.llm_client import LLMResponse
                return LLMResponse(text="import os\nresult = 1\n", model="always-unsafe")

        client = AlwaysUnsafeClient()
        result = run_analysis(client, "anything", sample_df, max_attempts=2)
        assert result.final_result is None


class TestQuestionVariety:
    def test_count_by_claim_type_question_works_end_to_end(self, sample_df):
        client = MockLLMClient()
        result = run_analysis(client, "Count claims by claim_type", sample_df)
        assert result.final_success is True
        assert result.final_result["Kfz-Haftpflicht"] == 2

    def test_top_longest_open_question_works_end_to_end(self, sample_df):
        client = MockLLMClient()
        result = run_analysis(client, "What are the top longest days_open claims?", sample_df)
        assert result.final_success is True
        assert isinstance(result.final_result, list)
        assert result.final_result[0]["claim_id"] == "C2"
