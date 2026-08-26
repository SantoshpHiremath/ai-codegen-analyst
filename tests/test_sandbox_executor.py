"""Tests for src/sandbox_executor.py -- running LLM-generated code
against a real DataFrame, in a subprocess, under a timeout, with a
restricted builtins namespace."""
from __future__ import annotations

import pandas as pd
import pytest

from src.sandbox_executor import execute_analysis_code


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "claim_id": ["C1", "C2", "C3", "C4"],
            "claim_type": ["Kfz-Haftpflicht", "Personenschaden", "Kfz-Haftpflicht", "Glasschaden"],
            "claim_amount_eur": [1000.0, 5000.0, 2000.0, 300.0],
            "days_open": [10, 40, 5, 2],
        }
    )


class TestSuccessfulExecution:
    def test_simple_mean_computation_succeeds(self, sample_df):
        code = "result = round(df['claim_amount_eur'].mean(), 2)\n"
        exec_result = execute_analysis_code(code, sample_df)
        assert exec_result.success is True
        assert exec_result.error is None
        assert exec_result.result == pytest.approx(2075.0)

    def test_groupby_computation_succeeds(self, sample_df):
        code = "result = df.groupby('claim_type').size().to_dict()\n"
        exec_result = execute_analysis_code(code, sample_df)
        assert exec_result.success is True
        assert exec_result.result["Kfz-Haftpflicht"] == 2

    def test_result_can_be_a_list_of_dicts(self, sample_df):
        code = "result = df.nlargest(1, 'days_open')[['claim_id']].to_dict('records')\n"
        exec_result = execute_analysis_code(code, sample_df)
        assert exec_result.success is True
        assert exec_result.result == [{"claim_id": "C2"}]

    def test_missing_result_variable_still_returns_structured_response(self, sample_df):
        # If the generated code never assigns `result`, namespace.get("result")
        # is None -- this should still be a clean success with result=None,
        # not a crash.
        code = "x = 1 + 1\n"
        exec_result = execute_analysis_code(code, sample_df)
        assert exec_result.success is True
        assert exec_result.result is None


class TestSafetyRejection:
    def test_unsafe_code_is_rejected_before_execution(self, sample_df):
        code = "import os\nresult = os.listdir('.')\n"
        exec_result = execute_analysis_code(code, sample_df)
        assert exec_result.success is False
        assert "Rejected by safety guard" in exec_result.error

    def test_sandbox_escape_gadget_is_rejected(self, sample_df):
        code = "result = ().__class__.__bases__[0].__subclasses__()\n"
        exec_result = execute_analysis_code(code, sample_df)
        assert exec_result.success is False
        assert "Rejected by safety guard" in exec_result.error

    def test_restricted_builtins_prevent_open_at_runtime(self, sample_df):
        # open() is both guard-rejected statically; this test asserts the
        # guard catches it (defence layer 1) rather than letting it reach
        # execution.
        code = "f = open('/etc/passwd')\nresult = f.read()\n"
        exec_result = execute_analysis_code(code, sample_df)
        assert exec_result.success is False


class TestSyntaxErrorHandling:
    def test_malformed_code_returns_structured_failure_not_a_crash(self, sample_df):
        code = "result = df['claim_amount_eur'].mean(\n"  # unclosed paren
        exec_result = execute_analysis_code(code, sample_df)
        assert exec_result.success is False
        assert "SyntaxError" in exec_result.error


class TestTimeout:
    def test_infinite_loop_times_out_rather_than_hanging(self, sample_df):
        code = "i = 0\nwhile True:\n    i += 1\nresult = i\n"
        exec_result = execute_analysis_code(code, sample_df, timeout_seconds=1.0)
        assert exec_result.success is False
        assert "timed out" in exec_result.error


class TestRuntimeErrors:
    def test_runtime_error_is_captured_not_raised(self, sample_df):
        code = "result = df['nonexistent_column'].mean()\n"
        exec_result = execute_analysis_code(code, sample_df)
        assert exec_result.success is False
        assert exec_result.error is not None
