"""Tests for src/code_generator.py -- prompt building and code extraction
from LLM responses."""
from __future__ import annotations

from src.code_generator import build_codegen_prompt, extract_code, generate_analysis_code
from src.llm_client import MockLLMClient


class TestBuildCodegenPrompt:
    def test_prompt_includes_the_question(self):
        prompt = build_codegen_prompt("What is the average claim amount?", ["a", "b"])
        assert "What is the average claim amount?" in prompt

    def test_prompt_includes_column_names(self):
        prompt = build_codegen_prompt("some question", ["claim_id", "claim_amount_eur"])
        assert "claim_id" in prompt
        assert "claim_amount_eur" in prompt

    def test_prompt_includes_column_notes_about_data_quality(self):
        prompt = build_codegen_prompt("some question", ["claim_amount_eur"])
        assert "nulls" in prompt.lower() or "negative" in prompt.lower()


class TestExtractCode:
    def test_extracts_code_from_python_fenced_block(self):
        response = "Here you go:\n```python\nresult = 1 + 1\n```\n"
        assert extract_code(response) == "result = 1 + 1"

    def test_extracts_code_from_untagged_fenced_block(self):
        response = "```\nresult = 2 + 2\n```"
        assert extract_code(response) == "result = 2 + 2"

    def test_falls_back_to_bare_response_when_no_fence_present(self):
        response = "result = 3 + 3"
        assert extract_code(response) == "result = 3 + 3"

    def test_strips_surrounding_whitespace(self):
        response = "```python\n\n  result = 5\n\n```"
        assert extract_code(response).strip() == "result = 5"

    def test_extracts_first_fenced_block_when_multiple_present(self):
        response = "```python\nresult = 1\n```\nsome text\n```python\nresult = 2\n```"
        assert extract_code(response) == "result = 1"


class TestGenerateAnalysisCode:
    def test_uses_mock_client_and_returns_runnable_code(self):
        client = MockLLMClient()
        code = generate_analysis_code(client, "What is the average claim_amount_eur?", ["claim_amount_eur"])
        assert "mean" in code

    def test_different_questions_route_to_different_code(self):
        client = MockLLMClient()
        avg_code = generate_analysis_code(client, "What is the average claim_amount_eur?", ["claim_amount_eur"])
        count_code = generate_analysis_code(client, "Count claims by claim_type", ["claim_type"])
        assert avg_code != count_code
