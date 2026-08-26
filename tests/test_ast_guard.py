"""Tests for src/ast_guard.py -- the static safety checker for
LLM-generated code, run before any execution happens."""
from __future__ import annotations

import pytest

from src.ast_guard import GuardViolation, check_code_is_safe


class TestAllowedCode:
    def test_simple_assignment_is_allowed(self):
        check_code_is_safe("result = 1 + 1")

    def test_pandas_style_filtering_is_allowed(self):
        code = (
            "cleaned = df[df['claim_amount_eur'].notna()]\n"
            "result = round(cleaned['claim_amount_eur'].mean(), 2)\n"
        )
        check_code_is_safe(code)

    def test_groupby_aggregation_is_allowed(self):
        code = "result = df.groupby('claim_type').size().to_dict()\n"
        check_code_is_safe(code)

    def test_list_and_dict_comprehensions_are_allowed(self):
        code = "result = {k: v for k, v in enumerate(range(10))}\n"
        check_code_is_safe(code)

    def test_regular_attribute_access_is_allowed(self):
        code = "result = df.columns.tolist()\n"
        check_code_is_safe(code)

    def test_regular_function_calls_are_allowed(self):
        code = "result = len(df)\n"
        check_code_is_safe(code)


class TestDisallowedImports:
    def test_import_statement_is_rejected(self):
        with pytest.raises(GuardViolation, match="import statements are not allowed"):
            check_code_is_safe("import os\nresult = 1\n")

    def test_from_import_statement_is_rejected(self):
        with pytest.raises(GuardViolation, match="import statements are not allowed"):
            check_code_is_safe("from os import system\nresult = 1\n")


class TestDisallowedCalls:
    @pytest.mark.parametrize(
        "func_name",
        ["exec", "eval", "compile", "__import__", "open", "input", "globals", "locals", "vars"],
    )
    def test_disallowed_builtin_calls_are_rejected(self, func_name):
        with pytest.raises(GuardViolation, match=f"call to '{func_name}' is not allowed"):
            check_code_is_safe(f"result = {func_name}('x')\n")

    def test_ordinary_calls_with_similar_names_are_not_falsely_rejected(self):
        # Sanity check the allowlist logic isn't overly broad (substring match, etc.)
        check_code_is_safe("result = sorted([3, 1, 2])\n")


class TestDunderAccess:
    def test_dunder_attribute_access_is_rejected(self):
        with pytest.raises(GuardViolation, match="dunder attribute access"):
            check_code_is_safe("result = ().__class__\n")

    def test_classic_sandbox_escape_gadget_is_rejected(self):
        with pytest.raises(GuardViolation, match="dunder attribute access"):
            check_code_is_safe("result = ().__class__.__bases__[0].__subclasses__()\n")

    def test_dunder_name_access_is_rejected(self):
        with pytest.raises(GuardViolation, match="dunder name access"):
            check_code_is_safe("result = __builtins__\n")


class TestWithStatement:
    def test_with_statement_is_rejected(self):
        with pytest.raises(GuardViolation, match="'with' statements are not allowed"):
            check_code_is_safe("with open('x') as f:\n    result = f.read()\n")


class TestSyntaxError:
    def test_malformed_code_raises_syntax_error_uncaught(self):
        # check_code_is_safe deliberately does NOT catch SyntaxError itself --
        # that's the caller's (sandbox_executor's) job, as a distinct failure
        # mode from an unsafe-but-valid script.
        with pytest.raises(SyntaxError):
            check_code_is_safe("result = df['x'].mean(\n")
