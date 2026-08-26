"""
ast_guard.py
------------

Static safety check for LLM-generated Python code, run BEFORE any
execution happens. This is the first of the two defence layers described
in sandbox_executor.py's module docstring.

Rejects, by walking the parsed AST rather than pattern-matching on
source text (which is easy to evade with formatting tricks):
- any `import` or `from ... import ...` statement
- calls to exec, eval, compile, __import__, open, input
- any dunder attribute access (e.g. `().__class__.__bases__`, the
  classic sandbox-escape gadget) or dunder name access
- any `with` statement (blocks context-manager-based escapes via a
  crafted object, and analysis code has no legitimate need for one)

This mirrors the same "walk the AST, allowlist only what's needed"
discipline as calculator_tool.py's safe_calculate() elsewhere in this
portfolio, applied to a much larger permitted grammar (arbitrary pandas
expressions, not just arithmetic) since analysis code legitimately needs
more surface area than a calculator.
"""
from __future__ import annotations

import ast


class GuardViolation(Exception):
    pass


_DISALLOWED_CALL_NAMES = {"exec", "eval", "compile", "__import__", "open", "input", "globals", "locals", "vars"}


class _SafetyVisitor(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> None:
        raise GuardViolation(f"import statements are not allowed (line {node.lineno}: 'import {node.names[0].name}')")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise GuardViolation(f"import statements are not allowed (line {node.lineno}: 'from {node.module} import ...')")

    def visit_With(self, node: ast.With) -> None:
        raise GuardViolation(f"'with' statements are not allowed (line {node.lineno})")

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in _DISALLOWED_CALL_NAMES:
            raise GuardViolation(f"call to '{node.func.id}' is not allowed (line {node.lineno})")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            raise GuardViolation(f"dunder attribute access '.{node.attr}' is not allowed (line {node.lineno})")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__") and node.id.endswith("__"):
            raise GuardViolation(f"dunder name access '{node.id}' is not allowed (line {node.lineno})")
        self.generic_visit(node)


def check_code_is_safe(code: str) -> None:
    """Raises GuardViolation if the code contains a disallowed
    construct. Raises SyntaxError (uncaught, deliberately -- the caller
    is expected to handle it as its own distinct failure mode) if the
    code doesn't parse at all, since that's a different problem (a
    malformed LLM response) from an unsafe-but-valid one.
    """
    tree = ast.parse(code, mode="exec")
    _SafetyVisitor().visit(tree)
