"""
sandbox_executor.py
--------------------

Runs LLM-generated Python code against a real pandas DataFrame safely.
This is the module that makes "AI writes its own analysis code" a
responsible pattern rather than a security hole: LLM output is never
handed to a bare `exec()` with full builtins.

Two layers of defence, both real and independently testable:

1. A static AST check (ast_guard.py's job, imported here) that rejects
   code containing imports, exec/eval, dunder attribute access, or other
   disallowed constructs -- BEFORE any code runs at all.
2. Execution in a restricted namespace (a minimal `__builtins__` with
   only safe, read-only builtins -- no `open`, `__import__`, `exec`,
   `eval`, `input`, etc.) with a wall-clock timeout, run in a separate
   process so a runaway or hanging script can't block the caller
   indefinitely and can't touch the parent process's memory.

Honest scope note: this is a defence-in-depth sandbox appropriate for a
demo/portfolio project showing the *pattern* correctly, not a claim of
production-grade isolation (a real production system would run generated
code in an actual OS-level sandbox/container, not just a restricted
Python namespace in a subprocess). That distinction is stated directly
here rather than overselling what this provides.
"""
from __future__ import annotations

import builtins as _builtins_module
import multiprocessing
from dataclasses import dataclass

import pandas as pd

from src.ast_guard import GuardViolation, check_code_is_safe

# An explicit allowlist of safe, read-only builtins -- deliberately does
# NOT include open, __import__, exec, eval, input, compile, or any
# filesystem/network/process-control function. Built from the real
# `builtins` module (not the ambient __builtins__, whose type varies by
# execution context -- module at top level, dict inside some exec()
# calls) so this is reliable regardless of how sandbox_executor.py itself
# is invoked.
_SAFE_NAMES = [
    "len", "range", "sum", "min", "max", "sorted", "list", "dict", "set",
    "tuple", "str", "int", "float", "bool", "round", "abs", "enumerate",
    "zip", "map", "filter", "isinstance", "print",
]
SAFE_BUILTINS = {name: getattr(_builtins_module, name) for name in _SAFE_NAMES}


@dataclass
class ExecutionResult:
    success: bool
    result: object
    error: str | None


def _run_in_subprocess(code: str, df_pickle_path: str, queue) -> None:
    """Runs in a separate process (spawned by multiprocessing) so a
    hanging or resource-heavy generated script can be killed via timeout
    without affecting the caller. Loads the DataFrame from a pickle file
    rather than passing it through the multiprocessing queue directly, to
    avoid large-object pickling overhead on every call.
    """
    try:
        df = pd.read_pickle(df_pickle_path)
        namespace = {"df": df, "pd": pd, "__builtins__": SAFE_BUILTINS}
        exec(code, namespace)  # noqa: S102 - deliberate, guarded exec; see module docstring
        queue.put(("ok", namespace.get("result")))
    except Exception as exc:  # noqa: BLE001 - deliberately broad, surfaced to caller
        queue.put(("error", f"{exc.__class__.__name__}: {exc}"))


def execute_analysis_code(code: str, df: pd.DataFrame, timeout_seconds: float = 5.0) -> ExecutionResult:
    """Validates then executes LLM-generated analysis code against a real
    DataFrame. Returns a structured result rather than raising, so a
    calling agent can inspect success/failure and decide whether to
    retry with a corrected prompt.
    """
    try:
        check_code_is_safe(code)
    except GuardViolation as exc:
        return ExecutionResult(success=False, result=None, error=f"Rejected by safety guard: {exc}")
    except SyntaxError as exc:
        # A malformed LLM response (e.g. an unclosed paren) is a
        # different, distinct failure mode from an unsafe-but-valid one
        # -- caught here explicitly so it surfaces as a normal
        # ExecutionResult failure the calling agent can retry against,
        # not an uncaught exception that crashes the caller.
        return ExecutionResult(success=False, result=None, error=f"SyntaxError: generated code does not parse ({exc})")

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        df_pickle_path = tmp.name
    try:
        df.to_pickle(df_pickle_path)

        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(target=_run_in_subprocess, args=(code, df_pickle_path, queue))
        proc.start()
        proc.join(timeout=timeout_seconds)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            return ExecutionResult(success=False, result=None, error=f"Execution timed out after {timeout_seconds}s")

        if queue.empty():
            return ExecutionResult(success=False, result=None, error="Process exited without producing a result (likely crashed)")

        status, payload = queue.get()
        if status == "ok":
            return ExecutionResult(success=True, result=payload, error=None)
        return ExecutionResult(success=False, result=None, error=payload)
    finally:
        if os.path.exists(df_pickle_path):
            os.remove(df_pickle_path)
