"""Smoke tests for the sandbox executor: runs trivial code in-process via subprocess."""

from __future__ import annotations

from main import _run_code
import pytest


def test_run_code_captures_stdout() -> None:
    result = _run_code("print('hello sandbox')", timeout_seconds=10)
    assert result.success
    assert result.returncode == 0
    assert "hello sandbox" in result.stdout


def test_run_code_reports_nonzero_exit() -> None:
    result = _run_code("import sys; sys.exit(3)", timeout_seconds=10)
    assert not result.success
    assert result.returncode == 3


def test_run_code_times_out_quickly() -> None:
    # 'while True: pass' is CPU-bound; even with RLIMIT_CPU off, the wall-clock
    # timeout must trip and return 124.
    result = _run_code("while True:\n    pass\n", timeout_seconds=2)
    assert not result.success
    assert result.returncode == 124
    assert "timed out" in result.stderr.lower()


@pytest.mark.parametrize(
    "snippet,expected",
    [
        ("print(1 + 1)", "2"),
        ("print('a' * 3)", "aaa"),
    ],
)
def test_run_code_parametrized(snippet: str, expected: str) -> None:
    result = _run_code(snippet, timeout_seconds=10)
    assert result.success
    assert expected in result.stdout
