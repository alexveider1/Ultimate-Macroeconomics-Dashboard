"""Tests for the per-source incremental update scheduler loop.

The loop blocks on a shared ``threading.Event``; each test makes its job set that
event so the loop exits deterministically instead of sleeping. ``run_on_start`` is
used so the due job fires on the first iteration (no wait), keeping tests fast.
"""

from collections.abc import Iterator

import pytest

from src import scheduler
from src.scheduler import SourceJob, run_scheduler


@pytest.fixture(autouse=True)
def _reset_stop() -> Iterator[None]:
    """Ensure the module-level stop event is clear before and after each test."""
    scheduler._stop.clear()
    yield
    scheduler._stop.clear()


def test_run_on_start_runs_due_job_once() -> None:
    calls: list[str] = []

    def tick() -> None:
        calls.append("x")
        scheduler._stop.set()  # break the loop after the first run

    run_scheduler([SourceJob("x", interval_seconds=3600, run_tick=tick)], run_on_start=True)
    assert calls == ["x"]


def test_no_jobs_returns_immediately() -> None:
    # Must not hang; nothing to schedule.
    run_scheduler([], run_on_start=True)


def test_failing_tick_does_not_crash_loop() -> None:
    calls: list[str] = []

    def tick() -> None:
        calls.append("x")
        scheduler._stop.set()
        raise RuntimeError("boom")

    # A raising tick is swallowed and logged, not propagated.
    run_scheduler([SourceJob("x", interval_seconds=3600, run_tick=tick)], run_on_start=True)
    assert calls == ["x"]


def test_stop_set_before_start_runs_no_ticks() -> None:
    calls: list[str] = []
    scheduler._stop.set()
    run_scheduler(
        [SourceJob("x", interval_seconds=3600, run_tick=lambda: calls.append("x"))],
        run_on_start=True,
    )
    assert calls == []


def test_stop_mid_iteration_skips_remaining_due_jobs() -> None:
    calls: list[str] = []

    def first() -> None:
        calls.append("first")
        scheduler._stop.set()  # stop before the second due job is reached

    def second() -> None:  # pragma: no cover - guarded out by the stop check
        calls.append("second")

    jobs = [
        SourceJob("first", interval_seconds=3600, run_tick=first),
        SourceJob("second", interval_seconds=3600, run_tick=second),
    ]
    # Both are due at start (run_on_start), but the loop re-checks the stop event
    # between due jobs, so only the first runs.
    run_scheduler(jobs, run_on_start=True)
    assert calls == ["first"]
