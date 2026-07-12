"""Long-running per-source incremental update scheduler for downloader_general.

After the one-shot initial ingest, the container stays alive and refreshes each
enabled source on its **own** interval by calling that source's incremental
``update()`` (append-only) — never the destructive full ``run()``. Modelled on
the ``backup`` service: a single SIGTERM/SIGINT-aware loop, per-source next-due
tracking, and a per-source ``try/except`` so one failing source never kills the
loop.
"""

from collections.abc import Callable
from dataclasses import dataclass
import logging
import signal
import threading
import time
from types import FrameType

logger = logging.getLogger(__name__)

_stop = threading.Event()

# Cap on how long a single wait blocks so SIGTERM is handled promptly even when
# the next source isn't due for a week.
_MAX_SLEEP_SECONDS = 3600.0


def _handle_signal(signum: int, _frame: FrameType | None) -> None:
    logger.info("Scheduler received signal %s; shutting down.", signum)
    _stop.set()


@dataclass
class SourceJob:
    """One scheduled source: its name, interval, and a callable that runs one tick.

    ``run_tick`` is expected to (re)establish connections and call the source's
    ``update()``; it must not raise for the loop's sake, but the loop also guards
    it defensively.
    """

    name: str
    interval_seconds: float
    run_tick: Callable[[], None]


def run_scheduler(jobs: list[SourceJob], run_on_start: bool = False) -> None:
    """Run ``jobs`` on their per-source intervals until SIGTERM/SIGINT.

    Args:
        jobs: Enabled sources to schedule (empty → the function returns at once).
        run_on_start: When true every job runs immediately on entry; otherwise the
            first run happens one interval later (the initial ingest just ran).
    """
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    if not jobs:
        logger.info("Scheduler: no sources enabled; nothing to schedule.")
        return

    jobs_by_name = {job.name: job for job in jobs}
    now = time.monotonic()
    next_due = {job.name: (now if run_on_start else now + job.interval_seconds) for job in jobs}
    logger.info(
        "Scheduler started for %d source(s): %s (run_on_start=%s)",
        len(jobs),
        {job.name: f"{job.interval_seconds / 60:.0f}min" for job in jobs},
        run_on_start,
    )

    while not _stop.is_set():
        now = time.monotonic()
        for name, due_at in list(next_due.items()):
            if _stop.is_set():
                break
            if due_at > now:
                continue
            logger.info("Scheduler: running incremental update for %s", name)
            try:
                jobs_by_name[name].run_tick()
            except Exception:
                logger.exception("Scheduler: update failed for %s; retry next interval", name)
            next_due[name] = time.monotonic() + jobs_by_name[name].interval_seconds

        if _stop.is_set():
            break
        now = time.monotonic()
        sleep_for = max(1.0, min(next_due.values()) - now)
        _stop.wait(min(sleep_for, _MAX_SLEEP_SECONDS))

    logger.info("Scheduler stopped.")
