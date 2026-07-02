"""FastAPI service that runs LLM-generated Python in a hardened subprocess.

Only used by the agent's ``plotly_agent`` worker today: the LLM emits a small
Plotly/Polars snippet, the agent POSTs it to ``/execute``, and the response
carries stdout/stderr/returncode. Each run gets a fresh ``subprocess`` with
RLIMIT_AS / RLIMIT_CPU caps so a runaway snippet can't take down the host.
"""

import logging
import resource  # POSIX only — container is Linux, so always available here
import subprocess
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from config import load_config

logger = logging.getLogger(__name__)

SANDBOX_MEMORY_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
SANDBOX_CPU_TIME_BUFFER_SECONDS = 5

CONFIG_PATH = Path("config.yaml")

CONFIG = load_config(CONFIG_PATH)

app = FastAPI(
    title="Python Sandbox API",
    description="API to execute Python code in a sandboxed environment",
)


class CodePayload(BaseModel):
    """Request body for the ``/execute`` endpoint.

    Args:
        code: Python source to run as a standalone script.
        timeout_seconds: Wall-clock budget in seconds (1-600 inclusive).
    """

    code: str = Field(..., description="Python code to execute.")
    timeout_seconds: int = Field(
        60, gt=0, le=600, description="Execution timeout in seconds (1-600)."
    )


class ExecutionResult(BaseModel):
    """Response body returned by ``/execute``.

    Args:
        success: ``True`` iff the subprocess exited with returncode 0.
        stdout: Captured standard output.
        stderr: Captured standard error.
        returncode: Subprocess exit code (``124`` is reserved for timeouts).
    """

    success: bool
    stdout: str
    stderr: str
    returncode: int


def _build_preexec(timeout_seconds: int):
    """Build a ``preexec_fn`` that caps memory + CPU before the child execs.

    LLM-generated code can attempt to allocate gigabytes or spin in a tight
    loop. The wall-clock timeout alone leaves the container under load for
    the full timeout; ``RLIMIT_AS`` + ``RLIMIT_CPU`` let the kernel kill
    misbehaving code immediately.

    Args:
        timeout_seconds: The wall-clock timeout the parent will enforce; the
            CPU rlimit is set slightly higher so the parent's timeout fires
            first and yields a clean ``returncode=124``.

    Returns:
        A callable suitable for ``subprocess.run(preexec_fn=...)`` on POSIX,
        or ``None`` when ``resource`` is unavailable (non-POSIX hosts).
    """
    if resource is None:
        return None

    cpu_seconds = timeout_seconds + SANDBOX_CPU_TIME_BUFFER_SECONDS

    def _apply_limits() -> None:
        """Apply RLIMIT_AS and RLIMIT_CPU before ``exec`` in the child."""
        resource.setrlimit(
            resource.RLIMIT_AS, (SANDBOX_MEMORY_LIMIT_BYTES, SANDBOX_MEMORY_LIMIT_BYTES)
        )
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))

    return _apply_limits


def _run_code(code: str, timeout_seconds: int) -> ExecutionResult:
    """Write ``code`` to a temp file and run it as a subprocess with rlimits.

    Args:
        code: Python source to execute.
        timeout_seconds: Wall-clock budget; on overrun the child is killed
            and the result reports ``returncode=124``.

    Returns:
        ExecutionResult capturing stdout, stderr, returncode, and a success
        flag. Environment errors (missing python, EACCES on the temp file)
        are returned as ``returncode=1`` with the message in ``stderr``.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
        temp_file.write(code)
        temp_file_path = Path(temp_file.name)

    logger.info(
        "sandbox: starting subprocess (timeout=%ss, code_bytes=%d)",
        timeout_seconds,
        len(code),
    )
    try:
        result = subprocess.run(
            [sys.executable, str(temp_file_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            preexec_fn=_build_preexec(timeout_seconds),
        )
        logger.info(
            "sandbox: subprocess finished (returncode=%s, stdout_bytes=%d, stderr_bytes=%d)",
            result.returncode,
            len(result.stdout or ""),
            len(result.stderr or ""),
        )
        return ExecutionResult(
            success=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )
    except subprocess.TimeoutExpired:
        logger.warning("sandbox: subprocess timed out after %ss", timeout_seconds)
        return ExecutionResult(
            success=False,
            stdout="",
            stderr="Execution timed out.",
            returncode=124,
        )
    except (FileNotFoundError, PermissionError, OSError) as exc:
        logger.exception("sandbox: environment error")
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"Execution environment error: {exc}",
            returncode=1,
        )
    finally:
        try:
            temp_file_path.unlink()
        except OSError as exc:
            logger.warning("sandbox: could not remove temp file %s: %s", temp_file_path, exc)


@app.get("/")
def root() -> dict[str, str]:
    """Return a static welcome banner — used as a liveness signal."""
    return {"message": "Welcome to the Python Sandbox API!"}


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return ``{"status": "ok"}`` for the Compose healthcheck."""
    return {"status": "ok"}


@app.post("/execute", response_model=ExecutionResult)
async def execute_code(payload: CodePayload) -> ExecutionResult:
    """Run ``payload.code`` in a sandboxed subprocess and return its output.

    Args:
        payload: User-submitted code and timeout.

    Returns:
        ExecutionResult: success flag, stdout, stderr, and exit code.
    """
    return await run_in_threadpool(_run_code, payload.code, payload.timeout_seconds)
