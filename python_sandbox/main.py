import os
import subprocess
import sys
import tempfile

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

CONFIG_PATH = "config.yaml"
ENV_FILE_PATH = ".env"

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)
load_dotenv(ENV_FILE_PATH)

app = FastAPI(
    title="Python Sandbox API",
    description="API to execute Python code in a sandboxed environment",
)


class CodePayload(BaseModel):
    code: str = Field(..., description="Python code to execute.")
    timeout_seconds: int = Field(
        60, gt=0, le=600, description="Execution timeout in seconds (1-600)."
    )


class ExecutionResult(BaseModel):
    success: bool
    stdout: str
    stderr: str
    returncode: int


def _run_code(code: str, timeout_seconds: int) -> ExecutionResult:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name

    try:
        result = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return ExecutionResult(
            success=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr="Execution timed out.",
            returncode=124,
        )
    except (FileNotFoundError, PermissionError, OSError) as exc:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"Execution environment error: {exc}",
            returncode=1,
        )
    finally:
        try:
            os.remove(temp_file_path)
        except OSError:
            pass


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Welcome to the Python Sandbox API!"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/execute", response_model=ExecutionResult)
async def execute_code(payload: CodePayload) -> ExecutionResult:
    return await run_in_threadpool(_run_code, payload.code, payload.timeout_seconds)
