from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import tempfile
import os
import sys

app = FastAPI(
    title="Python Sandbox API",
    description="API to execute Python code in a sandboxed environment",
)


class CodePayload(BaseModel):
    code: str
    timeout_seconds: int = 60


@app.get("/")
def root():
    return {"message": "Welcome to the Python Sandbox API!"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/execute")
async def execute_code(payload: CodePayload):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
        temp_file.write(payload.code)
        temp_file_path = temp_file.name

    try:
        result = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=payload.timeout_seconds,
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Execution timed out.",
            "returncode": 124,
        }
    finally:
        os.remove(temp_file_path)
