from __future__ import annotations

import subprocess
import sys
import threading
import time
import os
from pathlib import Path

import httpx

BACKEND_URL = os.getenv("CLIHBOT_API_BASE_URL", "http://localhost:8765")
_STARTUP_TIMEOUT = 120

_THIS_DIR = Path(__file__).parent
_BACKEND_DIR = _THIS_DIR / "backend"
_BACKEND_MAIN = _BACKEND_DIR / "main.py"

_start_lock: threading.Lock = threading.Lock()
_backend_proc: subprocess.Popen | None = None


def _healthy() -> bool:
    try:
        r = httpx.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _ensure_backend() -> None:
    global _backend_proc

    if _healthy():
        return

    if os.environ.get("CLIHBOT_EMBEDDED_BACKEND") == "1":
        raise RuntimeError(
            "CLIHBot backend is running, but the chat API is unavailable. "
            "Check the backend log for the underlying error."
        )

    with _start_lock:
        if _healthy():
            return

        python = sys.executable
        print(
            f"[CLIHBot] Backend not running - starting it now...\n"
            f"          python : {python}\n"
            f"          main   : {_BACKEND_MAIN}",
            flush=True,
        )

        _backend_proc = subprocess.Popen(
            [python, str(_BACKEND_MAIN)],
            cwd=str(_BACKEND_DIR),
            stdout=None,
            stderr=None,
        )

        deadline = time.monotonic() + _STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if _healthy():
                print("[CLIHBot] Backend is ready.", flush=True)
                return
            time.sleep(1)

        raise RuntimeError(
            f"CLIHBot backend didn't respond within {_STARTUP_TIMEOUT}s. "
            "Check the terminal for error messages."
        )



def get_ai_response_stream(message: str):
    try:
        _ensure_backend()
    except Exception as exc:
        yield f"[CLIHBot startup error: {exc}]"
        return

    import json as _json

    try:
        with httpx.stream(
            "POST",
            f"{BACKEND_URL}/chat/stream",
            json={"message": message},
            timeout=180,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = _json.loads(line[6:])
                if "chunk" in payload:
                    yield payload["chunk"]
                elif "error" in payload:
                    yield f"[Backend error: {payload['error']}]"
                    return
    except Exception as exc:
        yield f"[Backend error: {exc}]"



def get_ai_response(message: str) -> str:
    try:
        _ensure_backend()
    except Exception as exc:
        return f"[CLIHBot startup error: {exc}]"

    try:
        r = httpx.post(
            f"{BACKEND_URL}/chat",
            json={"message": message},
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["response"]
    except httpx.HTTPStatusError as exc:
        return f"[Backend HTTP {exc.response.status_code}: {exc.response.text[:300]}]"
    except Exception as exc:
        return f"[Backend error: {exc}]"
