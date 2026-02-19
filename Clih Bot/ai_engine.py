"""
ai_engine.py — CLIHBot backend bridge.

Replaces the original mock placeholder.  Forwards every chat message to the
CLIHBot FastAPI backend (http://localhost:8765/chat) and returns the real AI
response.  If the backend isn't running yet this module starts it automatically
in a background process and waits for it to come up — the user never needs to
launch a second terminal.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

# ── Config ────────────────────────────────────────────────────────────────────

BACKEND_URL = "http://localhost:8765"
_STARTUP_TIMEOUT = 120   # seconds to wait for the backend to come up

_THIS_DIR    = Path(__file__).parent
_BACKEND_DIR = _THIS_DIR / "backend"
_BACKEND_MAIN = _BACKEND_DIR / "main.py"

# ── Internal state ────────────────────────────────────────────────────────────

_start_lock: threading.Lock = threading.Lock()
_backend_proc: subprocess.Popen | None = None   # noqa: UP007  (compat)


def _healthy() -> bool:
    """Return True if the backend responds to /health."""
    try:
        r = httpx.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _ensure_backend() -> None:
    """Start the backend process if it isn't already running."""
    global _backend_proc

    if _healthy():
        return

    with _start_lock:
        # Re-check under the lock in case another thread already started it.
        if _healthy():
            return

        python = sys.executable
        print(
            f"[CLIHBot] Backend not running — starting it now …\n"
            f"          python : {python}\n"
            f"          main   : {_BACKEND_MAIN}",
            flush=True,
        )

        _backend_proc = subprocess.Popen(
            [python, str(_BACKEND_MAIN)],
            cwd=str(_BACKEND_DIR),
            # Inherit stdio so backend logs show in the same terminal
            stdout=None,
            stderr=None,
        )

        # Wait until the backend accepts requests.
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


# ── Public API (called by main.py) ────────────────────────────────────────────

def get_ai_response_stream(message: str):
    """
    Generator that yields response chunks as they arrive from the backend.
    Use this with Gradio's streaming ChatInterface (yield from a generator fn).
    """
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
    """
    Send *message* to the CLIHBot backend and return the assistant's reply.
    Starts the backend automatically on the first call if needed.
    """
    try:
        _ensure_backend()
    except Exception as exc:
        return f"[CLIHBot startup error: {exc}]"

    try:
        r = httpx.post(
            f"{BACKEND_URL}/chat",
            json={"message": message},
            timeout=180,   # model inference can be slow on CPU
        )
        r.raise_for_status()
        return r.json()["response"]
    except httpx.HTTPStatusError as exc:
        return f"[Backend HTTP {exc.response.status_code}: {exc.response.text[:300]}]"
    except Exception as exc:
        return f"[Backend error: {exc}]"
