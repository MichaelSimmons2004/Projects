"""
Terminal capture — dual mode, with automatic session hot-swap.

Mode A (logfile):
  Run `python session.py` to start a named, timestamped session.
  The session launcher writes the log path to sessions/.current.
  The watcher picks it up automatically — no .env editing required.

  Manual alternatives:
    Linux/macOS:  script -f /tmp/clihbot_terminal.log
    Windows PS:   Start-Transcript -Path $env:TEMP\\clihbot_terminal.log
    Git Bash:     exec > >(tee -a logfile) 2>&1

Mode B (pty):
  A Python PTY shim wraps the shell (Unix/Linux only).
  Use: python -m tools.terminal_watcher

Both modes share the same ring buffer and get_terminal_buffer() function.
The active session name and log path are exposed in the buffer response.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Shared ring buffer — all captures write here
_ring_buffer: collections.deque[str] = collections.deque(maxlen=2000)

# Tracks the currently active session for reporting back to the agent
_active_session: dict[str, str] = {
    "name": "",
    "log_path": "",
}


def _append_lines(text: str) -> None:
    """Split text into lines and append to ring buffer."""
    for line in text.splitlines():
        _ring_buffer.append(line)


def _session_name_from_path(log_path: str) -> str:
    """Extract the session name from a log file path (stem without extension)."""
    return Path(log_path).stem


# ── Public tool function ──────────────────────────────────────────────────────

async def get_terminal_buffer(lines: int = 100) -> dict[str, Any]:
    """Return the last N lines from the terminal ring buffer."""
    lines = min(lines, 500)
    snapshot = list(_ring_buffer)[-lines:]
    result: dict[str, Any] = {
        "lines": snapshot,
        "count": len(snapshot),
        "total_buffered": len(_ring_buffer),
    }
    if _active_session["name"]:
        result["session"] = _active_session["name"]
        result["log_path"] = _active_session["log_path"]
    return result


# ── TerminalWatcher — launched by main.py ─────────────────────────────────────

class TerminalWatcher:
    """Starts the appropriate capture backend and keeps running."""

    def __init__(self, settings: Any | None = None) -> None:
        if settings is None:
            from config import get_settings
            settings = get_settings()
        self._settings = settings

    async def run(self) -> None:
        mode = self._settings.terminal_mode
        if mode == "logfile":
            await self._run_logfile_mode()
        elif mode == "pty":
            await self._run_pty_mode()
        else:
            logger.error("Unknown terminal mode: %s", mode)

    # ── Mode A: log file watcher with session hot-swap ────────────────────────

    async def _run_logfile_mode(self) -> None:
        """
        Watch whichever log file is currently active.

        Priority order for resolving the active log:
          1. sessions/.current pointer file (written by session.py)
          2. TERMINAL_LOG_FILE from .env / config
          3. <tmpdir>/clihbot/terminal.log (built-in default)

        When sessions/.current changes, the watcher hot-swaps to the new file
        without restarting. Each new session clears the ring buffer so the agent
        only sees the current session's output.
        """
        pointer = self._settings.sessions_pointer

        logger.info("Terminal watcher (logfile mode) — pointer: %s", pointer)

        while True:
            log_path = self._resolve_log_path(pointer)

            if not log_path:
                logger.debug("No session log found yet — waiting...")
                await asyncio.sleep(2)
                continue

            logger.info("Watching session log: %s", log_path)
            _active_session["log_path"] = str(log_path)
            _active_session["name"] = _session_name_from_path(str(log_path))

            # Watch this file until the pointer changes or the file disappears
            changed = await self._watch_one_file(log_path, pointer)

            if changed:
                # New session started — clear the buffer so the agent doesn't
                # conflate outputs from different sessions
                _ring_buffer.clear()
                logger.info("Session changed — ring buffer cleared, switching log.")

    def _resolve_log_path(self, pointer: Path) -> Path | None:
        """Return the current active log path, or None if nothing is set up."""
        # Pointer file takes priority
        if pointer.exists():
            try:
                target = Path(pointer.read_text().strip())
                if target.exists():
                    return target
            except OSError:
                pass

        # Config / env override
        cfg_path = self._settings.terminal_log_file
        if cfg_path:
            p = Path(cfg_path)
            if p.exists():
                return p

        # Built-in fallback
        fallback = Path(self._settings.resolved_terminal_log)
        if fallback.exists():
            return fallback

        return None

    async def _watch_one_file(self, log_path: Path, pointer: Path) -> bool:
        """
        Tail log_path, appending new content to the ring buffer.
        Returns True when the pointer changes (new session), False on other exits.
        """
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
        except ImportError:
            logger.warning("watchdog not installed — using polling fallback")
            return await self._poll_until_change(log_path, pointer)

        loop = asyncio.get_event_loop()
        file_modified = asyncio.Event()
        watch_dir = str(log_path.parent)
        abs_log = str(log_path.resolve())
        abs_ptr = str(pointer.resolve()) if pointer.exists() else ""

        class _Handler(FileSystemEventHandler):
            def on_modified(self, event):
                if event.is_directory:
                    return
                src = os.path.abspath(event.src_path)
                if src == abs_log or src == abs_ptr:
                    loop.call_soon_threadsafe(file_modified.set)

        observer = Observer()
        observer.schedule(_Handler(), path=watch_dir, recursive=False)
        # Also watch the sessions dir for pointer changes
        sessions_dir = str(pointer.parent)
        if sessions_dir != watch_dir:
            observer.schedule(_Handler(), path=sessions_dir, recursive=False)
        observer.start()

        position = 0
        try:
            # Seed with existing content
            with open(log_path, "r", errors="replace") as fh:
                content = fh.read()
                if content:
                    _append_lines(content)
                position = fh.tell()

            while True:
                await file_modified.wait()
                file_modified.clear()

                # Check if pointer changed → new session
                new_target = self._resolve_log_path(pointer)
                if new_target and new_target.resolve() != log_path.resolve():
                    return True  # signal hot-swap

                # Read new content from the current log
                try:
                    with open(log_path, "r", errors="replace") as fh:
                        fh.seek(position)
                        new_content = fh.read()
                        position = fh.tell()
                    if new_content:
                        _append_lines(new_content)
                except FileNotFoundError:
                    position = 0

        finally:
            observer.stop()
            observer.join()

    async def _poll_until_change(self, log_path: Path, pointer: Path) -> bool:
        """Polling fallback when watchdog is unavailable."""
        position = 0
        while True:
            # Check for session change
            new_target = self._resolve_log_path(pointer)
            if new_target and new_target.resolve() != log_path.resolve():
                return True

            try:
                with open(log_path, "r", errors="replace") as fh:
                    fh.seek(position)
                    new_content = fh.read()
                    position = fh.tell()
                if new_content:
                    _append_lines(new_content)
            except FileNotFoundError:
                position = 0
            await asyncio.sleep(1)

    # ── Mode B: PTY wrapper ───────────────────────────────────────────────────

    async def _run_pty_mode(self) -> None:
        if sys.platform == "win32":
            logger.warning(
                "PTY mode is not supported on Windows. "
                "Use logfile mode instead (e.g. Start-Transcript)."
            )
            return

        logger.info("Terminal watcher (PTY mode) — starting PTY wrapper")
        shell = os.environ.get("SHELL", "/bin/bash")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._pty_session, shell)

    def _pty_session(self, shell: str) -> None:
        """Run the shell inside a PTY, capturing all output to the ring buffer."""
        try:
            import ptyprocess
        except ImportError:
            logger.error("ptyprocess not installed — cannot start PTY mode")
            return

        proc = ptyprocess.PtyProcessUnicode.spawn([shell])
        logger.info("PTY shell started: %s (pid=%d)", shell, proc.pid)

        while proc.isalive():
            try:
                data = proc.read(4096)
                if data:
                    _append_lines(data)
            except EOFError:
                break
            except Exception as exc:
                logger.debug("PTY read error: %s", exc)
                break

        logger.info("PTY shell exited")


# ── Standalone PTY entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run as a PTY wrapper shell session.
    Usage: python -m tools.terminal_watcher
    The user interacts with their shell normally; all I/O is captured.
    """
    import sys as _sys

    if _sys.platform == "win32":
        print("PTY mode is not supported on Windows. Use logfile mode (Start-Transcript).")
        _sys.exit(1)

    try:
        import ptyprocess as _ptyprocess
    except ImportError:
        print("ptyprocess not installed. Run: pip install ptyprocess")
        _sys.exit(1)

    _shell = os.environ.get("SHELL", "/bin/bash")
    _proc = _ptyprocess.PtyProcessUnicode.spawn([_shell])

    import select as _select
    import termios as _termios
    import tty as _tty

    old_attrs = _termios.tcgetattr(_sys.stdin.fileno())
    try:
        _tty.setraw(_sys.stdin.fileno())
        while _proc.isalive():
            rlist, _, _ = _select.select([_sys.stdin, _proc.fd], [], [], 0.05)
            if _sys.stdin.fileno() in [r.fileno() if hasattr(r, "fileno") else r for r in rlist]:
                data = os.read(_sys.stdin.fileno(), 1024).decode("utf-8", errors="replace")
                _proc.write(data)
            if _proc.fd in rlist or hasattr(_proc, "fd") and _proc.fd in [getattr(r, "fd", r) for r in rlist]:
                try:
                    out = _proc.read(4096)
                    _append_lines(out)
                    _sys.stdout.write(out)
                    _sys.stdout.flush()
                except EOFError:
                    break
    finally:
        _termios.tcsetattr(_sys.stdin.fileno(), _termios.TCSADRAIN, old_attrs)
