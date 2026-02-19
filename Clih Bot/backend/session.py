"""
CLIHBot — Terminal Session Launcher

Usage:
    python session.py

Creates a named, timestamped terminal session and wires it up to the CLIHBot
log watcher automatically. No .env editing required.

Session names look like:  swift-eagle-20260218-143022
  ↑ adjective  ↑ noun      ↑ date       ↑ time
The word pair is easy to say aloud ("the swift-eagle session").
The timestamp makes every session unique and naturally sortable.
"""
from __future__ import annotations

import datetime
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

# ── Word bank ─────────────────────────────────────────────────────────────────
# Short, unambiguous words — easy to type, say, and remember.

ADJECTIVES = [
    "amber", "azure", "bold", "brave", "calm", "chill", "clean", "clear",
    "cold", "cool", "crisp", "cyber", "dark", "deep", "dense", "deft",
    "dual", "dusk", "echo", "elite", "fast", "feral", "firm", "fixed",
    "flux", "gray", "green", "grim", "hard", "hazy", "hot", "iron",
    "jade", "keen", "kind", "late", "lean", "live", "lone", "loud",
    "mute", "neon", "null", "null", "open", "pale", "prime", "pure",
    "quick", "quiet", "rapid", "raw", "real", "red", "royal", "safe",
    "sharp", "silent", "sleek", "slim", "slow", "smart", "soft", "solar",
    "solid", "stark", "static", "still", "storm", "swift", "teal", "thin",
    "tight", "true", "vast", "void", "warm", "wild", "wise", "zero",
]

NOUNS = [
    "apex", "arc", "ash", "axe", "base", "beam", "bit", "blade",
    "blaze", "block", "bolt", "bond", "byte", "cache", "cell", "chain",
    "chip", "cipher", "claw", "cliff", "clock", "cloud", "code", "core",
    "crow", "crypt", "dart", "dawn", "deck", "depth", "disk", "drift",
    "drone", "dune", "dust", "eagle", "edge", "falcon", "field", "file",
    "flame", "flash", "flux", "forge", "fox", "gate", "ghost", "grid",
    "hash", "hawk", "heap", "hex", "hive", "hook", "hull", "key",
    "layer", "link", "lock", "log", "loop", "mesh", "mint", "mist",
    "mode", "moss", "node", "null", "orbit", "pack", "patch", "path",
    "peak", "pipe", "pixel", "port", "probe", "pulse", "rack", "relay",
    "ring", "root", "route", "rune", "salt", "scope", "seal", "seed",
    "shell", "shift", "signal", "slab", "slate", "spark", "spike", "stack",
    "star", "steel", "storm", "stream", "tide", "token", "trace", "vault",
    "veil", "wave", "wire", "wolf", "wren", "zone",
]


def generate_session_name() -> str:
    adj = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{adj}-{noun}-{ts}"


# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent.resolve()
_SESSIONS_DIR = _HERE / "sessions"
_POINTER_FILE = _SESSIONS_DIR / ".current"


def setup_session(name: str) -> Path:
    """Create the session log file and update the pointer."""
    _SESSIONS_DIR.mkdir(exist_ok=True)

    log_path = _SESSIONS_DIR / f"{name}.log"

    # Write a structured header so the agent has session context
    now = datetime.datetime.now()
    header = (
        f"{'='*60}\n"
        f"  CLIHBot Session: {name}\n"
        f"  Started:         {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  Platform:        {platform.system()} {platform.release()}\n"
        f"  Host:            {platform.node()}\n"
        f"{'='*60}\n\n"
    )
    log_path.write_text(header, encoding="utf-8")

    # Update the pointer so the watcher hot-swaps to this file
    _POINTER_FILE.write_text(str(log_path), encoding="utf-8")

    return log_path


# ── Platform-specific session start ───────────────────────────────────────────

def _is_git_bash() -> bool:
    return sys.platform == "win32" and bool(os.environ.get("MSYSTEM"))


def _is_wsl() -> bool:
    return sys.platform != "win32" and "microsoft" in platform.uname().release.lower()


def start_session_windows(name: str, log_path: Path) -> None:
    """Windows: try PowerShell Start-Transcript, with Git Bash fallback."""
    ps_cmd = f'Start-Transcript -Path "{log_path}" -Append -NoClobber; $host.UI.RawUI.WindowTitle = "{name}"'
    print()
    print(f"  Session : {name}")
    print(f"  Log     : {log_path}")
    print()

    if _is_git_bash():
        # Git Bash can't run PowerShell interactively inline — give both options
        print("  You're in Git Bash. Choose how to start logging:")
        print()
        print("  Option A — open a new PowerShell window (recommended):")
        _print_box(f'powershell -NoExit -Command "{ps_cmd}"')
        print()
        print("  Option B — stay in Git Bash, use tee (partial capture):")
        bash_fn = _bash_tee_snippet(name, log_path)
        _print_box(bash_fn)
        print()
        print("  Option C — add this function to ~/.bashrc for a one-liner next time:")
        _print_box(_bashrc_snippet())
        _try_clipboard(f'powershell -NoExit -Command "{ps_cmd}"')
    else:
        # Regular PowerShell / cmd — open a new titled window
        print("  Starting a new PowerShell window with transcript logging...")
        try:
            subprocess.Popen(
                ["powershell", "-NoExit", "-Command", ps_cmd],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
            print("  PowerShell window opened. Work in that window — CLIHBot is watching.")
        except FileNotFoundError:
            print("  PowerShell not found. Run this manually in PowerShell:")
            _print_box(ps_cmd)
            _try_clipboard(ps_cmd)


def start_session_unix(name: str, log_path: Path) -> None:
    """Linux / macOS / WSL: use `script` to wrap the shell session."""
    # `script` behaviour differs between GNU (Linux) and BSD (macOS)
    is_macos = sys.platform == "darwin"
    if is_macos:
        script_cmd = ["script", "-q", str(log_path)]
    else:
        script_cmd = ["script", "-q", "-f", str(log_path)]

    print()
    print(f"  Session : {name}")
    print(f"  Log     : {log_path}")
    print()

    # Check if `script` is available
    if subprocess.run(["which", "script"], capture_output=True).returncode != 0:
        print("  `script` not found — falling back to tee mode.")
        print("  Add this to your shell session:")
        _print_box(_bash_tee_snippet(name, log_path))
        print()
        print("  Or add to ~/.bashrc for a one-liner:")
        _print_box(_bashrc_snippet())
        return

    print("  Starting logged shell session via `script`...")
    print("  Type `exit` to end the session.")
    print()

    # Exec replaces this process — the user stays in their shell
    os.execvp("script", script_cmd)
    # (unreachable after execvp)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bash_tee_snippet(name: str, log_path: Path) -> str:
    return (
        f'# Paste into your current bash session:\n'
        f'exec > >(tee -a "{log_path}") 2>&1\n'
        f'echo "CLIHBot session: {name}"'
    )


def _bashrc_snippet() -> str:
    script_path = str(_HERE / "session.py").replace("\\", "/")
    return (
        "# Add to ~/.bashrc or ~/.bash_profile:\n"
        f'alias cbot="python \\"{script_path}\\""\n'
        "# Then just run: cbot"
    )


def _print_box(text: str) -> None:
    lines = text.splitlines()
    width = max(len(l) for l in lines) + 4
    print("  ┌" + "─" * width + "┐")
    for line in lines:
        print(f"  │  {line:<{width - 4}}  │")
    print("  └" + "─" * width + "┘")


def _try_clipboard(text: str) -> None:
    """Best-effort clipboard copy — silently skip if unavailable."""
    try:
        if sys.platform == "win32":
            subprocess.run(["clip"], input=text.encode("utf-8"), check=True, capture_output=True)
            print("  (command copied to clipboard)")
        elif sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True, capture_output=True)
            print("  (command copied to clipboard)")
        else:
            subprocess.run(["xclip", "-selection", "clipboard"],
                           input=text.encode("utf-8"), check=True, capture_output=True)
            print("  (command copied to clipboard)")
    except Exception:
        pass  # clipboard is optional


def _print_header(name: str) -> None:
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║              CLIHBot Terminal Session            ║")
    print("  ╚══════════════════════════════════════════════════╝")


def _print_footer() -> None:
    print()
    print("  The CLIHBot agent is now watching this session.")
    print("  Ask it anything about what you see in the terminal.")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    name = generate_session_name()
    log_path = setup_session(name)

    _print_header(name)

    if sys.platform == "win32":
        start_session_windows(name, log_path)
    else:
        start_session_unix(name, log_path)

    _print_footer()


if __name__ == "__main__":
    main()
