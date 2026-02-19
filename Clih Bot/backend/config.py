from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the project root once at import time so all defaults are stable.
_PROJECT_ROOT = Path(__file__).parent.resolve()


def _default_sessions_dir() -> str:
    return str(_PROJECT_ROOT / "sessions")


def _default_terminal_log() -> str:
    """
    Cross-platform default — no .env required.
    Points at sessions/.current which session.py keeps updated,
    falling back to a stable path in the OS temp dir.
    """
    current_ptr = _PROJECT_ROOT / "sessions" / ".current"
    if current_ptr.exists():
        try:
            target = current_ptr.read_text().strip()
            if target and Path(target).exists():
                return target
        except OSError:
            pass
    # Stable fallback: <tmpdir>/clihbot/terminal.log
    return str(Path(tempfile.gettempdir()) / "clihbot" / "terminal.log")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # ── LMStudio / Model ──────────────────────────────────────────────────────
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_model: str = "qwen/qwen3-vl-8b"
    lmstudio_api_key: str = "lm-studio"
    model_vision_capable: bool = True

    # ── LMStudio auto-launch ──────────────────────────────────────────────────
    # Automatically start LMStudio's server if it isn't already running.
    lmstudio_auto_start: bool = True
    # Path to the `lms` CLI executable.  Leave blank for auto-detection.
    lmstudio_executable: str = ""
    # Send `lms server stop` when CLIHBot exits.
    # Default false — LMStudio may be used by other apps simultaneously.
    lmstudio_stop_on_exit: bool = False

    # ── Model lifecycle ───────────────────────────────────────────────────────
    # Auto-load the model before the first message if it isn't already loaded.
    model_auto_load: bool = True
    # Idle TTL in seconds passed to LMStudio with every request.
    # LMStudio resets this timer on each request and auto-unloads on expiry.
    # 0 = never auto-unload (rely on LMStudio's own default of 60 min).
    model_idle_ttl_seconds: int = 600  # 10 minutes

    # ── KV Cache quantization ─────────────────────────────────────────────────
    # Reduces VRAM usage significantly with minimal quality loss.
    # Valid values: f32, f16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1, or "" to disable.
    # q8_0 on both K and V typically saves ~30–40% VRAM vs f16.
    # Note: V cache quantization requires flash_attention=True (always enabled here).
    model_k_cache_quant: str = "q8_0"
    model_v_cache_quant: str = "q8_0"

    # ── Context Window ────────────────────────────────────────────────────────
    context_window_tokens: int = 40_000
    response_reserve_tokens: int = 4_096
    tool_result_max_tokens: int = 5_000
    terminal_context_lines: int = 100

    # ── API Server ────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8765

    # ── Browser CDP ───────────────────────────────────────────────────────────
    cdp_port: int = 9222
    cdp_host: str = "localhost"

    # ── Browser Extension ─────────────────────────────────────────────────────
    extension_ws_port: int = 8766

    # ── Terminal Capture ──────────────────────────────────────────────────────
    terminal_mode: Literal["logfile", "pty"] = "logfile"
    # Smart default: reads sessions/.current if present, else OS temp dir.
    # Override in .env only if you want a fixed path regardless of sessions.
    terminal_log_file: str = ""
    terminal_ring_buffer_lines: int = 2_000

    # ── Security Scanner ──────────────────────────────────────────────────────
    alert_severity_threshold: Literal["critical", "high", "medium", "low"] = "high"
    auto_scan_extensions: str = ".py,.js,.ts,.sh,.bash,.php,.rb,.go,.java,.cs,.cpp,.c"

    @property
    def resolved_terminal_log(self) -> str:
        """The actual terminal log path to watch, resolving smart defaults."""
        if self.terminal_log_file:
            return self.terminal_log_file
        return _default_terminal_log()

    @property
    def sessions_dir(self) -> Path:
        return _PROJECT_ROOT / "sessions"

    @property
    def sessions_pointer(self) -> Path:
        """File that always contains the path of the currently active session log."""
        return self.sessions_dir / ".current"

    @property
    def auto_scan_ext_set(self) -> set[str]:
        return {ext.strip() for ext in self.auto_scan_extensions.split(",")}

    @property
    def available_token_budget(self) -> int:
        """Tokens available for system prompt + context + history (reserves space for response)."""
        return self.context_window_tokens - self.response_reserve_tokens

    @property
    def history_soft_limit(self) -> int:
        """Start summarising history when it exceeds this many tokens."""
        return self.available_token_budget - 12_000  # leave ~12k for system + context

    @property
    def cdp_url(self) -> str:
        return f"http://{self.cdp_host}:{self.cdp_port}"

    @property
    def lmstudio_management_url(self) -> str:
        """Base URL for LMStudio's native v1 management API (load/unload/list)."""
        # Strip the /v1 OpenAI-compat suffix → replace with /api/v1
        base = self.lmstudio_base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        return f"{base}/api/v1"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
