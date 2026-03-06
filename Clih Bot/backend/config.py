from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
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

    # ── LLM Server / Model ──────────────────────────────────────────────────────
    # The LLM server that CLIHBot connects to for chat completions.
    # This can be LMStudio, llama.cpp, Ollama, vLLM, or any OpenAI-compatible server.
    llm_server_url: str = Field(
        default="http://localhost:1234/v1",
        validation_alias=AliasChoices("LLM_SERVER_URL", "LMSTUDIO_BASE_URL"),
    )
    llm_server_model: str = Field(
        default="qwen/qwen3-vl-8b",
        validation_alias=AliasChoices("LLM_SERVER_MODEL", "LMSTUDIO_MODEL"),
    )
    llm_server_api_key: str = Field(
        default="lm-studio",
        validation_alias=AliasChoices("LLM_SERVER_API_KEY", "LMSTUDIO_API_KEY"),
    )
    model_vision_capable: bool = True

    # ── Backward Compatibility ──────────────────────────────────────────────────
    # Old variable names for backward compatibility
    @property
    def lmstudio_base_url(self) -> str:
        """Alias for llm_server_url for backward compatibility."""
        return self.llm_server_url

    @property
    def lmstudio_model(self) -> str:
        """Alias for llm_server_model for backward compatibility."""
        return self.llm_server_model

    @property
    def lmstudio_api_key(self) -> str:
        """Alias for llm_server_api_key for backward compatibility."""
        return self.llm_server_api_key

    # ── LMStudio auto-launch ──────────────────────────────────────────────────
    # Automatically start LMStudio's server if it isn't already running.
    lmstudio_auto_start: bool = Field(
        default=True,
        validation_alias=AliasChoices("LMSTUDIO_AUTO_START", "LLM_SERVER_AUTO_START"),
    )
    # Path to the `lms` CLI executable.  Leave blank for auto-detection.
    lmstudio_executable: str = Field(
        default="",
        validation_alias=AliasChoices("LMSTUDIO_EXECUTABLE", "LLM_SERVER_EXECUTABLE"),
    )
    # Send `lms server stop` when CLIHBot exits.
    # Default false — LMStudio may be used by other apps simultaneously.
    lmstudio_stop_on_exit: bool = Field(
        default=False,
        validation_alias=AliasChoices("LMSTUDIO_STOP_ON_EXIT", "LLM_SERVER_STOP_ON_EXIT"),
    )

    # ── Model lifecycle ───────────────────────────────────────────────────────
    # Auto-load the model before the first message if it isn't already loaded.
    # Works with both LMStudio and external LLM servers (llama.cpp, Ollama, etc.)
    model_auto_load: bool = True
    # Automatically detect if an external LLM server is running and use it instead
    # of LMStudio for model loading. Only applies when model_auto_load=true.
    auto_detect_external_server: bool = True
    # Prefer external servers (if detected) over LMStudio when both are available.
    # Set to true if you always use llama.cpp (not LMStudio).
    prefer_external_servers: bool = True
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
    # Enforce KV quantization by requiring the LM Studio Python SDK path.
    # If true and K/V quant is configured, REST fallback is refused.
    lmstudio_require_sdk_for_kv_quant: bool = True

    # ── Model Selector / Ranking ──────────────────────────────────────────────
    # Model ranking mode: 'auto' for smart selection, 'manual' to respect only explicit selection
    model_ranking_mode: str = "manual"
    # Prefer models with larger context windows when multiple options available
    prefer_larger_context: bool = True
    # Maximum VRAM to use for model selection (in MB), None for unlimited
    max_vram_mb: int | None = None
    # Minimum context window required (tokens), 0 for no minimum
    min_context_tokens: int = 0
    # Ranking weights (adjust to prioritize different factors)
    # Sum of all weights is normalized to 100
    model_family_weight: float = 25.0      # Match requested model family
    context_window_weight: float = 20.0    # Larger context windows
    quantization_weight: float = 15.0      # Higher precision quantization
    size_weight: float = 15.0              # Prefer smaller/lighter models
    loaded_weight: float = 10.0            # Prefer already-loaded models
    vram_weight: float = 10.0              # VRAM fit headroom
    manual_weight: float = 10.0            # Explicit user selection

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
