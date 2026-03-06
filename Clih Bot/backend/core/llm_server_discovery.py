"""
LLM Server Discovery and Management.

Provides smart discovery of running LLM servers without being invasive.
Supports: llama.cpp, Ollama, vLLM, LM Studio, and any OpenAI-compatible server.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class LLMServer:
    """Represents a discovered LLM server."""
    name: str
    url: str
    is_available: bool = False
    models_count: int = 0
    server_type: str = "unknown"
    description: str = ""

    def __str__(self) -> str:
        status = "available" if self.is_available else "unavailable"
        return f"{self.name} ({self.url}) - {status}"


# Common LLM server endpoints to check
DEFAULT_SERVERS = [
    LLMServer(
        name="llama.cpp",
        url="http://localhost:1234/v1",
        server_type="llama.cpp",
        description="llama.cpp server (default port)",
    ),
    LLMServer(
        name="Ollama",
        url="http://localhost:11434",
        server_type="ollama",
        description="Ollama API",
    ),
    LLMServer(
        name="vLLM",
        url="http://localhost:8000/v1",
        server_type="vllm",
        description="vLLM server",
    ),
    LLMServer(
        name="LM Studio",
        url="http://localhost:1234/v1",
        server_type="lm-studio",
        description="LM Studio API",
    ),
    LLMServer(
        name="text-generation-webui",
        url="http://localhost:5000/v1",
        server_type="tgui",
        description="Text Generation WebUI",
    ),
    LLMServer(
        name="Together AI",
        url="http://localhost:21434/v1",
        server_type="together",
        description="Together AI local server",
    ),
]


async def discover_servers() -> list[LLMServer]:
    """
    Discover available LLM servers on the local machine.

    This is non-invasive - it only makes lightweight HTTP requests to check
    if servers are responding, without attempting to load models or perform
    any operations.
    """
    servers = []

    for server in DEFAULT_SERVERS:
        try:
            # Check if server is reachable with a lightweight request
            is_available, info = await check_server(server.url)
            server.is_available = is_available

            if is_available:
                # Try to get model count
                models_count = await get_model_count(server.url)
                server.models_count = models_count
        except Exception as e:
            logger.debug(f"Failed to check {server.name}: {e}")
            server.is_available = False

        servers.append(server)

    return servers


async def check_server(url: str) -> tuple[bool, dict[str, Any] | None]:
    """
    Check if an LLM server is reachable.

    Uses a lightweight HEAD request to check availability without
    loading any models or performing expensive operations.
    """
    try:
        async with aiohttp.ClientSession() as session:
            # Try HEAD first (lightweight), fallback to GET
            try:
                async with session.head(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    return resp.status == 200, None
            except aiohttp.ClientError:
                # HEAD failed, try GET with minimal data
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        return True, await resp.text()[:200]  # Just get a snippet
                    return False, None
    except Exception as e:
        return False, str(e)


async def get_model_count(url: str) -> int:
    """
    Get the number of models available on the server.

    This is a lightweight operation that doesn't load any models.
    """
    try:
        async with aiohttp.ClientSession() as session:
            # Try /models endpoint (LM Studio, Together)
            try:
                async with session.get(f"{url}/models", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if isinstance(data, dict) and "data" in data:
                            return len(data["data"])
                        return len(data) if isinstance(data, list) else 0
            except aiohttp.ClientError:
                pass

            # Try /v1/models endpoint (vLLM, some others)
            try:
                async with session.get(f"{url}/v1/models", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return len(data.get("data", [])) if isinstance(data, dict) else 0
            except aiohttp.ClientError:
                pass

            # Try /api/tags endpoint (Ollama)
            try:
                async with session.get(f"{url}/api/tags", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return len(data.get("model", [])) if isinstance(data, dict) else 0
            except aiohttp.ClientError:
                pass

        return 0
    except Exception:
        return 0


async def get_server_info(url: str) -> dict[str, Any]:
    """
    Get detailed information about a server.

    This may be more invasive than discovery but provides useful info
    for the UI (model names, capabilities, etc.).
    """
    info = {"url": url, "server_type": "unknown", "models": []}

    try:
        async with aiohttp.ClientSession() as session:
            # Try to identify server type and get model info
            endpoints_to_try = [
                (f"{url}/models", "models"),
                (f"{url}/v1/models", "v1_models"),
                (f"{url}/api/tags", "tags"),
            ]

            for endpoint, key in endpoints_to_try:
                try:
                    async with session.get(endpoint, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            info["server_type"] = endpoint.split("/")[-2]  # Extract type from URL
                            info["models"] = _extract_models(data, key)
                            return info
                except Exception:
                    continue

        info["models"] = []
        return info
    except Exception as e:
        logger.debug(f"Failed to get server info: {e}")
        return info


def _extract_models(data: Any, key: str) -> list[str]:
    """Extract model names from server response."""
    if not isinstance(data, dict):
        return []

    if key == "models":
        models = data.get("data", [])
        return [m.get("id") if isinstance(m, dict) else str(m) for m in models]

    if key == "v1_models":
        models = data.get("data", [])
        return [m.get("id") if isinstance(m, dict) else str(m) for m in models]

    if key == "tags":
        models = data.get("model", [])
        return [m.get("name") if isinstance(m, dict) else str(m) for m in models]

    return []


def make_server_url(server: str | LLMServer) -> str:
    """Convert a server string or LLMServer object to a URL."""
    if isinstance(server, LLMServer):
        return server.url
    return server


def get_default_server_url() -> str:
    """Get the default server URL from environment or return localhost:1234."""
    import os
    return os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")