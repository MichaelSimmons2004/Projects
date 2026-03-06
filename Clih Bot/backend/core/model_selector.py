"""
Model Selector and Ranking System.

Provides intelligent model selection based on:
1. User's manual selection (highest priority)
2. Auto-selection based on available models and system constraints
3. Family equivalence detection (matching model families across versions)

The ranking algorithm considers:
- Model family compatibility (matching the requested family)
- Quantization level (higher precision = better quality)
- Context window capability (larger is generally better)
- Estimated VRAM usage (smaller is better for constrained hardware)
- Model size (lighter models are faster and more efficient)

Configuration options (set in backend/.env):
- MODEL_RANKING_MODE: 'auto' (smart selection), 'manual' (respect only manual selection)
- PREFER_LARGER_CONTEXT: True/False (prefer larger context windows)
- MAX_VRAM_MB: Maximum VRAM to use (for auto-selection)
- MIN_CONTEXT_TOKENS: Minimum context required (for auto-selection)
- MODEL_FAMILY_WEIGHT: 0-100 (how much to prioritize matching model family)
- CONTEXT_WINDOW_WEIGHT: 0-100 (how much to prioritize context window)
- QUANT_WEIGHT: 0-100 (how much to prioritize quantization quality)
- SIZE_WEIGHT: 0-100 (how much to prefer smaller models)
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from config import get_settings

logger = logging.getLogger(__name__)

# ── Quantization quality weights (higher = better quality) ───────────────────

QUANT_SCORES: dict[str, float] = {
    "f32": 1.0,   # Full precision (very rare for LLMs)
    "f16": 0.9,   # FP16 - standard
    "bf16": 0.9,  # BF16 - similar to FP16
    "q8_0": 0.85, # Q8_0 - very high quality
    "q6_k": 0.84,
    "q5_k_m": 0.83,
    "q5_k_s": 0.82,
    "q4_1": 0.80, # Q4_1 - good balance
    "q4_k_m": 0.81,
    "q4_k_s": 0.80,
    "q4_k_xl": 0.80,
    "q5_0": 0.82, # Q5_0 - slightly better than Q4_1
    "q5_1": 0.83, # Q5_1 - better than Q4_1
    "q4_0": 0.78, # Q4_0 - slightly worse than Q4_1
    "q3_k_xl": 0.74,
    "q2_k_xl": 0.68,
    "iq2_xs": 0.70,
    "iq2_s": 0.73,
    "iq1_s": 0.76,
    "iq1_xs": 0.75,
    "iq3_xs": 0.78,
    "iq3_s": 0.80,
    "iq3_m": 0.82,
    "iq4_nl": 0.84,
    "": 0.0,      # No quantization specified
}

# ── Model family patterns for matching ──────────────────────────────────────

# Family patterns: extracts the base model family from a model identifier
FAMILY_PATTERNS = [
    # Meta models
    (re.compile(r"^meta/llama[^/]*$|^llama[^/]*$"), "llama"),
    (re.compile(r"^llama3[^/]*$|^llama-3[^/]*$"), "llama3"),
    # Qwen models
    (re.compile(r"^qwen[^/]*$|^qwen/[^/]*$"), "qwen"),
    # Mistral models
    (re.compile(r"^mistral[^/]*$|^mistral[^/]*$"), "mistral"),
    # Gemma models
    (re.compile(r"^gemma[^/]*$|^google/gemma[^/]*$"), "gemma"),
    # Phi models
    (re.compile(r"^microsoft/phi[^/]*$|^phi[^/]*$"), "phi"),
    # Llama-based models (including Llama 3.1, 3.2, etc.)
    (re.compile(r"^llama3\.[0-9]"), "llama3"),
    (re.compile(r"^llama[^/]*$"), "llama"),
    # Generic fallback
    (re.compile(r"^([^/]+)/([^/]+)$"), lambda m: m.group(1)),
]


def extract_family(model_id: str) -> str:
    """
    Extract the model family from a model identifier.

    Examples:
      "qwen/qwen3-vl-8b" -> "qwen"
      "mistralai/mistral-7b-instruct" -> "mistral"
      "meta-llama/Llama-3-8B" -> "llama3"
    """
    lowered = model_id.lower()
    if "qwen" in lowered:
        return "qwen"
    if "llama" in lowered:
        return "llama3" if "llama3" in lowered or "llama-3" in lowered else "llama"
    if "mistral" in lowered:
        return "mistral"
    if "gemma" in lowered:
        return "gemma"
    if "phi" in lowered:
        return "phi"
    if "liquid" in lowered or "lfm" in lowered:
        return "liquid"
    if "glm" in lowered:
        return "glm"

    for pattern, result in FAMILY_PATTERNS:
        match = pattern.match(model_id)
        if match:
            if callable(result):
                return result(match)
            return result
    return "unknown"


def normalize_model_id(model_id: str) -> str:
    """
    Normalize a model identifier for comparison.

    Examples:
      "qwen/qwen3-vl-8b" -> "qwen/qwen3-vl-8b"
      "mistralai/mistral-7b" -> "mistralai/mistral-7b"
      "qwen3-vl-8b" -> "qwen/qwen3-vl-8b"
    """
    model_id = model_id.strip()

    # If it doesn't contain a slash, add a default namespace
    if "/" not in model_id:
        # Try to detect the family
        family = extract_family(model_id)
        if family != "unknown" and "/" not in model_id:
            model_id = f"{family}/{model_id}"

    return model_id


# ── Model Info Data Structure ───────────────────────────────────────────────

@dataclass
class ModelInfo:
    """Information about a model."""
    id: str
    family: str = ""
    model_type: str = "llm"
    architecture: str = ""
    size_gb: float | None = None
    file_size_gb: float | None = None
    parameter_count_b: float | None = None
    quantization: str = ""
    context_window: int | None = None
    description: str = ""
    is_loaded: bool = False
    vision: bool = False
    trained_for_tool_use: bool = False

    # Pre-computed score for ranking (0-100)
    _rank_score: float = field(default=0.0, repr=False)
    score_breakdown: dict[str, Any] = field(default_factory=dict, repr=False)
    estimated_vram_gb: float | None = field(default=None, repr=False)

    @property
    def rank_score(self) -> float:
        """Get the computed rank score."""
        return self._rank_score

    @property
    def family_normalized(self) -> str:
        """Get the normalized family name."""
        if not self.family:
            self.family = extract_family(self.id)
        return self.family

    @classmethod
    def from_api_data(cls, model_data: dict[str, Any], settings: Any) -> ModelInfo:
        """Create a ModelInfo from LMStudio API data."""
        model_id = (
            model_data.get("modelKey")
            or model_data.get("id")
            or model_data.get("indexedModelIdentifier")
            or model_data.get("path")
            or ""
        )
        size_bytes = model_data.get("sizeBytes")
        file_size_gb = (float(size_bytes) / (1024 ** 3)) if size_bytes else None
        parameter_count_b = cls._parse_parameter_count_b(
            model_data.get("paramsString"),
            model_id,
        )
        size_gb = parameter_count_b or cls._estimate_size_from_id(model_id)
        quantization = cls._normalize_quantization(
            model_data.get("quantization"),
            model_data.get("selectedVariant"),
            model_id,
        )
        description_parts = [
            part for part in [
                model_data.get("displayName"),
                model_data.get("publisher"),
            ] if part
        ]

        return cls(
            id=model_id,
            model_type=str(model_data.get("type", "llm")).lower(),
            architecture=str(model_data.get("architecture", "")),
            size_gb=size_gb,
            file_size_gb=file_size_gb,
            parameter_count_b=parameter_count_b,
            quantization=quantization,
            context_window=model_data.get("maxContextLength") or model_data.get("contextWindow"),
            description=" | ".join(description_parts),
            is_loaded=model_data.get("isLoaded", False),
            vision=bool(model_data.get("vision", False)),
            trained_for_tool_use=bool(model_data.get("trainedForToolUse", False)),
        )

    @staticmethod
    def _normalize_quantization(quantization: Any, selected_variant: str | None, model_id: str) -> str:
        if isinstance(quantization, dict):
            name = quantization.get("name")
            if name:
                return str(name).lower()
        if isinstance(quantization, str) and quantization:
            return quantization.lower()

        candidate = (selected_variant or model_id or "").lower()
        match = re.search(r"(bf16|f16|q\d(?:_[0-9a-z]+)+|iq\d(?:_[0-9a-z]+)+)", candidate)
        return match.group(1) if match else ""

    @staticmethod
    def _parse_parameter_count_b(params_string: Any, model_id: str) -> float | None:
        text = str(params_string or "").strip().lower()
        model_text = model_id.lower()
        match = re.search(r"(\d+(?:\.\d+)?)\s*b", text or model_text)
        if match:
            return float(match.group(1))
        return None

    @staticmethod
    def _estimate_size_from_id(model_id: str) -> float:
        """Estimate model size from model ID patterns."""
        size_gb = None

        # Parse model ID to extract parameters
        parts = model_id.replace("/", "_").lower()

        # Check for common size indicators
        if "8b" in parts or "8k" in parts:
            size_gb = 8.0
        elif "7b" in parts or "7k" in parts:
            size_gb = 7.0
        elif "13b" in parts:
            size_gb = 13.0
        elif "14b" in parts:
            size_gb = 14.0
        elif "33b" in parts:
            size_gb = 33.0
        elif "40b" in parts:
            size_gb = 40.0
        elif "70b" in parts:
            size_gb = 70.0
        elif "120b" in parts:
            size_gb = 120.0
        elif "175b" in parts or "176b" in parts:
            size_gb = 175.0
        elif "8k" in parts:
            size_gb = 8.0

        # Check for quantization indicators
        if size_gb:
            if "q4_0" in parts or "q4_0" in parts:
                size_gb *= 0.40
            elif "q4_1" in parts:
                size_gb *= 0.45
            elif "q5_0" in parts:
                size_gb *= 0.50
            elif "q5_1" in parts:
                size_gb *= 0.55
            elif "q8_0" in parts:
                size_gb *= 0.80
            elif "f16" in parts or "fp16" in parts:
                size_gb *= 1.0

        return size_gb or 7.0  # Default to 7B if unknown

    def estimate_vram_usage(self, kv_quantization: str | None = None) -> float:
        """
        Estimate VRAM usage in GB.

        Base model weights (7B equivalent):
        - 7B params * 4 bytes (FP8) / 1024 = ~0.027 GB per B
        - Adjust for quantization

        KV cache adds ~0.1-0.2 GB per 10k tokens depending on quantization.
        """
        if not self.size_gb:
            self.size_gb = self._estimate_size_from_id(self.id)

        file_gb = self.file_size_gb or max((self.size_gb or 0) * 0.55, 0.1)
        quant_name = (self.quantization or kv_quantization or "").lower()
        quant_score = QUANT_SCORES.get(quant_name, QUANT_SCORES.get((kv_quantization or "").lower(), 0.78))

        # File size is the dominant signal. Add modest runtime overhead and a context-scaled KV term.
        runtime_overhead = max(0.6, file_gb * 0.08)
        parameter_count = self.parameter_count_b or self.size_gb or 7.0
        context_tokens = self.context_window or 32768
        context_ratio = max(context_tokens / 131072, 0.25)
        kv_vram = parameter_count * 0.045 * context_ratio * max(0.55, quant_score)

        return round(file_gb + runtime_overhead + kv_vram, 2)

    def calculate_rank_details(
        self,
        requested_family: str = "",
        requested_model: str = "",
        max_vram_mb: float | None = None,
        min_context: int = 0,
        prefer_larger_context: bool = False,
        weights: dict[str, float] | None = None,
        settings: Any | None = None,
    ) -> dict[str, Any]:
        """
        Calculate a normalized ranking score and expose the exact breakdown.

        The score is normalized against only the active criteria so it remains
        readable even when some constraints are unset.
        """
        weights = weights or {
            "family": 25.0,
            "context": 20.0,
            "quant": 15.0,
            "size": 15.0,
            "loaded": 10.0,
            "vram": 10.0,
            "manual": 10.0,
        }
        kv_quantization = getattr(settings, "model_k_cache_quant", "q8_0") if settings else "q8_0"
        estimated_vram_gb = self.estimate_vram_usage(kv_quantization)
        self.estimated_vram_gb = estimated_vram_gb

        active_weights: dict[str, float] = {
            "quant": weights.get("quant", 0.0),
            "size": weights.get("size", 0.0),
            "loaded": weights.get("loaded", 0.0),
        }
        if requested_model:
            active_weights["manual"] = weights.get("manual", 0.0)
        if requested_family:
            active_weights["family"] = weights.get("family", 0.0)
        if min_context > 0 or prefer_larger_context:
            active_weights["context"] = weights.get("context", 0.0)
        if max_vram_mb:
            active_weights["vram"] = weights.get("vram", 0.0)

        max_score = sum(active_weights.values()) or 1.0
        components: dict[str, Any] = {}
        raw_score = 0.0

        if self.model_type != "llm":
            details = {
                "score": 0.0,
                "raw_score": 0.0,
                "max_score": round(max_score, 2),
                "estimated_vram_gb": round(estimated_vram_gb, 2),
                "components": {
                    "eligibility": {
                        "weight": 0.0,
                        "ratio": 0.0,
                        "contribution": 0.0,
                        "reason": f"Excluded because type is {self.model_type}",
                    }
                },
            }
            self.score_breakdown = details
            self._rank_score = 0.0
            return details

        if not self.trained_for_tool_use:
            details = {
                "score": 0.0,
                "raw_score": 0.0,
                "max_score": round(max_score, 2),
                "estimated_vram_gb": round(estimated_vram_gb, 2),
                "components": {
                    "eligibility": {
                        "weight": 0.0,
                        "ratio": 0.0,
                        "contribution": 0.0,
                        "reason": "Excluded because model is not marked as tool-use capable",
                    }
                },
            }
            self.score_breakdown = details
            self._rank_score = 0.0
            return details

        def add_component(name: str, ratio: float, reason: str) -> None:
            nonlocal raw_score
            weight = active_weights.get(name, 0.0)
            if weight <= 0:
                return
            clamped_ratio = max(0.0, min(1.0, ratio))
            contribution = weight * clamped_ratio
            raw_score += contribution
            components[name] = {
                "weight": round(weight, 2),
                "ratio": round(clamped_ratio, 3),
                "contribution": round(contribution, 2),
                "reason": reason,
            }

        manual_match = bool(requested_model and self.id.lower() == requested_model.lower())
        add_component(
            "manual",
            1.0 if manual_match else 0.0,
            "Exact manual match" if manual_match else "Does not match the requested exact model",
        )

        family_match = bool(requested_family and self.family_normalized.lower() == requested_family.lower())
        add_component(
            "family",
            1.0 if family_match else 0.0,
            "Matches requested family" if family_match else "Different model family",
        )

        if min_context > 0:
            context_ratio = min((self.context_window or 0) / min_context, 1.0) if self.context_window else 0.0
            context_reason = f"Context {self.context_window or 0} / required {min_context}"
        else:
            context_ratio = min((self.context_window or 0) / 200000, 1.0) if self.context_window else 0.0
            context_reason = f"Context window {self.context_window or 0}"
        add_component("context", context_ratio, context_reason)

        quant_ratio = QUANT_SCORES.get(self.quantization, 0.0)
        add_component("quant", quant_ratio, f"Quantization {self.quantization or 'unknown'}")

        # Prefer the largest capable model that remains within budget. Outside budget, score drops sharply.
        if max_vram_mb:
            safe_budget_gb = (max_vram_mb / 1024) * 0.9
            if estimated_vram_gb <= safe_budget_gb:
                size_ratio = min((self.parameter_count_b or self.size_gb or 0) / 35.0, 1.0)
                size_reason = f"Model scale {(self.parameter_count_b or self.size_gb or 0):.1f}B fits safe budget"
            else:
                size_ratio = 0.0
                size_reason = f"Estimated VRAM {estimated_vram_gb:.2f} GB exceeds safe budget {safe_budget_gb:.2f} GB"
        else:
            size_ratio = min((self.parameter_count_b or self.size_gb or 0) / 35.0, 1.0)
            size_reason = f"Model scale {(self.parameter_count_b or self.size_gb or 0):.1f}B"
        add_component("size", size_ratio, size_reason)

        add_component("loaded", 1.0 if self.is_loaded else 0.0, "Already loaded" if self.is_loaded else "Not currently loaded")

        if max_vram_mb:
            vram_cap_gb = (max_vram_mb / 1024) * 0.9
            if estimated_vram_gb <= vram_cap_gb:
                vram_ratio = 1.0
                vram_reason = f"Fits within safe VRAM budget {vram_cap_gb:.2f} GB"
            else:
                vram_ratio = 0.0
                vram_reason = f"Exceeds safe VRAM budget {vram_cap_gb:.2f} GB"
            add_component("vram", vram_ratio, vram_reason)

        final_score = max(0.0, min(100.0, (raw_score / max_score) * 100.0))
        details = {
            "score": round(final_score, 2),
            "raw_score": round(raw_score, 2),
            "max_score": round(max_score, 2),
            "estimated_vram_gb": round(estimated_vram_gb, 2),
            "components": components,
        }
        self.score_breakdown = details
        self._rank_score = final_score
        return details

    def calculate_rank_score(
        self,
        requested_family: str = "",
        requested_model: str = "",
        max_vram_mb: float | None = None,
        min_context: int = 0,
        prefer_larger_context: bool = False,
        weights: dict[str, float] | None = None,
        settings: Any | None = None,
    ) -> float:
        return self.calculate_rank_details(
            requested_family=requested_family,
            requested_model=requested_model,
            max_vram_mb=max_vram_mb,
            min_context=min_context,
            prefer_larger_context=prefer_larger_context,
            weights=weights,
            settings=settings,
        )["score"]

    def __str__(self) -> str:
        return f"{self.id} ({self.family})"


# ── ModelSelector Class ─────────────────────────────────────────────────────

class ModelSelector:
    """
    Selects the best model based on ranking criteria.

    Usage:
        selector = ModelSelector()
        models = await selector.get_available_models()
        selected = selector.select_best_model(models, manual_preference=None)
    """

    def __init__(self, settings: Any | None = None):
        self._settings = settings or get_settings()
        self._model_cache: dict[str, ModelInfo] = {}
        self._available_models: list[ModelInfo] | None = None
        self.refresh_from_settings()

    def refresh_from_settings(self) -> None:
        """Reload selector configuration from the current settings object."""
        self._available_models = None
        self._ranking_mode = getattr(self._settings, "model_ranking_mode", "auto")
        self._prefer_larger_context = getattr(self._settings, "prefer_larger_context", True)
        self._max_vram_mb = getattr(self._settings, "max_vram_mb", None)
        self._min_context = getattr(self._settings, "min_context_tokens", 0)

        self._family_weight = getattr(self._settings, "model_family_weight", 25.0)
        self._context_weight = getattr(self._settings, "context_window_weight", 20.0)
        self._quant_weight = getattr(self._settings, "quantization_weight", 15.0)
        self._size_weight = getattr(self._settings, "size_weight", 15.0)
        self._loaded_weight = getattr(self._settings, "loaded_weight", 10.0)
        self._vram_weight = getattr(self._settings, "vram_weight", 10.0)
        self._manual_weight = getattr(self._settings, "manual_weight", 10.0)

        total = sum([
            self._family_weight,
            self._context_weight,
            self._quant_weight,
            self._size_weight,
            self._loaded_weight,
            self._vram_weight,
            self._manual_weight,
        ]) or 1.0
        scale = 100.0 / total
        self._weights = {
            "family": self._family_weight * scale,
            "context": self._context_weight * scale,
            "quant": self._quant_weight * scale,
            "size": self._size_weight * scale,
            "loaded": self._loaded_weight * scale,
            "vram": self._vram_weight * scale,
            "manual": self._manual_weight * scale,
        }

    def get_settings(self) -> Any:
        """Get the settings object."""
        return self._settings

    async def get_available_models(self) -> list[ModelInfo]:
        """
        Get all available models from LMStudio.

        Returns:
            List of ModelInfo objects, sorted by rank score.
        """
        if self._available_models is not None:
            return self._available_models

        launcher = self._get_launcher()
        model_entries = await launcher.list_available_model_entries()

        if not model_entries:
            logger.warning("No models found in LMStudio")
            return []

        models = []
        for model_entry in model_entries:
            info = ModelInfo.from_api_data(model_entry, self._settings)
            if info.model_type != "llm":
                continue
            self._model_cache[info.id] = info
            models.append(info)

        models = self.rank_models(models=models)
        self._available_models = models
        return models

    def rank_models(
        self,
        models: list[ModelInfo],
        manual_preference: str | None = None,
        override_max_vram_mb: float | None = None,
    ) -> list[ModelInfo]:
        """Score and sort models according to the current selector settings."""
        requested_family = ""
        requested_model = manual_preference or ""
        if requested_model:
            requested_model = normalize_model_id(requested_model)
            requested_family = extract_family(requested_model)
        else:
            requested_family = extract_family(getattr(self._settings, "lmstudio_model", ""))
        effective_max_vram_mb = override_max_vram_mb if override_max_vram_mb is not None else self._max_vram_mb

        for model in models:
            model.calculate_rank_details(
                requested_family=requested_family,
                requested_model=requested_model,
                max_vram_mb=effective_max_vram_mb,
                min_context=self._min_context,
                prefer_larger_context=self._prefer_larger_context,
                weights=self._weights,
                settings=self._settings,
            )

        models.sort(key=lambda m: m.rank_score, reverse=True)
        return models

    def select_best_model(
        self,
        models: list[ModelInfo] | None = None,
        manual_preference: str | None = None,
        override_max_vram_mb: float | None = None,
    ) -> tuple[ModelInfo | None, str]:
        """
        Select the best model from available models.

        Args:
            models: List of available ModelInfo objects. If None, fetches from LMStudio.
            manual_preference: User's manually selected model ID.

        Returns:
            Tuple of (selected ModelInfo, reason string).
        """
        if models is None:
            models = asyncio.get_event_loop().run_until_complete(
                self.get_available_models()
            )

        if not models:
            return None, "No models available in LMStudio"

        requested_model = normalize_model_id(manual_preference) if manual_preference else ""

        if manual_preference:
            for model in models:
                if model.id.lower() == requested_model.lower():
                    return model, f"Manual selection: {model.id}"

        ranked_models = self.rank_models(
            models=list(models),
            manual_preference=manual_preference,
            override_max_vram_mb=override_max_vram_mb,
        )
        if not ranked_models:
            return None, "Failed to score any models"

        best_model = ranked_models[0]
        best_score = best_model.rank_score
        reason = f"Best match: {best_model.id} (score: {best_score:.1f})"
        if manual_preference and requested_model and best_model.id.lower() != requested_model.lower():
            reason += f" (closest match to {requested_model})"
        return best_model, reason

    def get_model_family(self, model_id: str) -> str:
        """Get the family name for a model."""
        return extract_family(model_id)

    def find_equivalent_model(
        self,
        family: str,
        max_vram_mb: float | None = None,
        min_context: int = 0,
    ) -> ModelInfo | None:
        """
        Find the best model in a specific family.

        Useful for when the user wants a specific model family
        but we should pick the best available version.
        """
        models = asyncio.get_event_loop().run_until_complete(
            self.get_available_models()
        )

        family_models = [m for m in models if m.family_normalized.lower() == family.lower()]

        if not family_models:
            return None

        # Score family models
        best = None
        best_score = -1

        for model in family_models:
            score = model.calculate_rank_score(
                requested_family=family,
                max_vram_mb=max_vram_mb,
                min_context=min_context,
            )
            if score > best_score:
                best_score = score
                best = model

        return best

    def _get_launcher(self):
        from core.lmstudio_launcher import get_launcher
        return get_launcher()

    def _get_auth_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self._settings.lmstudio_api_key
        if key and key != "lm-studio":
            headers["Authorization"] = f"Bearer {key}"
        return headers


# ── Convenience Functions ───────────────────────────────────────────────────

def get_model_selector(settings: Any | None = None) -> ModelSelector:
    """Get or create a ModelSelector instance."""
    selector = ModelSelector(settings)
    return selector


async def get_best_model(
    settings: Any | None = None,
    manual_preference: str | None = None,
) -> tuple[ModelInfo | None, str]:
    """
    Convenience function to get the best model.

    Args:
        settings: Optional settings object.
        manual_preference: User's manually selected model ID.

    Returns:
        Tuple of (ModelInfo, reason string).
    """
    selector = get_model_selector(settings)
    return selector.select_best_model(manual_preference=manual_preference)
