"""
Static security scanner — fast regex-based analysis.

Provides:
  scan_code(content, filename?)  — async wrapper (for tool calls)
  scan_code_sync(content, filename?) — sync version (for auto-scan in file_reader)

Returns a list of Finding dicts:
  {
    "id":          str       — pattern identifier
    "category":    str       — e.g. "Hardcoded Credential"
    "severity":    str       — "critical" | "high" | "medium" | "low"
    "line":        int       — 1-indexed line number
    "match":       str       — the matched text (redacted if secret-like)
    "explanation": str       — brief human-readable explanation
  }
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ── Pattern definitions ────────────────────────────────────────────────────────
# Imported from prompts.security_patterns at call time to allow hot-reloading.

async def scan_code(content: str, filename: str | None = None) -> dict[str, Any]:
    """Async tool function — wraps the synchronous scanner."""
    findings = scan_code_sync(content, filename)
    return _format_result(findings, filename)


def scan_code_sync(content: str, filename: str | None = None) -> list[dict[str, Any]]:
    """Run all security patterns against content. Returns raw finding list."""
    from prompts.security_patterns import PATTERNS

    lines = content.splitlines()
    findings: list[dict[str, Any]] = []

    for pattern_def in PATTERNS:
        regex: re.Pattern = pattern_def["regex"]
        skip_ext = pattern_def.get("skip_extensions", set())
        if filename and skip_ext:
            ext = _ext(filename)
            if ext in skip_ext:
                continue

        for line_no, line in enumerate(lines, start=1):
            match = regex.search(line)
            if match:
                matched_text = match.group(0)
                # Redact likely secret values (keep pattern visible)
                if pattern_def.get("redact"):
                    matched_text = _redact(matched_text)
                findings.append({
                    "id": pattern_def["id"],
                    "category": pattern_def["category"],
                    "severity": pattern_def["severity"],
                    "line": line_no,
                    "match": matched_text,
                    "explanation": pattern_def["explanation"],
                })

    # Deduplicate: same id + line → keep first
    seen: set[tuple[str, int]] = set()
    deduped: list[dict[str, Any]] = []
    for f in findings:
        key = (f["id"], f["line"])
        if key not in seen:
            seen.add(key)
            deduped.append(f)

    # Sort by severity then line
    _sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    deduped.sort(key=lambda f: (_sev_order.get(f["severity"], 9), f["line"]))

    return deduped


def _format_result(findings: list[dict[str, Any]], filename: str | None) -> dict[str, Any]:
    from config import get_settings
    threshold = get_settings().alert_severity_threshold
    _sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    threshold_val = _sev_order.get(threshold, 1)

    alert_findings = [f for f in findings if _sev_order.get(f["severity"], 9) <= threshold_val]

    return {
        "findings": findings,
        "total": len(findings),
        "alerts": alert_findings,    # findings at or above threshold
        "alert_count": len(alert_findings),
        "filename": filename,
        "summary": _summarise(findings),
    }


def _summarise(findings: list[dict[str, Any]]) -> str:
    if not findings:
        return "No security issues detected."
    counts: dict[str, int] = {}
    for f in findings:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1
    parts = [f"{v} {k}" for k, v in sorted(counts.items(), key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x[0], 9))]
    return f"{len(findings)} issue(s) found: {', '.join(parts)}."


def _ext(filename: str) -> str:
    return ("." + filename.rsplit(".", 1)[-1]).lower() if "." in filename else ""


def _redact(text: str) -> str:
    """Partially redact a matched credential string."""
    if len(text) <= 8:
        return "***"
    return text[:4] + "***" + text[-2:]
