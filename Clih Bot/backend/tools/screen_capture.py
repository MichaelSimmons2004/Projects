"""
Screen capture tool.

Supports three modes:
  - full        : entire desktop (all monitors combined)
  - active_window : the currently focused window (Windows/macOS/Linux best-effort)
  - region      : a user-specified rectangle {x, y, width, height}

Returns a dict with:
  - "image_b64"   : base64-encoded PNG (for vision-capable models)
  - "image_bytes" : raw bytes (for passing to multimodal message builder)
  - "description" : human-readable metadata (resolution, mode)
  - "error"       : present only when capture fails
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

# ── Optional OCR support ──────────────────────────────────────────────────────
try:
    import pytesseract
    from PIL import Image as _PilImage
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False


async def capture_screen(
    mode: str = "active_window",
    region: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Async wrapper — runs the blocking capture in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _capture_sync, mode, region)


def _capture_sync(mode: str, region: dict[str, int] | None) -> dict[str, Any]:
    try:
        import mss
        import mss.tools
        from PIL import Image
    except ImportError:
        return {"error": "mss and/or Pillow not installed. Run: pip install mss Pillow"}

    try:
        with mss.mss() as sct:
            if mode == "full":
                # All monitors combined
                monitor = sct.monitors[0]
                raw = sct.grab(monitor)

            elif mode == "active_window":
                monitor = _get_active_window_rect()
                if monitor is None:
                    # Fall back to primary monitor
                    monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                    logger.debug("Active window detection failed — using primary monitor.")
                raw = sct.grab(monitor)

            elif mode == "region":
                if not region:
                    return {"error": "mode='region' requires a 'region' dict with x, y, width, height."}
                monitor = {
                    "left": region["x"],
                    "top": region["y"],
                    "width": region["width"],
                    "height": region["height"],
                }
                raw = sct.grab(monitor)

            else:
                return {"error": f"Unknown mode '{mode}'. Use 'full', 'active_window', or 'region'."}

            # Convert mss screenshot to PIL Image → PNG bytes
            img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            png_bytes = buf.getvalue()

        b64 = base64.b64encode(png_bytes).decode()
        result: dict[str, Any] = {
            "image_b64": b64,
            "image_bytes": png_bytes,
            "description": f"Screenshot ({mode}) — {raw.width}x{raw.height}px, {len(png_bytes)//1024}KB",
        }

        # Optionally add OCR text for non-vision models
        from config import get_settings
        if not get_settings().model_vision_capable and _OCR_AVAILABLE:
            try:
                pil_img = _PilImage.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                result["ocr_text"] = pytesseract.image_to_string(pil_img)
            except Exception as ocr_exc:
                logger.debug("OCR failed: %s", ocr_exc)

        return result

    except Exception as exc:
        logger.error("Screen capture failed: %s", exc)
        return {"error": str(exc)}


def _get_active_window_rect() -> dict[str, int] | None:
    """Best-effort active window bounding rect.  Returns None on failure."""
    if sys.platform == "win32":
        return _active_window_windows()
    elif sys.platform == "darwin":
        return _active_window_macos()
    else:
        return _active_window_linux()


def _active_window_windows() -> dict[str, int] | None:
    try:
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        rect = ctypes.wintypes.RECT()
        ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
        return {
            "left": rect.left,
            "top": rect.top,
            "width": rect.right - rect.left,
            "height": rect.bottom - rect.top,
        }
    except Exception as exc:
        logger.debug("Windows active window detection failed: %s", exc)
        return None


def _active_window_macos() -> dict[str, int] | None:
    try:
        from AppKit import NSWorkspace  # type: ignore[import]
        from Quartz import (  # type: ignore[import]
            CGWindowListCopyWindowInfo,
            kCGNullWindowID,
            kCGWindowListOptionOnScreenOnly,
        )
        active_app = NSWorkspace.sharedWorkspace().activeApplication()
        pid = active_app["NSApplicationProcessIdentifier"]
        windows = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
        for win in windows:
            if win.get("kCGWindowOwnerPID") == pid:
                bounds = win.get("kCGWindowBounds", {})
                return {
                    "left": int(bounds.get("X", 0)),
                    "top": int(bounds.get("Y", 0)),
                    "width": int(bounds.get("Width", 800)),
                    "height": int(bounds.get("Height", 600)),
                }
    except Exception as exc:
        logger.debug("macOS active window detection failed: %s", exc)
    return None


def _active_window_linux() -> dict[str, int] | None:
    try:
        import subprocess
        # xdotool must be installed
        result = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowgeometry", "--shell"],
            capture_output=True, text=True, timeout=2,
        )
        info: dict[str, int] = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                try:
                    info[k.strip()] = int(v.strip())
                except ValueError:
                    pass
        if "X" in info and "Y" in info and "WIDTH" in info and "HEIGHT" in info:
            return {"left": info["X"], "top": info["Y"], "width": info["WIDTH"], "height": info["HEIGHT"]}
    except Exception as exc:
        logger.debug("Linux active window detection failed: %s", exc)
    return None
