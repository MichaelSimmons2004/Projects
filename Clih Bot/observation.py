import pyautogui

# Optional: OCR support via pytesseract (install separately if needed)
try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

# Optional: Image processing via OpenCV
try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


def analyze_screen(region: tuple = (0, 0, 800, 600)) -> str:
    """
    Capture a screen region and return analysis.
    All processing is local — no data is sent externally.

    Args:
        region: (x, y, width, height) of the screen area to capture.

    Returns:
        A string describing the captured region (OCR text if available,
        otherwise a mock result).
    """
    screenshot = pyautogui.screenshot(region=region)

    if _TESSERACT_AVAILABLE:
        text = pytesseract.image_to_string(screenshot).strip()
        return f"Screen OCR result:\n{text}" if text else "Screen captured — no readable text found."

    # Fallback: return image dimensions as basic confirmation
    width, height = screenshot.size
    return (
        f"Screen captured ({width}x{height}px). "
        "Install 'pytesseract' and Tesseract-OCR for text extraction."
    )