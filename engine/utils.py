# engine/utils.py
import os
import platform
import shutil
import pytesseract

def setup_tesseract_path():
    """
    Configure pytesseract path depending on OS.
    Will try environment variable, PATH, and common install paths.
    """
    # 1. If user set env var, use it
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd and os.path.exists(env_cmd):
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return

    # 2. If in PATH, use that
    found = shutil.which("tesseract")
    if found:
        pytesseract.pytesseract.tesseract_cmd = found
        return

    # 3. Fall back to common OS-specific install locations
    system = platform.system().lower()
    if "darwin" in system:  # macOS (Homebrew or MacPorts)
        candidates = [
            "/opt/homebrew/bin/tesseract",
            "/usr/local/bin/tesseract",
            "/opt/local/bin/tesseract",
        ]
    elif "windows" in system:
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Tesseract-OCR\tesseract.exe",
        ]
    else:  # Linux
        candidates = [
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
        ]

    for path in candidates:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return

    # 4. If still nothing → fail
    raise RuntimeError(
        "Tesseract not found. Please install it and set PATH or TESSERACT_CMD."
    )
