# engine/core.py
import os
import re
import json
import platform
import shutil
from datetime import datetime
from pdf2image import convert_from_path
import pytesseract
from difflib import get_close_matches

# -------- Cross-platform CONFIG --------
system_os = platform.system()

# --- Detect Tesseract ---
tess_cmd = shutil.which("tesseract")
if not tess_cmd:
    if system_os == "Windows":
        cand = r"C:\Tesseract-OCR\tesseract.exe"
        if os.path.exists(cand):
            tess_cmd = cand
    elif system_os == "Darwin":  # macOS
        for cand in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"):
            if os.path.exists(cand):
                tess_cmd = cand
                break
    else:  # Linux
        for cand in ("/usr/bin/tesseract", "/usr/local/bin/tesseract"):
            if os.path.exists(cand):
                tess_cmd = cand
                break

if not tess_cmd:
    raise RuntimeError("❌ Tesseract not found. Install and update PATH.")
pytesseract.pytesseract.tesseract_cmd = tess_cmd

# --- Detect Poppler (Windows only) ---
poppler_path = None
if system_os == "Windows":
    poppler_path = os.environ.get("POPPLER_PATH", r"C:\poppler-25.07.0\Library\bin")
    if not os.path.isdir(poppler_path):
        raise RuntimeError("❌ Poppler not found. Install Poppler and set POPPLER_PATH to its ...\\Library\\bin folder.")

# -------- Helpers --------
def normalize_for_dates(s: str) -> str:
    trans = str.maketrans({
        'O': '0', 'o': '0',
        'I': '1', 'l': '1', '|': '1',
        'S': '5', 's': '5',
        'B': '8',
        '—': '-', '–': '-', '‚': ','
    })
    return s.translate(trans)


def try_parse_date(s: str):
    s = s.strip()
    s = re.sub(r'\s*([./-])\s*', r'\1', s)
    formats = ["%d.%m.%Y", "%d-%m-%Y", "%d/%m/%Y",
               "%d.%m.%y", "%d-%m-%y", "%d/%m/%y"]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def extract_invoice_date(full_text: str, invoice_no: str = None):
    norm = normalize_for_dates(full_text)
    date_pattern = re.compile(r'([0-3]?\d[./-][01]?\d[./-]\d{2,4})')

    for line in norm.splitlines():
        if re.search(r'\b(date|invoice|d\.?o\.?|d\.?o\.?no)\b', line, re.IGNORECASE):
            m = date_pattern.search(line)
            if m:
                dt = try_parse_date(m.group(1))
                if dt:
                    return dt.strftime("%d.%m.%Y")

    for m in date_pattern.finditer(norm):
        dt = try_parse_date(m.group(1))
        if dt:
            return dt.strftime("%d.%m.%Y")

    return None


def find_value(pattern: str, text: str):
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def extract_lr(text: str):
    m = re.search(r'(?:L\.?R\.?|R\.?R\.?)\s*No[:.\s/\\-]*\s*([0-9]{2,8})', text, re.IGNORECASE)
    if m:
        return m.group(1)
    nums = re.findall(r'\b([0-9]{3,6})\b', text)
    return nums[0] if nums else None


def _clean_text(tok: str) -> str:
    if not tok:
        return None
    tok = re.sub(r'[^A-Za-z\s]', ' ', tok).strip()
    return re.sub(r'\s+', ' ', tok).upper()


# -------- Main Extraction --------
def run_extraction(pdf_path: str) -> dict:
    if poppler_path:
        images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    else:
        images = convert_from_path(pdf_path, dpi=300)

    text = ""
    for page in images:
        text += pytesseract.image_to_string(page) + "\n"

    invoice_no_found = find_value(r'Invoice\s*No[^0-9]*([0-9]{5,})', text)

    raw_data = {
        "Consignment No": lr_no,
        "Source": source_city,
        "Destination": destination,
        "E-Way Bill No": eway_no,
        "E-Way Bill Date": invoice_date,
        "E-Way Bill Valid Upto": eway_expiry,
        "Consignor": consignor,
        "Consignee": consignee,
        "Billing Party": consignee,
        "Delivery Address": delivery_address,
        "Vehicle": vehicle_no,
        "Date (ERP entry date)": invoice_date,
        "Invoice No": invoice_no,
        "Invoice Date": invoice_date,
        "Content Name (Goods Name)": content_name,
        "Actual Weight": weight,
        "E-Way Bill No (Goods)": eway_no,  # same value on purpose
        "Rate": rate,
    }

    return raw_data
