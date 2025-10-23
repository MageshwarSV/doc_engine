# engine/extractors/client1_format1.py
# -------------------------------------------------------------
# Client 1 - Format 1 OCR Extractor (fast + robust) — Option B
# Returns ONLY raw_data (dict). No final mapping here.
# -------------------------------------------------------------

import os
import re
import json
import shutil
import logging
import platform
from datetime import datetime
from collections import defaultdict
from difflib import get_close_matches
from typing import Dict, Any, Tuple, Optional, List

import pytesseract
from pdf2image import convert_from_path
from pytesseract import Output
from PIL import Image, ImageOps, ImageFilter

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# -------------------------
# Cross-platform: Tesseract / Poppler
# -------------------------
def _detect_tesseract(explicit_cmd: Optional[str] = None) -> str:
    if explicit_cmd and os.path.exists(explicit_cmd):
        pytesseract.pytesseract.tesseract_cmd = explicit_cmd
        return explicit_cmd

    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd and os.path.exists(env_cmd):
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return env_cmd

    found = shutil.which("tesseract")
    if found:
        pytesseract.pytesseract.tesseract_cmd = found
        return found

    system = platform.system()
    if system == "Windows":
        cands = [
            r"C:\Tesseract-OCR\tesseract.exe",
        ]
    elif system == "Darwin":
        cands = ["/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract", "/opt/local/bin/tesseract"]
    else:
        cands = ["/usr/bin/tesseract", "/usr/local/bin/tesseract"]

    for c in cands:
        if os.path.exists(c):
            pytesseract.pytesseract.tesseract_cmd = c
            return c

    raise RuntimeError(
        "Tesseract not found. Install it or set TESSERACT_CMD or pass tesseract_cmd to run()."
    )

def _detect_poppler(explicit_path: Optional[str] = None) -> Optional[str]:
    if platform.system() != "Windows":
        return None
    if explicit_path and os.path.isdir(explicit_path):
        return explicit_path
    env_p = os.getenv("POPPLER_PATH")
    if env_p and os.path.isdir(env_p):
        return env_p
    guesses = [
        r"C:\poppler-25.07.0\Library\bin",
        r"C:\Users\Public\poppler\bin",
    ]
    for g in guesses:
        if os.path.isdir(g):
            return g
    return None

# -------------------------
# OCR helpers (fast)
# -------------------------
def _rotate_upright(img: Image.Image) -> Image.Image:
    try:
        osd = pytesseract.image_to_osd(img)
        for ln in osd.splitlines():
            if "Rotate:" in ln:
                deg = int(ln.split(":")[1].strip())
                if deg != 0:
                    return img.rotate(360 - deg, expand=True)
                break
    except Exception:
        pass
    return img

def _preprocess_variants_fast(img: Image.Image) -> List[Image.Image]:
    img = _rotate_upright(img)
    g = ImageOps.autocontrast(img.convert("L"))
    sharp = g.filter(ImageFilter.UnsharpMask(radius=1.4, percent=140, threshold=3))
    return [g, sharp]

def _ocr_configs_fast(img: Image.Image) -> List[str]:
    cfgs = [
        r"--oem 1 --psm 6 -c preserve_interword_spaces=1",
        r"--oem 1 --psm 4 -c preserve_interword_spaces=1",
    ]
    outs: List[str] = []
    for cfg in cfgs:
        try:
            txt = pytesseract.image_to_string(img, config=cfg, lang="eng") or ""
            if txt.strip():
                outs.append(txt)
        except Exception:
            continue
    return outs

def _merge_text(blocks: List[str]) -> str:
    seen = set()
    merged = []
    for block in blocks:
        for ln in (block or "").splitlines():
            s = (ln or "").rstrip()
            if not s:
                continue
            key = s.strip()
            if key in seen:
                continue
            seen.add(key)
            merged.append(s)
    return "\n".join(merged)

# -------------------------
# Generic helpers
# -------------------------
def normalize_for_dates(s: str) -> str:
    trans = str.maketrans({
        'O': '0','o': '0','I': '1','l': '1','|': '1','S': '5','s': '5','B': '8','—': '-','–': '-','‚': ','
    })
    return s.translate(trans)

def try_parse_date(s: str):
    s = re.sub(r'\s*([./-])\s*', r'\1', (s or "").strip())
    for fmt in ["%d.%m.%Y","%d-%m-%Y","%d/%m/%Y","%d.%m.%y","%d-%m-%y","%d/%m/%y"]:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None

def extract_invoice_date(full_text: str, invoice_no: str = None) -> Optional[str]:
    norm = normalize_for_dates(full_text)
    pat = re.compile(r'([0-3]?\d[./-][01]?\d[./-]\d{2,4})')
    for line in norm.splitlines():
        if re.search(r'\b(date|invoice|d\.?o\.?|d\.?o\.?no)\b', line, re.IGNORECASE):
            m = pat.search(line)
            if m:
                dt = try_parse_date(m.group(1))
                if dt:
                    return dt.strftime("%d.%m.%Y")
    for m in pat.finditer(norm):
        dt = try_parse_date(m.group(1))
        if dt:
            return dt.strftime("%d.%m.%Y")
    frag = re.search(r'([./-][01]?\d[./-]\d{4})', norm)
    if frag:
        trial = ("01." if '.' in frag.group(1) else "01-") + frag.group(1).lstrip('./-')
        dt = try_parse_date(trial)
        if dt:
            return dt.strftime("%d.%m.%Y")
    return None

def find_value(pattern: str, text: str, flags=re.IGNORECASE | re.DOTALL) -> Optional[str]:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None

# -------------------------
# Field extractors
# -------------------------
def extract_invoice_no(text: str) -> Optional[str]:
    U = (text or "").upper()
    def keep_digits(s: str) -> Optional[str]:
        if not s:
            return None
        nums = re.findall(r'\d{6,}', s)
        if not nums:
            nums = re.findall(r'(?:\d\s+){5,}\d', s)
            if nums:
                return re.sub(r'\s+', '', nums[0])
            return None
        nums.sort(key=len, reverse=True)
        return nums[0]

    patterns = [
        r'D[.\s]*[I1L][.\s]*(?:[IL1][.\s]*)?N[O0][.\s]*(?:&\s*DATE)?[:\-\s]*([0-9\s]{6,})',
        r'\bD[.\s]*I[.\s]*N[O0][.\s]*[:\-\s]*([0-9\s]{6,})',
        r'\bINV(?:OICE)?\s*NO[.\s:]*([A-Z0-9\-/\s]{6,})',
        r'Invoice\s*No[^0-9]*([0-9]{5,})',
    ]
    for pat in patterns:
        m = re.search(pat, U, re.IGNORECASE | re.DOTALL)
        if m:
            val = keep_digits(m.group(1))
            if val:
                return val

    m = re.search(r'D[.\s]*[I1L][.\s]*(?:[IL1][.\s]*)?N[O0].{0,30}?([0-9]{6,})', U, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1)
    return None

def extract_lr(text: str) -> Optional[str]:
    m = re.search(r'L\.?R\.?.*RR\s*No[:.\s/\\-]*([0-9]{2,8})', text, re.IGNORECASE)
    if m: return m.group(1)
    m = re.search(r'(?:L\.?R\.?\.?|L\.?R\.?)\s*No[:.\s/\\-]*([0-9]{2,8})', text, re.IGNORECASE)
    if m: return m.group(1)
    m = re.search(r'RR\s*No[:.\s/\\-]*([0-9]{2,8})', text, re.IGNORECASE)
    if m: return m.group(1)
    m = re.search(r'\bLR[:.\s-]*([0-9]{2,8})', text, re.IGNORECASE)
    if m: return m.group(1)
    return None

def extract_irn(text: str) -> Optional[str]:
    return find_value(r'IRN[:.\s]*([a-zA-Z0-9]+)', text)

def extract_driver_mobile(text: str) -> Optional[str]:
    return find_value(r'\b([6-9][0-9]{9})\b', text, flags=re.MULTILINE)

def extract_consignee(text: str) -> Optional[str]:
    header_re = re.compile(r'(?:Name\s*(?:&|and)\s*Address\s*of\s*(?:Recipient|Consignee))\s*:?', re.IGNORECASE)
    stop_tokens = re.compile(r'\b(?:GSTIN|GST|STATE|PLACE\s+OF\s+SUPPLY|PO\s*NO(?:/DATE)?|DATE|EMAIL|MOBILE|PHONE|PAN)\b', re.IGNORECASE)
    suffix_pat = re.compile(r'^(.*?\b(?:PRIVATE\s+LIMITED|PVT\.?\s*LTD|LIMITED|LTD|LLP|LLC|INC|CO\.?))\b', re.IGNORECASE)
    lines = (text or "").splitlines()

    def _is_company_line(s: str) -> bool:
        if not s: return False
        U = s.upper().strip()
        if stop_tokens.search(U): return False
        if any(k in U for k in ["RECIPIENT","ADDRESS","NAME","CONSIGNEE"]): return False
        if re.search(r'\b(PVT|PRIVATE|LTD|LIMITED|LLP|COMPANY|TRADING|CEMENT|ENGINEERS?)\b', U): return True
        if U == s.strip() and 1 <= len(U.split()) <= 4 and len(U) >= 4: return True
        letters = re.findall(r'[A-Za-z]', s)
        return bool(letters) and (sum(ch.isalpha() for ch in letters) / len(letters) >= 0.7)

    def _clean_company_line(raw: str) -> Optional[str]:
        s = (raw or "").strip(" :-\t")
        m_stop = stop_tokens.search(s)
        if m_stop: s = s[:m_stop.start()].rstrip()
        m = suffix_pat.search(s)
        if m: s = m.group(1)
        s = re.split(r'\s+\d[\d\s/\-\.]*', s, maxsplit=1)[0]
        s = re.sub(r'\s{2,}', ' ', s).strip(' ,:-.')
        return s or None

    for idx, line in enumerate(lines):
        m = header_re.search(line)
        if not m: continue
        tail = (line[m.end():] or "").strip(" :-")
        if tail and _is_company_line(tail):
            return _clean_company_line(tail)
        for j in range(idx + 1, min(idx + 10, len(lines))):
            cand = (lines[j] or "").strip()
            if not cand: continue
            if _is_company_line(cand):
                return _clean_company_line(cand)
        break
    return None

SEP = r'[\s\W]*'
CODE_PATTERNS = [
    ("OPC53", [re.compile(rf'O{SEP}P{SEP}C{SEP}[5S]{SEP}3\b', re.IGNORECASE), re.compile(rf'OPC{SEP}[5S]{SEP}3\b', re.IGNORECASE), re.compile(rf'[5S]{SEP}3{SEP}OPC\b', re.IGNORECASE)]),
    ("OPC43", [re.compile(rf'O{SEP}P{SEP}C{SEP}4{SEP}3\b', re.IGNORECASE), re.compile(rf'OPC{SEP}4{SEP}3\b', re.IGNORECASE), re.compile(rf'4{SEP}3{SEP}OPC\b', re.IGNORECASE)]),
    ("PPC53", [re.compile(rf'P{SEP}P{SEP}C{SEP}[5S]{SEP}3\b', re.IGNORECASE), re.compile(rf'PPC{SEP}[5S]{SEP}3\b', re.IGNORECASE), re.compile(rf'[5S]{SEP}3{SEP}PPC\b', re.IGNORECASE)]),
    ("PPC43", [re.compile(rf'P{SEP}P{SEP}C{SEP}4{SEP}3\b', re.IGNORECASE), re.compile(rf'PPC{SEP}4{SEP}3\b', re.IGNORECASE), re.compile(rf'4{SEP}3{SEP}PPC\b', re.IGNORECASE)]),
    ("OPC",  [re.compile(rf'O{SEP}P{SEP}C\b', re.IGNORECASE), re.compile(r'\bOPC\b', re.IGNORECASE)]),
    ("PPC",  [re.compile(rf'P{SEP}P{SEP}C\b', re.IGNORECASE), re.compile(r'\bPPC\b', re.IGNORECASE)]),
    ("PSC",  [re.compile(rf'P{SEP}S{SEP}C\b', re.IGNORECASE), re.compile(r'\bPSC\b', re.IGNORECASE)]),
    ("SRC",  [re.compile(rf'S{SEP}R{SEP}C\b', re.IGNORECASE), re.compile(r'\bSRC\b', re.IGNORECASE)]),
]

def _merge_split_letters_in_text(t: str) -> str:
    t = re.sub(r'(?i)\b([A-Z])\s+([A-Z])\s+([A-Z])\b', r'\1\2\3', t)
    t = re.sub(r'(?i)\b([A-Z])\s+([A-Z])\b', r'\1\2', t)
    return t

def extract_material_code_global(txt: str) -> Tuple[Optional[str], Optional[str]]:
    U = txt.upper().replace('0','O').replace('1','I').replace('5','S')
    U = _merge_split_letters_in_text(U)
    best = None
    for code, patterns in CODE_PATTERNS:
        for pat in patterns:
            m = pat.search(U)
            if m:
                idx = m.start()
                if (best is None) or (idx < best[0]):
                    best = (idx, code)
                break
    if best:
        return best[1], "global_regex"
    squashed = re.sub(r'[^A-Z0-9]+', '', U)
    for code in ["OPC53","PPC53","OPC43","PPC43","OPC","PPC","PSC","SRC"]:
        if code in squashed:
            return code, "global_squash"
    return None, None

def extract_quantity_weight(images: List[Image.Image], text: str) -> Tuple[Optional[str], Optional[str]]:
    num_pat = re.compile(r'\b\d+\.\d{3}\b')
    for page in images:
        data = pytesseract.image_to_data(page, output_type=Output.DICT, config="--psm 6")
        n = len(data['text'])
        lines = defaultdict(list)
        for i in range(n):
            txt = (data['text'][i] or '').strip()
            if not txt: continue
            key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
            l = data['left'][i]; lines[key].append((txt, l))
        for key, wlist in lines.items():
            line_txt = ' '.join([w[0] for w in sorted(wlist, key=lambda z: z[1])])
            if re.search(r'\b(Quantity|Qty)\b', line_txt, re.IGNORECASE):
                (bn, pn, ln) = key
                for offset in range(1, 7):
                    key2 = (bn, pn, ln + offset)
                    if key2 not in lines: continue
                    row_str = ' '.join([w[0] for w in sorted(lines[key2], key=lambda z: z[1])])
                    if '%' in row_str: continue
                    m = num_pat.search(row_str)
                    if m: return m.group(0), "quantity_ocr"
    for m in re.finditer(r'\b\d+\.\d{3}\b', text):
        if '%' not in m.group(0): return m.group(0), "quantity_global"
    return None, None

def _normalize_money(s: str) -> Optional[str]:
    if not s: return None
    s = s.replace(',', '')
    m = re.search(r'\d+(?:\.\d+)?', s)
    if not m: return None
    return f"{float(m.group(0)):.2f}"

def extract_rate(text: str) -> Optional[str]:
    UNIT = r'(?:MT|M\.?T\.?|TON|TONNE|T|BAG|BAGS|KG|KGS|QUINTAL|QTL|METRIC\s*TON)'
    CURR = r'(?:Rs\.?|₹|INR)\s*'
    AMT  = r'([0-9]{1,3}(?:,[0-9]{2,3})*(?:\.\d+)?|[0-9]+(?:\.\d+)?)'
    patterns = [
        rf'@\s*{CURR}?{AMT}\s*(?:Per|/)\s*{UNIT}\b',
        rf'(?:Freight|Rate|Price)[^.\n\r]{{0,120}}{CURR}?{AMT}\s*(?:Per|/)\s*{UNIT}\b',
        rf'{CURR}{AMT}\s*(?:Per|/)\s*{UNIT}\b',
        rf'(?:Per|/)\s*{UNIT}\s*[:\-]?\s*{CURR}{AMT}\b',
        rf'\bRate\s*(?:Per|/)\s*{UNIT}\s*[:\-]?\s*{CURR}?{AMT}\b',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            return _normalize_money(m.group(1))
    return None

def _collapse_repeated_words(s: str) -> str:
    s = re.sub(r'([A-Z]{3,})(?:\W*\1)+', r'\1', s, flags=re.IGNORECASE)
    s = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', s, flags=re.IGNORECASE)
    return s

# -------------------------
# Goods type extractor (ADDED)
# -------------------------
_RE_PRODUCT_ANCHOR = re.compile(r'\b(?:OPC|PPC|PSC|SRC|CEMENT)\b(?:\s*[45]3)?', re.IGNORECASE)
_RE_WEIGHT_HINT    = re.compile(r'\b(?:25|35|40|50)\s*KGS?\b|\bKGS?\b|\bKG\b', re.IGNORECASE)
_RE_BAG   = re.compile(r'\bBAGS?\b', re.IGNORECASE)
_RE_BULK  = re.compile(r'\bBULK\b', re.IGNORECASE)
_RE_LOOSE = re.compile(r'\bLOOSE\b', re.IGNORECASE)
_RE_PAPER = re.compile(r'\bPAPER\b', re.IGNORECASE)

def _collect_product_blocks(text: str, window: int = 6) -> List[str]:
    if not text:
        return []
    lines = [ln.strip() for ln in (text or "").splitlines()]
    idxs = []
    for i, ln in enumerate(lines):
        if not ln:
            continue
        if _RE_PRODUCT_ANCHOR.search(ln) or _RE_WEIGHT_HINT.search(ln):
            idxs.append(i)
    blocks, seen = [], set()
    for i in idxs:
        lo, hi = max(0, i - window), min(len(lines), i + window + 1)
        block = "\n".join(l for l in lines[lo:hi] if l)
        u = block.upper()
        if u and u not in seen:
            seen.add(u)
            blocks.append(block)
    return blocks

def extract_goods_type(text: str) -> Optional[str]:
    """
    Return one of: 'PAPER', 'BAG', 'BULK' or None
    Heuristic: look for product anchors / weight / bag / bulk tokens in nearby blocks.
    """
    for block in _collect_product_blocks(text, window=6):
        U = block.upper()
        if _RE_PAPER.search(U):
            return "PAPER"
        if _RE_BAG.search(U):
            return "BAG"
        if _RE_BULK.search(U) or _RE_LOOSE.search(U):
            return "BULK"
    U = (text or "").upper()
    if _RE_PAPER.search(U):
        return "PAPER"
    if _RE_BAG.search(U):
        return "BAG"
    if _RE_BULK.search(U) or _RE_LOOSE.search(U):
        return "BULK"
    return None

def extract_delivery_address_old_style(text: str) -> Optional[str]:
    delivery_match = re.search(
        r'(?:Name\s*&\s*Address\s*of\s*(?:Recipient|Delivery|Ship\s*To)\s*:?\s*)([\s\S]{0,400}?)' +
        r'(?:\n\s*(?:Place\s*of\s*Supply|State|State\s*Code|GSTIN|Phone|Mobile|Recipient\s*PO|Invoice|Amount|EWB|IRN|PAN)\b)',
        text, re.IGNORECASE
    ) or re.search(
        r'(?:Name\s*&\s*Address\s*of\s*(?:Recipient|Delivery|Ship\s*To)\s*:?\s*)([\s\S]{0,400})',
        text, re.IGNORECASE
    )

    if not delivery_match:
        return None

    block = delivery_match.group(1) or ""
    block = re.sub(r'^(Name\s*&\s*Address\s*of[^\n]*\n?)', '', block, flags=re.IGNORECASE)
    block = re.split(r'\b(?:Phone|Tel|Contact|Mobile|GSTIN|PAN|Invoice|Amount)\b', block, 1, flags=re.IGNORECASE)[0]
    block = block.replace('|', ' ').replace('¦', ' ').replace('•', ' ')
    lines = [ln.strip(" ,:-") for ln in block.splitlines() if ln.strip()]

    cleaned_parts = []
    for ln in lines:
        ln = re.sub(r'\bRecipient\s*Code\b.*', '', ln, flags=re.IGNORECASE)
        ln = re.sub(r'\bRecipient\s*PO\b.*', '', ln, flags=re.IGNORECASE)
        ln = re.sub(r'^(Name\s*&\s*Address.*)', '', ln, flags=re.IGNORECASE).strip()
        if not ln:
            continue
        if re.search(r'\b(PVT|PRIVATE|LTD|LIMITED|CO|COMPANY|ULTRATECH|ENGINEER|ENGINEERS|PVT\.? LTD|TRADING|TRADERS)\b', ln, re.IGNORECASE):
            tail = re.split(r'\b(?:PVT\.?\s*LTD|PRIVATE\s+LIMITED|PVT\s+LTD|LTD|LIMITED|ENGINEERS?|TRADING|TRADERS|COMPANY|CO\.?)\b', ln, flags=re.IGNORECASE)[-1]
            tail = tail.strip(" ,:-")
            if tail:
                ln = tail
            else:
                continue
        ln = re.sub(r'\bNO\.?\s*(?=\d)', '', ln, flags=re.IGNORECASE)
        ln = re.sub(r'\s*,\s*', ', ', ln)
        ln = re.sub(r'\s{2,}', ' ', ln).strip(' ,:-')
        if ln:
            cleaned_parts.append(ln)

    if not cleaned_parts:
        return None

    addr_keywords = re.compile(r'\b(RS\s*NO|RS\b|NO\b|ROAD|STREET|STATION|NAGAR|VILLAGE|LANE|TOWN|CITY|DIST|MAIN|COLONY|COMPLEX|BUILDING|SURVEY|FLAT)\b', re.IGNORECASE)
    state_kw = re.compile(r'\b(TAMIL\s*NADU|KARNATAKA|KERALA|ANDHRA|TELANGANA|MAHARASHTRA|GUJARAT|DELHI|PUDUCHERRY)\b', re.IGNORECASE)

    candidates = []
    for cp in cleaned_parts:
        if len(cp) < 3:
            continue
        if addr_keywords.search(cp) or re.search(r'\d', cp) or state_kw.search(cp):
            candidates.append(cp)
    if not candidates:
        candidates = sorted(cleaned_parts, key=lambda s: len(s), reverse=True)[:3]

    final_parts = []
    seen = set()
    for p in candidates:
        p2 = _collapse_repeated_words(p)
        p2 = re.split(r'\b(Phone|Tel|Contact|Mobile|GSTIN|PAN|Invoice|Amount)\b', p2, 1, flags=re.IGNORECASE)[0]
        p2 = re.sub(r'[^A-Za-z0-9 ,/:-]', ' ', p2)
        p2 = re.sub(r'\s{2,}', ' ', p2).strip(' ,:-')
        if not p2:
            continue
        key = p2.upper()
        if key in seen:
            continue
        seen.add(key)
        final_parts.append(p2.upper())

    st = find_value(r'State\s*[:\-]\s*([A-Z ]+)', text) or find_value(r',\s*([A-Z ]{3,20})\s*$', ' '.join(final_parts)) or None
    if st:
        st = st.strip().upper()
        if not any(st in fp for fp in final_parts):
            final_parts.append(st)

    addr = ", ".join(final_parts)
    addr = re.sub(r',\s*,+', ', ', addr)
    addr = re.sub(r'\s{2,}', ' ', addr).strip(' ,:-')
    addr = re.sub(r'\b(Phone|Tel|Contact|Mobile|GSTIN|PAN|Invoice|Amount)\b.*$', '', addr, flags=re.IGNORECASE).strip(' ,:-')
    addr = re.sub(r'\s{2,}', ' ', addr)
    return addr if addr else None

# -------------------------
# City normalization (ARAKONAM → ARAKKONAM)
# -------------------------
_CANON_CITY_MAP = {
    "ARAKONAM": "ARAKKONAM",
    "ARAKKONAM": "ARAKKONAM",
    # add any common OCR misspellings if you like:
    "ARRAKONAM": "ARAKKONAM",
    "ARAKONNAM": "ARAKKONAM",
}

def _canon_city(name: Optional[str]) -> Optional[str]:
    if not name:
        return name
    up = (name or "").strip().upper()
    return _CANON_CITY_MAP.get(up, up)

# -------------------------
# City extractors (with normalization)
# -------------------------
def extract_despatch_city(text: str) -> Optional[str]:
    raw = None
    for p in (r'Despatch\s*From[:.\s]*([^\n\r]+)',
              r'Despatched?\s*From[:.\s]*([^\n\r]+)',
              r'Despatch[:.\s]*([^\n\r]+)'):
        m = re.search(p, text, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            break
    if not raw:
        return None

    raw = re.sub(r'\b(UTCL|ULTRATECH|CEMENT|WORKS|UNIT|DEPOT|PLANT|CODE|MA|FACILITY)\b', ' ', raw, flags=re.IGNORECASE)
    toks = [t.strip() for t in re.split(r'[,\-/ ]+', raw) if t.strip()]

    # include both variants so get_close_matches/direct in checks work
    CANONICAL = [
        "ARAKONAM","ARAKKONAM","CHROMPET","CHENNAI",
        "TIRUNANIMALI","TIRUNINRAVELI","VELLORE","MADURAI",
        "POONAMALLEE","SRIPERUMBUDUR","SRIERUMBUDUR"
    ]
    up = [t.upper() for t in toks]

    # direct token match
    for c in CANONICAL:
        if c in up:
            return _canon_city(c)

    # fallback: pick last alpha token
    for t in reversed(up):
        if len(t) >= 3 and t.isalpha():
            return _canon_city(t)

    return None

def extract_destination(text: str) -> Optional[str]:
    m = re.search(r'Destination\s*[:\-]\s*([^\n\r]+)', text, re.IGNORECASE)
    if m:
        cand = m.group(1)
    else:
        cand = None
        for line in (text or "").splitlines():
            if 'destination' in line.lower():
                cand = line.split(':', 1)[-1].strip()
                break
    if not cand:
        return None

    cand = re.split(r'\b(?:DESPATCH|DISPATCH|FROM|BOOKING|STATION|COMMERCIAL|TERMS)\b', cand, 1, flags=re.IGNORECASE)[0]
    cand = cand.split('(')[0].strip(" ,:-")
    cand = re.sub(r'[^A-Za-z\s]', ' ', cand)
    cand = re.sub(r'\s+', ' ', cand).strip()

    CANONICAL = [
        "ARAKONAM","ARAKKONAM","CHROMPET","CHENNAI",
        "TIRUNINRAVELI","VELLORE","MADURAI","POONAMALLEE",
        "SRIPERUMBUDUR","SRIERUMBUDUR"
    ]

    upcand = cand.upper()

    # direct contains
    for c in CANONICAL:
        if c in upcand:
            return _canon_city(c)

    # fuzzy
    lst = get_close_matches(upcand, CANONICAL, n=1, cutoff=0.6)
    if lst:
        return _canon_city(lst[0])

    # else just uppercase candidate, then canonicalize if needed
    return _canon_city(upcand) if cand else None

def extract_eway_bill(text: str) -> Optional[str]:
    pat = re.compile(r'(?:E[\s\-]*W(?:AY|B)?\s*(?:BILL)?\s*(?:NO|NUMBER|#)?\s*[:\-]?\s*)([0-9\s]{10,25})', re.IGNORECASE)
    m = pat.search(text or "")
    if m:
        num = re.sub(r'\s+', '', m.group(1))
        if 10 <= len(num) <= 20:
            return num
    m = re.search(r'\b([0-9]{12,16})\b', text or "")
    if m:
        return m.group(1)
    return None

# -------------------------
# NEW: Delivery address cleaner
# -------------------------
def clean_delivery_address(addr: str) -> str:
    """
    Clean delivery address:
    - Keep from 'NO:' onward if present
    - Remove RS/SURVEY blocks, PROJECT prefixes, agency prefixes, 'RECIPIENT CODE/CADE ...'
    - Remove leading numeric tokens like '1134942973/'; normalize DNO etc.
    - Deduplicate comma-separated segments using a canonical key
    """
    if not addr:
        return addr

    # If NO: is present, keep from there onward
    m = re.search(r'\bNO[:\-]?\s*.*', addr, re.IGNORECASE)
    if m:
        addr = m.group(0)

    # Remove RS/SURVEY chunks
    addr = re.sub(r'\bRS\s*[0-9A-Z\s/:-]+,?\s*', '', addr, flags=re.IGNORECASE)
    addr = re.sub(r'\bSURVEY\s*NO\.?\s*[0-9A-Z\s/:-]+,?\s*', '', addr, flags=re.IGNORECASE)

    # Remove PROJECT prefixes like "CPRR PROJECTS -"
    addr = re.sub(r'\b[A-Z ]*PROJECTS?\s*-\s*', '', addr, flags=re.IGNORECASE)

    # Remove PRIVATE L 113603... prefix fragments
    addr = re.sub(r'\bPRIVATE\s+L\b[^,]*,?', '', addr, flags=re.IGNORECASE)

    # Remove RECIPIENT CODE/CADE blocks
    addr = re.sub(r'\bRECIPIENT\s*(?:CODE|CADE)\b[^,]*,?', '', addr, flags=re.IGNORECASE)

    # Remove leading "FOR ... PROJECT,"
    addr = re.sub(r'^\s*FOR\s+.*?PROJECT,?\s*', '', addr, flags=re.IGNORECASE)

    # Remove leading agency info like "SRY AGENCIES 1134942973/"
    addr = re.sub(r'^[A-Z0-9\s]+AGENCIES[^\w]+', '', addr, flags=re.IGNORECASE)

    # Remove leading numeric codes like "1134942973/"
    addr = re.sub(r'^[0-9]{5,}/\s*', '', addr)

    # Normalize DNO -> D NO
    addr = re.sub(r'\bDNO\s*', 'D NO ', addr, flags=re.IGNORECASE)
    # Normalize 'D <digits>' -> 'D NO <digits>' (but do NOT touch existing 'D NO')
    addr = re.sub(r'\bD\s+(?=\d)', 'D NO ', addr, flags=re.IGNORECASE)

    # Split by commas and deduplicate with a canonical key
    parts = [p.strip() for p in addr.split(',') if p.strip()]

    def _canon(seg: str) -> str:
        s = seg.upper()
        # unify spacing and remove non-alphanum (keep spaces)
        s = re.sub(r'\bD\s+NO\b', 'D NO', s)
        s = re.sub(r'[^A-Z0-9 ]+', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    seen = set()
    unique_parts = []
    for p in parts:
        key = _canon(p)
        if key not in seen:
            seen.add(key)
            unique_parts.append(p)

    addr = ', '.join(unique_parts)

    # Final tidy
    addr = re.sub(r'\s{2,}', ' ', addr)
    addr = re.sub(r'\s*,\s*', ', ', addr)
    addr = addr.strip(" ,:-")

    return addr

# -------------------------
# Main runner (returns RAW ONLY)
# -------------------------
def run(
    pdf_path: str,
    *,
    tesseract_cmd: Optional[str] = None,
    poppler_path: Optional[str] = None,
    save_raw_path: Optional[str] = None,   # pass a path to also persist RAW if you want
) -> Dict[str, Any]:
    """
    Extract fields from a PDF using OCR; return RAW dict only.
    """
    tess = _detect_tesseract(tesseract_cmd)
    logger.info("Using Tesseract: %s", tess)
    poppler = _detect_poppler(poppler_path)

    try:
        if poppler:
            pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler)
        else:
            pages = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        if platform.system() == "Windows":
            raise RuntimeError(
                f"Failed to render PDF with pdf2image: {e}\n"
                "Install Poppler for Windows and set POPPLER_PATH to its 'bin' folder, or pass poppler_path=..."
            )
        raise

    # OCR
    all_blocks: List[str] = []
    originals: List[Image.Image] = []
    for pg in pages:
        originals.append(pg)
        variants = _preprocess_variants_fast(pg)
        page_texts: List[str] = []
        for v in variants:
            page_texts.extend(_ocr_configs_fast(v))
        all_blocks.append(_merge_text(page_texts) if page_texts else "")

    text = "\n".join(all_blocks)

    # ------- Extract fields -------
    invoice_no   = extract_invoice_no(text)
    invoice_date = extract_invoice_date(text, invoice_no)
    lr_no        = extract_lr(text)
    consignee    = extract_consignee(text)
    consignor    = "UltraTech Cement Limited" if "ULTRATECH CEMENT LIMITED" in (text or "").upper() else None
    weight, _    = extract_quantity_weight(originals, text)
    content_name, _ = extract_material_code_global(text)
    if not content_name:
        content_name = (find_value(r'Name\s*of\s*Commodity\s*[:\-]\s*([A-Za-z0-9 /-]+)', text) or
                        find_value(r'Description\s*of\s*Goods\s*[:\-]\s*([A-Za-z0-9 /-]+)', text))
    rate             = extract_rate(text)
    despatch_city    = _canon_city(extract_despatch_city(text)) or "ARAKKONAM"
    destination      = _canon_city(extract_destination(text))
    eway_no          = extract_eway_bill(text)
    eway_expiry      = (find_value(r'(?:E-?Way\s*Bill\s*(?:Validity|Valid\s*Upto|Exp(?:iry)?)\s*[:\-]?\s*)([0-9./-]+\s+[0-9:]{4,8})', text)
                        or find_value(r'(?:EWB\s*Exp\.\s*[:\-]?\s*)([0-9./-]+\s+[0-9:]{4,8})', text))
    vehicle_no       = (find_value(r'\b([A-Z]{2}\s?[0-9]{1,2}\s?[A-Z]{1,3}\s?[0-9]{3,4})\b', text)
                        or find_value(r'\b(TN[0-9A-Z]{2,}-?[0-9A-Z]{3,})\b', text))
    irn              = extract_irn(text)
    driver_mobile    = extract_driver_mobile(text)
    delivery_address = extract_delivery_address_old_style(text)
    # NEW: post-clean Delivery Address
    delivery_address = clean_delivery_address(delivery_address)
    GSTType = "Unregistered"

    # NEW: goods_type extraction (ADDED)
    goods_type = extract_goods_type(text)

    raw_data: Dict[str, Any] = {
        "Consignment No": lr_no,
        "Source": despatch_city,
        "Destination": destination,
        "E-Way Bill No": eway_no,
        "E-Way Bill Date": invoice_date,
        "E-Way Bill Valid Upto": eway_expiry,
        "Consignor": consignor,
        "Consignee": consignee,
        "Billing Party": consignee,
        "Delivery Address": delivery_address,
        "Vehicle": vehicle_no,
        "Driver Mobile": driver_mobile,
        "Date (ERP entry date)": invoice_date,
        "Invoice No": invoice_no,
        "Invoice Date": invoice_date,
        "Content Name (Goods Name)": content_name,
        "Actual Weight": weight,
        "E-Way Bill No (Goods)": eway_no,
        "Rate": rate,
        "IRN": irn,
        "GST Type": GSTType,
        "goods_type": goods_type
    }

    if save_raw_path:
        try:
            os.makedirs(os.path.dirname(save_raw_path), exist_ok=True)
            with open(save_raw_path, "w", encoding="utf-8") as f:
                json.dump(raw_data, f, indent=2, ensure_ascii=False)
            logger.info("Wrote RAW JSON: %s", save_raw_path)
        except Exception as e:
            logger.warning("Failed to write RAW JSON %s: %s", save_raw_path, e)

    return raw_data
