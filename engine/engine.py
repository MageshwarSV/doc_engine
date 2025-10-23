# engine.py
from engine.extractors import client1_format1

# Map client_id + format_id → extractor function
EXTRACTOR_MAP = {
    ("1", "1"): client1_format1.run,
}


def extract_document(client_id: str, format_id: str, pdf_path: str):
    """
    Normal service entrypoint:
    - Takes client_id, format_id, pdf_path
    - Uses EXTRACTOR_MAP
    - Returns raw_data only (transform handled elsewhere)
    """
    key = (client_id, format_id)
    if key not in EXTRACTOR_MAP:
        raise ValueError(f"No extractor found for client {client_id}, format {format_id}")

    raw_data = EXTRACTOR_MAP[key](pdf_path)

    return {
        "client_id": client_id,
        "format_id": format_id,
        "extracted_data": raw_data,
    }


def extract_direct_adapter(pdf_path: str,
                           tesseract_cmd: str = None,
                           poppler_path: str = None,
                           save_raw_path: str = None):
    """
    Direct-Adapter mode (DA):
    - Call extractor directly with extra params.
    - Useful for debugging outside of the client_id/format_id pipeline.
    """
    raw = client1_format1.run(
        pdf_path,
        tesseract_cmd=tesseract_cmd,
        poppler_path=poppler_path,
        save_raw_path=save_raw_path,
    )
    return {
        "client_id": "1",
        "format_id": "1",
        "extracted_data": raw,
    }
