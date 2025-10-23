import importlib
from engine.transformers import transform

def run_extraction(pdf_path: str, client_id: str, format_id: str):
    try:
        module_name = f"engine.extractors.client{client_id}_format{format_id}"
        extractor_module = importlib.import_module(module_name)

        if not hasattr(extractor_module, "run"):
            raise ImportError(f"Extractor {module_name} has no `run` function")

        raw_data = extractor_module.run(pdf_path)
        final_data = transform(raw_data, client_id, format_id)

        return {
            "client_id": client_id,
            "format_id": format_id,
            "raw_data": raw_data,
            "final_data": final_data
        }

    except Exception as e:
        return {"error": str(e)}
