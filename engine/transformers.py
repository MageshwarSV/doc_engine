import json
import os

def transform(raw_data: dict, client_id: str, format_id: str):
    """
    Map raw_data into final ERP-ready structure using mapping.json
    """
    mapping_path = f"engine/configs/client{client_id}/format{format_id}_mapping.json"

    if not os.path.exists(mapping_path):
        return {"error": f"Mapping config not found: {mapping_path}"}

    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    transformed = {}
    for final_field, raw_field in mapping.items():
        transformed[final_field] = raw_data.get(raw_field)

    return transformed
