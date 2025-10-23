from fastapi import FastAPI, UploadFile, File, Form
import os
from engine.extractor import run_extraction

app = FastAPI()

# /extract → returns raw_data + final_data
@app.post("/extract")
async def extract_invoice(
    client_id: int ,
    format_id: int ,
    file: UploadFile = File(...)
):
    os.makedirs("uploads", exist_ok=True)

    pdf_path = os.path.join("uploads", file.filename)
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    result = run_extraction(pdf_path, client_id, format_id)
    return result


