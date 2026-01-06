from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi import BackgroundTasks
import shutil

import json
import tempfile
import zipfile
import pandas as pd
import io
from typing import List

from app.schemas import TextRequest, PredictResponse
from app.model import predict, predict_batch  # batch inference

app = FastAPI(
    title="Customer Support Complaint Analyzer",
    version="1.0"
)

# ----------------------------
# Single text prediction
# ----------------------------
@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(request: TextRequest):
    return predict(request.text)


# ----------------------------
# Frontend
# ----------------------------
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    with open("app/static/index.html", encoding="utf-8") as f:
        return f.read()


# ----------------------------
# Bulk complaint analysis
# ----------------------------
@app.post("/analyze-complaints")
async def analyze_complaints(  background_tasks: BackgroundTasks,
    file: UploadFile = File(...)):
    contents = await file.read()

    # ---- Parse input file ----
    try:
        if file.filename.endswith(".json"):
            complaints = json.loads(contents)

        elif file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))

            possible_cols = ["message", "text", "complaint", "description"]
            col = next((c for c in possible_cols if c in df.columns), None)

            if not col:
                raise HTTPException(
                    status_code=400,
                    detail="CSV must contain a complaint text column"
                )

            df = df.rename(columns={col: "message"})
            complaints = df.to_dict(orient="records")

        else:
            raise HTTPException(
                status_code=400,
                detail="Only JSON or CSV files are supported"
            )

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")

    if not isinstance(complaints, list):
        raise HTTPException(
            status_code=400,
            detail="Input file must contain a list of complaints"
        )

    if len(complaints) == 0:
        raise HTTPException(status_code=400, detail="No complaints found")

    if len(complaints) > 20000:
        raise HTTPException(
            status_code=400,
            detail="File too large (max 20,000 complaints)"
        )

    # ---- Clean + collect messages ----
    cleaned_items = []
    messages: List[str] = []

    for item in complaints:
        msg = item.get("message", "")
        if not isinstance(msg, str) or not msg.strip():
            continue

        msg = msg.strip()
        if len(msg) > 500:
            msg = msg[:500]

        cleaned_items.append(item)
        messages.append(msg)

    if not messages:
        raise HTTPException(
            status_code=400,
            detail="No valid complaint messages found"
        )

    # ---- Batch inference ----
    predictions = predict_batch(messages)

    high_priority = []
    low_priority = []

    for item, message, result in zip(cleaned_items, messages, predictions):
        enriched = {
            "id": item.get("id"),
            "message": message,
            **result
        }

        if result["priority"] == "high":
            high_priority.append(enriched)
        else:
            low_priority.append(enriched)

    # ---- Summary ----
    summary = {
        "total_complaints": len(high_priority) + len(low_priority),
        "high_priority": len(high_priority),
        "low_priority": len(low_priority)
    }

 # ---- Create temp directory ----
    temp_dir = tempfile.mkdtemp()

    high_path = f"{temp_dir}/high_priority_complaints.json"
    low_path = f"{temp_dir}/low_priority_complaints.json"
    summary_path = f"{temp_dir}/summary.json"
    zip_path = f"{temp_dir}/sorted_complaints.zip"

    # ---- ALWAYS write files (even if empty) ----
    with open(high_path, "w", encoding="utf-8") as f:
        json.dump(high_priority or [], f, indent=2)

    with open(low_path, "w", encoding="utf-8") as f:
        json.dump(low_priority or [], f, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ---- Zip them ----
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(high_path, arcname="high_priority_complaints.json")
        zipf.write(low_path, arcname="low_priority_complaints.json")
        zipf.write(summary_path, arcname="summary.json")

    # ---- Cleanup AFTER response ----
    background_tasks.add_task(shutil.rmtree, temp_dir)

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename="sorted_complaints.zip"
    )



# ----------------------------
# Health check
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
