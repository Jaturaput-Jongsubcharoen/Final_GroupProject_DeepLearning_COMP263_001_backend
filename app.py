"""
FastAPI Backend — Deep Learning for Pneumonia Detection from Chest X-Ray Images
Serves trained models, metrics, and prediction endpoints.
"""

import os
import json
import io
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results_xray")
EXP1_DIR = os.path.join(RESULTS_DIR, "exp1_custom_cnn")
EXP2_DIR = os.path.join(RESULTS_DIR, "exp2_autoencoder")

# ---------------------------------------------------------------------------
# Model registry — maps display name → .keras path + metrics source
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "Baseline CNN": {
        "path": os.path.join(EXP1_DIR, "exp1_mri_baseline_cnn.keras"),
        "metrics_file": os.path.join(EXP1_DIR, "exp1_mri_summary.json"),
        "metrics_key": "Baseline",
        "experiment": 1,
    },
    "Deep CNN": {
        "path": os.path.join(EXP1_DIR, "exp1_mri_deep_cnn.keras"),
        "metrics_file": os.path.join(EXP1_DIR, "exp1_mri_summary.json"),
        "metrics_key": "Deep",
        "experiment": 1,
    },
    "Wide CNN": {
        "path": os.path.join(EXP1_DIR, "exp1_mri_wide_cnn.keras"),
        "metrics_file": os.path.join(EXP1_DIR, "exp1_mri_summary.json"),
        "metrics_key": "Wide",
        "experiment": 1,
    },
    "Autoencoder Transfer": {
        "path": os.path.join(EXP2_DIR, "exp2_xray_transfer.keras"),
        "metrics_file": os.path.join(EXP2_DIR, "exp2_results.json"),
        "metrics_key": None,  # top-level JSON
        "experiment": 2,
    },
    "ResNet50 Transfer": {
        "path": os.path.join(RESULTS_DIR, "exp3_xray_resnet_transfer.keras"),
        "metrics_file": os.path.join(RESULTS_DIR, "exp3_xray_summary_FIXED.json"),
        "metrics_key": "transfer_learning",
        "experiment": 3,
    },
    "ResNet50 From-Scratch": {
        "path": os.path.join(RESULTS_DIR, "exp3_xray_resnet_scratch_FIXED.keras"),
        "metrics_file": os.path.join(RESULTS_DIR, "exp3_xray_summary_FIXED.json"),
        "metrics_key": "from_scratch",
        "experiment": 3,
    },
}

CLASS_NAMES = ["Normal", "Pneumonia"]
IMG_SIZE = (224, 224)

# ---------------------------------------------------------------------------
# Lazy model cache (load on first request)
# ---------------------------------------------------------------------------
_loaded_models: dict = {}


def _get_model(model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")
    if model_name not in _loaded_models:
        path = MODEL_REGISTRY[model_name]["path"]
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Model file not found: {path}")
        _loaded_models[model_name] = tf.keras.models.load_model(path)
    return _loaded_models[model_name]


def _get_metrics(model_name: str) -> dict:
    info = MODEL_REGISTRY[model_name]
    mfile = info["metrics_file"]
    if not os.path.exists(mfile):
        raise HTTPException(status_code=404, detail=f"Metrics file not found: {mfile}")
    with open(mfile, "r") as f:
        data = json.load(f)
    if info["metrics_key"] is not None:
        data = data[info["metrics_key"]]
    return {
        "accuracy": round(data["test_accuracy"], 4),
        "precision": round(data["test_precision"], 4),
        "recall": round(data["test_recall"], 4),
        "f1_score": round(data["test_f1"], 4),
    }


# ---------------------------------------------------------------------------
# Preprocessing (matches training pipeline — ResNet50 preprocess_input)
# ---------------------------------------------------------------------------
_preprocess = tf.keras.applications.resnet50.preprocess_input


def _prepare_image(raw_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)          # (224, 224, 3)
    arr = _preprocess(arr)                          # match training pipeline
    return np.expand_dims(arr, axis=0)              # (1, 224, 224, 3)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Pneumonia Detection API")

_cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/models")
def list_models():
    """Return available model names, experiment group, and metrics."""
    models = []
    for name, info in MODEL_REGISTRY.items():
        entry = {
            "name": name,
            "experiment": info["experiment"],
            "available": os.path.exists(info["path"]),
            "metrics": None,
        }
        try:
            entry["metrics"] = _get_metrics(name)
        except Exception:
            pass
        models.append(entry)
    return {"models": models}


@app.get("/metrics/{model_name}")
def get_metrics(model_name: str):
    """Return saved test-set metrics for a model."""
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")
    metrics = _get_metrics(model_name)
    return {"model": model_name, "metrics": metrics}


@app.post("/predict")
async def predict(image: UploadFile = File(...), model_name: str = Form(...)):
    """Run inference on an uploaded chest X-ray image."""
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

    raw = await image.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty image file")

    model = _get_model(model_name)
    img_tensor = _prepare_image(raw)
    preds = model.predict(img_tensor, verbose=0)[0]  # shape (2,)
    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx])

    return {
        "model": model_name,
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": round(confidence, 4),
        "probabilities": {
            CLASS_NAMES[i]: round(float(preds[i]), 4) for i in range(len(CLASS_NAMES))
        },
    }


# ---------------------------------------------------------------------------
# Run with: uvicorn app:app --reload --port 8000
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
