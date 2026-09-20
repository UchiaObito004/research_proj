import time
import os
from io import BytesIO
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf

# ==========================================
# FastAPI Application Configuration
# ==========================================
app = FastAPI(
    title="Fruit Freshness Classification API",
    description=(
        "Production-grade RESTful API for automated fruit freshness detection "
        "using deep transfer learning (MobileNetV2 champion model). "
        "Developed by Ayush Kumar, Bhushan Verma, Jayant Jain, and Dr. Mitu Sehgal "
        "(Panipat Institute of Engineering and Technology)."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-platform integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Model Loading & Warm-up
# ==========================================
MODEL_PATH = "model/freshness_model.h5"
IMG_SIZE = (224, 224)
TARGET_CLASSES = ["fresh", "rotten"]

model = None

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Loaded production model from {MODEL_PATH}")
    else:
        print(f"Warning: {MODEL_PATH} not found. Please verify model directory.")
except Exception as e:
    print(f"Error loading model: {e}")


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess uploaded PIL Image to match model input specifications."""
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
    return img_array


# ==========================================
# API Endpoints
# ==========================================
@app.get("/", tags=["General"])
def root():
    return {
        "title": "Fruit Freshness Classification API",
        "version": "2.0.0",
        "status": "online",
        "documentation": "/docs",
        "authors": [
            "Ayush Kumar (ayushsyntax@gmail.com)",
            "Bhushan Verma (vermabhushan004@gmail.com)",
            "Jayant Jain (jayantjain058@gmail.com)",
            "Dr. Mitu Sehgal (technomitusehgal@gmail.com)"
        ],
        "institution": "Panipat Institute of Engineering and Technology"
    }


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "champion_architecture": "MobileNetV2 (Transfer Learning)",
        "test_accuracy": "97.96%",
        "roc_auc_score": "0.9985",
        "input_tensor_shape": [1, IMG_SIZE[0], IMG_SIZE[1], 3]
    }


@app.get("/model/info", tags=["Model"])
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    return {
        "model_name": "MobileNetV2_Classifier",
        "total_parameters": model.count_params(),
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "classes": TARGET_CLASSES,
        "threshold": 0.5
    }


@app.post("/predict", tags=["Inference"])
async def predict_freshness(file: UploadFile = File(...)):
    """
    Accepts an uploaded fruit image and predicts its freshness status.
    Returns:
        - prediction: 'fresh' or 'rotten'
        - confidence: Probability score (0.0 to 1.0)
        - latency_ms: Execution time in milliseconds
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is not available. Please verify model/freshness_model.h5"
        )

    # Validate content type
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Please upload a JPEG or PNG image."
        )

    start_time = time.time()
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Preprocessing and inference
    try:
        processed_img = preprocess_image(image)
        raw_pred = float(model.predict(processed_img, verbose=0)[0][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    latency_ms = round((time.time() - start_time) * 1000, 2)

    # Label calculation (0 = fresh, 1 = rotten)
    is_rotten = raw_pred > 0.5
    predicted_label = "rotten" if is_rotten else "fresh"
    confidence = raw_pred if is_rotten else (1.0 - raw_pred)

    recommendation = (
        "Safe for commercial consumption and retail packaging."
        if not is_rotten
        else "Immediate sorting recommended. Isolate to prevent microbial cross-contamination."
    )

    return JSONResponse({
        "success": True,
        "prediction": predicted_label,
        "confidence": round(confidence, 4),
        "probability_rotten": round(raw_pred, 4),
        "probability_fresh": round(1.0 - raw_pred, 4),
        "recommendation": recommendation,
        "latency_ms": latency_ms,
        "model": "MobileNetV2"
    })