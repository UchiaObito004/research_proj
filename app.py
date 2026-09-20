import os
import time
from io import BytesIO
import requests
import streamlit as st
from PIL import Image
import numpy as np

# ==========================================
# Page Configuration & Styling
# ==========================================
st.set_page_config(
    page_title="Fruit Freshness Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern aesthetic
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-fresh {
        background-color: #ecfdf5;
        border: 1.5px solid #10b981;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .status-rotten {
        background-color: #fef2f2;
        border: 1.5px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Sidebar: Project Info & Author Details
# ==========================================
with st.sidebar:
    st.image("fig_workflow.png" if os.path.exists("fig_workflow.png") else None, use_column_width=True)
    st.header("Project Information")
    st.markdown("""
    **College Project**: Deep Learning Fruit Freshness Classifier  
    **Institution**: Panipat Institute of Engineering and Technology (PIET)  
    **Champion Model**: **MobileNetV2** (Transfer Learning)
    """)
    
    st.divider()
    st.subheader("Model Benchmark")
    st.markdown("""
    | Metric | MobileNetV2 | EfficientNetB0 |
    | :--- | :---: | :---: |
    | **Accuracy** | **97.96%** | 49.89% |
    | **ROC-AUC** | **0.9985** | 0.7475 |
    | **F1-Score** | **0.9792** | 0.4256 |
    | **Disk Size** | **13.11 MB** | 20.11 MB |
    """)

    st.divider()
    st.subheader("Project Authors")
    st.markdown("""
    - **Ayush Kumar** (`ayushsyntax@gmail.com`)
    - **Bhushan Verma** (`vermabhushan004@gmail.com`)
    - **Jayant Jain** (`jayantjain058@gmail.com`)
    - **Dr. Mitu Sehgal** (`technomitusehgal@gmail.com`)
    """)

# ==========================================
# Main Interface
# ==========================================
st.markdown('<div class="main-title">🍎 Automated Fruit Freshness Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Computer Vision & Deep Transfer Learning for Automated Post-Harvest Food Quality Control</div>', unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000/predict"
MODEL_PATH = "model/freshness_model.h5"

# Cached local model loader (fallback when API is offline)
@st.cache_resource
def load_local_model():
    if os.path.exists(MODEL_PATH):
        import tensorflow as tf
        return tf.keras.models.load_model(MODEL_PATH)
    return None

col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    st.subheader("📤 Upload Fruit Image")
    uploaded_file = st.file_uploader(
        "Select a fruit photo (Apple, Banana, Orange, etc.)", 
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Input Preview", use_column_width=True)

with col2:
    st.subheader("🔍 Freshness Diagnostics")
    
    if uploaded_file is None:
        st.info("👈 Upload an image on the left panel to begin classification.")
    else:
        if st.button("🚀 Analyze Freshness", type="primary", use_container_width=True):
            with st.spinner("Executing deep neural inference..."):
                result = None
                used_local = False
                start_time = time.time()

                # Attempt 1: Call FastAPI backend
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(API_URL, files=files, timeout=4)
                    if response.status_code == 200:
                        result = response.json()
                except Exception:
                    pass

                # Attempt 2: Fallback to direct local model inference
                if result is None:
                    local_model = load_local_model()
                    if local_model is not None:
                        used_local = True
                        img_resized = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                        img_arr = np.expand_dims(np.array(img_resized, dtype=np.float32) / 255.0, axis=0)
                        pred = float(local_model.predict(img_arr, verbose=0)[0][0])
                        is_rotten = pred > 0.5
                        conf = pred if is_rotten else (1.0 - pred)
                        result = {
                            "prediction": "rotten" if is_rotten else "fresh",
                            "confidence": round(conf, 4),
                            "probability_rotten": round(pred, 4),
                            "probability_fresh": round(1.0 - pred, 4),
                            "recommendation": (
                                "Safe for commercial consumption and retail packaging."
                                if not is_rotten else
                                "Immediate sorting recommended. Isolate to prevent microbial cross-contamination."
                            ),
                            "latency_ms": round((time.time() - start_time) * 1000, 2),
                            "model": "MobileNetV2 (Local Fallback)"
                        }

                # Display Results
                if result is not None:
                    label = result["prediction"]
                    confidence = result["confidence"]
                    conf_pct = confidence * 100

                    if label == "fresh":
                        st.markdown(f"""
                        <div class="status-fresh">
                            <h2 style="color: #065f46; margin:0;">✅ FRESH FRUIT</h2>
                            <p style="color: #047857; font-size: 1.1rem; margin-top: 6px;">
                                <strong>Confidence:</strong> {conf_pct:.2f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="status-rotten">
                            <h2 style="color: #991b1b; margin:0;">❌ ROTTEN / SPOILED</h2>
                            <p style="color: #b91c1c; font-size: 1.1rem; margin-top: 6px;">
                                <strong>Confidence:</strong> {conf_pct:.2f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.write("")
                    st.progress(float(confidence))

                    # Metric cards
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Predicted Class", label.upper())
                    with m2:
                        st.metric("Confidence", f"{conf_pct:.2f}%")
                    with m3:
                        st.metric("Inference Time", f"{result.get('latency_ms', 0)} ms")

                    st.info(f"💡 **Recommendation:** {result['recommendation']}")

                    if used_local:
                        st.caption("ℹ️ *Inference served via Direct Model Engine (FastAPI offline).*")
                    else:
                        st.caption("ℹ️ *Inference served via FastAPI REST Microservice.*")
                else:
                    st.error("Failed to generate prediction. Please ensure `model/freshness_model.h5` exists.")

st.divider()
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.9rem;">
    Fruit Freshness Classification Project | Panipat Institute of Engineering and Technology (PIET) | 2026
</div>
""", unsafe_allow_html=True)