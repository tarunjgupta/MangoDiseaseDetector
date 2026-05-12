"""
app.py
------
Streamlit web interface for Mango Disease Detection.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
import numpy as np
from PIL import Image

import tensorflow as tf

from predict import load_model_and_classes
from patch_utils import patch_based_predict
from gradcam import generate_gradcam


# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Mango Disease Detector",
    page_icon="🥭",
    layout="centered",
)

# ──────────────────────────────────────────────
# Custom CSS for a clean look
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .prob-bar {
        background: #f0f0f0;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 6px;
        height: 28px;
        position: relative;
    }
    .prob-fill {
        height: 100%;
        border-radius: 8px;
        display: flex;
        align-items: center;
        padding-left: 10px;
        font-size: 0.85rem;
        font-weight: 600;
        color: white;
        min-width: fit-content;
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown('<div class="main-title">🥭 Mango Disease Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Patch-Based CNN with Grad-CAM Explainability</div>',
    unsafe_allow_html=True,
)

st.divider()


# ──────────────────────────────────────────────
# Load model (cached)
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load model once and cache it."""
    model, class_names = load_model_and_classes()
    return model, class_names


try:
    model, class_names = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(
        f"Could not load model. Make sure you have trained the model first.\n\n"
        f"Run: `python train.py --dataset ./dataset`\n\n"
        f"Error: {e}"
    )


# ──────────────────────────────────────────────
# File uploader
# ──────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a mango leaf image",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Supported formats: JPG, PNG, BMP",
)

if uploaded_file is not None and model_loaded:
    image = Image.open(uploaded_file).convert("RGB")

    # Show uploaded image
    st.image(image, caption="Uploaded Image")

    st.divider()

    with st.spinner(" Running patch-based prediction..."):
        # ── Patch-based prediction ──
        result = patch_based_predict(model, image, class_names)

    # ── Results ──
    predicted = result["predicted_class"]
    probs = result["probabilities"]
    n_patches = result["num_patches"]

    # Predicted disease
    st.markdown(f"### Predicted Disease: **{predicted}**")
    st.caption(f"Based on {n_patches} patches (128×128) with mean aggregation")

    st.divider()

    # ── Class-wise probabilities ──
    st.markdown("####  Class Probabilities")

    # Color mapping for the bars
    colors = {
        "Anthracnose": "#e74c3c",
        "Bacterial_Canker": "#e67e22",
        "Healthy": "#2ecc71",
        "Scab": "#9b59b6",
    }

    for cls in class_names:
        pct = probs[cls]
        color = colors.get(cls, "#3498db")
        st.markdown(
            f"""
            <div class="prob-bar">
                <div class="prob-fill" style="width: {max(pct, 8)}%; background: {color};">
                    {cls}: {pct:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Grad-CAM ──
    st.markdown("####  Grad-CAM Heatmap")
    st.caption("Highlights the regions the model focuses on for its prediction.")

    with st.spinner("Generating Grad-CAM..."):
        gradcam_image = generate_gradcam(model, image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original")
    with col2:
        st.image(gradcam_image, caption="Grad-CAM Overlay")

elif uploaded_file is None and model_loaded:
    st.info(" Upload a mango leaf image to get started.")
