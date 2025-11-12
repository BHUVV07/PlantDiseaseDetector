# app_streamlit.py — enhanced UI with Top-3 predictions
# app_streamlit.py — enhanced UI with Top-3 predictions
import os
import streamlit as st
from PIL import Image
import joblib
import json
import numpy as np
from model_utils import preprocess_image_for_ml


st.set_page_config(page_title="Plant Disease Detector", layout="centered")

st.title("🌿 Plant Disease Detector (Random Forest / SVM)")
st.write("Upload a leaf image and get the top predictions with helpful info.")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Load disease information (optional JSON)
info_file = "disease_info.json"
if os.path.exists(info_file):
    with open(info_file, "r") as f:
        disease_info = json.load(f)
else:
    disease_info = {}

# File uploader
uploaded = st.file_uploader("Upload a leaf image (jpg, jpeg, png):", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    X = preprocess_image_for_ml(img)
    probs = None

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([X])[0]
        classes = list(model.classes_)
        top_idx = np.argsort(probs)[::-1][:3]

        st.subheader("🔍 Top-3 Predictions")
        for i in top_idx:
            conf = probs[i]
            st.write(f"**{classes[i]}** — Confidence: {conf:.2f}")
            st.progress(int(conf * 100))
    else:
        pred = model.predict([X])[0]
        st.success(f"Predicted class: **{pred}**")
        probs = None

    # Best prediction
    if probs is not None:
        best_idx = np.argmax(probs)
        best_class = classes[best_idx]
        confidence = probs[best_idx]
        st.markdown(f"### ✅ Most Likely Class: **{best_class}** ({confidence:.2f})")

        # Display disease info if available
        if best_class in disease_info:
            st.info(disease_info[best_class])
        else:
            st.warning("ℹ️ No disease info available for this class. Add details to `disease_info.json`.")

st.caption("Developed with ❤️ using Streamlit + Scikit-Learn")
