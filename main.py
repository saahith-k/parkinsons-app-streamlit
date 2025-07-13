import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import zipfile

# Constants
MODEL_URL = "https://drive.google.com/uc?id=1iCIb3jHourYKuGdSsPGnOeqkBWV18fIE"
MODEL_DIR = "model"

# Download and unzip model if not already present
if not os.path.exists(MODEL_DIR):
    with st.spinner("⬇️ Downloading model..."):
        gdown.download(MODEL_URL, "model.zip", quiet=False)
        with zipfile.ZipFile("model.zip", "r") as zip_ref:
            zip_ref.extractall(".")
    st.success("✅ Model downloaded and extracted!")

# Load model
with st.spinner("🔄 Loading model..."):
    model = tf.keras.models.load_model(MODEL_DIR)
st.success("✅ Model loaded successfully!")

# UI
st.title("🧠 Parkinson's Disease Detection")
st.write("Upload a spiral or wave image to check for Parkinson's symptoms.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("🔍 Predicting..."):
        prediction = model.predict(img_array)
        prob = float(prediction[0])  # assuming model outputs a single sigmoid value
        label = "Parkinson's" if prob > 0.5 else "Healthy"
        confidence = prob * 100 if prob > 0.5 else (1 - prob) * 100

    # Output
    st.write(f"### 🩺 Prediction: **{label}**")
    st.write(f"Confidence: `{confidence:.2f}%`")
