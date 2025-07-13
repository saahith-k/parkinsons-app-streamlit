import streamlit as st
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

# --- Download model from Google Drive ---
def download_model_from_gdrive(file_id, dest_path):
    if not os.path.exists(dest_path):
        with st.spinner("📥 Downloading model from Google Drive..."):
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            response = requests.get(url)
            with open(dest_path, 'wb') as f:
                f.write(response.content)
            st.success("✅ Model downloaded successfully!")

# --- Setup ---
MODEL_FILE = "model.h5"
GDRIVE_FILE_ID = "1cndFy5v6750UPrBgEJ0hb70ZDUe_LnTy"
download_model_from_gdrive(GDRIVE_FILE_ID, MODEL_FILE)

# --- Load model ---
model = load_model(MODEL_FILE)
IMG_SIZE = 256
CLASS_NAMES = ['Healthy', 'Parkinson']

# --- Streamlit UI ---
st.set_page_config(page_title="🧠 Parkinson's Detection", layout="centered")
st.title("🧠 Parkinson's Disease Detection")
st.markdown("Upload a **spiral or wave image** to check if it's predicted as Healthy or Parkinson's.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred_probs = model.predict(img_array)[0]
    pred_class = np.argmax(pred_probs)
    confidence = pred_probs[pred_class] * 100
    label = CLASS_NAMES[pred_class]

    # Output
    st.markdown("---")
    st.subheader("📊 Prediction")
    st.markdown(f"**🧠 Predicted Class:** `{label}`")
    st.markdown(f"**🔍 Confidence:** `{confidence:.2f}%`")
