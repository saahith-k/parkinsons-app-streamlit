# main.py

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="🧠 Parkinson's Detector", layout="centered")

st.title("🧠 Parkinson's Disease Detection from Drawing")
st.markdown("Upload a spiral or wave drawing image, and this app will predict whether the patient is **Healthy** or has **Parkinson's Disease**.")

# Load model
@st.cache_resource
def load_parkinsons_model():
    model = load_model("model.h5")
    return model

model = load_parkinsons_model()
IMG_SIZE = 256  # must match training input size

# Upload image
uploaded_file = st.file_uploader("\ud83d\udcc4 Upload a Spiral or Wave Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="\ud83c\udfa8 Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))

    class_names = ['Healthy', 'Parkinson']
    predicted_label = class_names[predicted_class]

    # Result
    st.subheader("\ud83d\udccc Prediction")
    st.success(f"**{predicted_label}** ({confidence * 100:.2f}% confidence)")

    # Probability chart
    st.subheader("\ud83d\udcca Class Probabilities")
    st.bar_chart({
        "Healthy": [prediction[0][0]],
        "Parkinson": [prediction[0][1]]
    })
