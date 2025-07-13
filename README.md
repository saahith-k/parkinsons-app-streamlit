# 🧠 Parkinson's Disease Detection App

A Streamlit web application that predicts whether a person has Parkinson's Disease based on spiral or wave-drawing images using a deep learning model (ResNet50).

---

## 🚀 Live Demo

👉 [Click to Try the App](https://your-app-link.streamlit.app)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link.streamlit.app)

---

## 📥 How to Use

1. Upload a spiral or wave drawing image (JPG or PNG).
2. The app uses a trained deep learning model to classify it as:
   - ✅ **Healthy**
   - 🚨 **Parkinson**
3. It also displays the prediction confidence and class probabilities as a chart.

---

## 🧠 Model Information

- Architecture: **ResNet50** (fine-tuned)
- Input Image Size: **256x256**
- Output: **2 classes** – Healthy, Parkinson
- Accuracy: **~86.73%** on test set
- Trained on real spiral and wave drawings dataset

---

## 📊 Evaluation Metrics

|                        | Predicted Healthy | Predicted Parkinson |
|------------------------|-------------------|---------------------|
| **Actual Healthy**     | 58                | 4                   |
| **Actual Parkinson**   | 7                 | 51                  |

> ✅ Final Test Accuracy: **86.73%**

---

## 📦 Local Setup

Clone the repo and run locally:

```bash
git clone https://github.com/saahith-k/parkinsons-app-streamlit.git
cd parkinsons-app-streamlit
pip install -r requirements.txt
streamlit run main.py
