import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

CLASSES = ["BrownRust", "Healthy", "Mildew", "Septoria", "YellowRust"]
IMG_SIZE = 224

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r"D:\AgriVision\models\leaf_model.keras")

model = load_model()

st.title("AgriVision: Wheat Leaf Disease Diagnosis")
st.write("Upload a photo of a wheat leaf to get a diagnosis.")

file = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded leaf", use_container_width=True)

    x = np.array(img.resize((IMG_SIZE, IMG_SIZE)), dtype="float32")
    probs = model.predict(x[None, ...])[0]

    best = int(np.argmax(probs))
    st.subheader(f"Prediction: {CLASSES[best]}")
    st.write(f"Confidence: {probs[best] * 100:.1f}%")
    st.bar_chart({c: float(p) for c, p in zip(CLASSES, probs)})