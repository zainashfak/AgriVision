import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

CLASSES = ["BrownRust", "Healthy", "Mildew", "Septoria", "YellowRust"]
IMG_SIZE = 224

# Assumed maximum yield loss (%) at 100% severity (not measured from data)
MAX_LOSS = {"Healthy": 0, "YellowRust": 50, "BrownRust": 35, "Septoria": 30, "Mildew": 25}


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r"D:\AgriVision\models\leaf_model.keras")


def estimate_severity(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (512, 512))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    green = cv2.inRange(hsv, (35, 60, 50), (85, 255, 255))

    leaf = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    leaf = cv2.dilate(leaf, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(leaf)
    leaf_mask = np.zeros_like(leaf)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 3000:
            leaf_mask[labels == i] = 255

    diseased = cv2.inRange(hsv, (10, 70, 90), (34, 255, 255))
    diseased = cv2.bitwise_and(diseased, leaf_mask)

    healthy_px = int(np.count_nonzero(green))
    diseased_px = int(np.count_nonzero(diseased))
    total = healthy_px + diseased_px
    severity = 100 * diseased_px / total if total else 0.0

    overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay[green > 0] = (0, 200, 0)
    overlay[diseased > 0] = (255, 0, 0)
    return severity, overlay


def weather_factor(humidity_pct, rain_mm):
    h = np.clip(humidity_pct / 60, 0.8, 1.3)
    r = np.clip(1 + (rain_mm - 2) * 0.05, 0.9, 1.2)
    return float(h * r)


def estimate_yield_loss(disease, severity_pct, humidity_pct, rain_mm):
    base = MAX_LOSS[disease] * (severity_pct / 100)
    return round(min(base * weather_factor(humidity_pct, rain_mm), 100), 1)


model = load_model()

st.title("AgriVision: Wheat Leaf Diagnosis")
st.write("Upload a wheat leaf photo to get the disease, its severity and the estimated yield loss.")

st.sidebar.header("Season weather")
humidity = st.sidebar.slider("Average humidity (%)", 30, 90, 55)
rain = st.sidebar.slider("Average rainfall (mm/day)", 0.0, 8.0, 2.1)

file = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded leaf", use_container_width=True)

    # 1. diagnosis
    x = np.array(img.resize((IMG_SIZE, IMG_SIZE)), dtype="float32")
    probs = model.predict(x[None, ...])[0]
    best = int(np.argmax(probs))
    disease = CLASSES[best]

    # 2. severity
    severity, overlay = estimate_severity(img)

    # 3. yield loss
    loss = estimate_yield_loss(disease, severity, humidity, rain)

    st.subheader(f"Prediction: {disease}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{probs[best] * 100:.1f}%")
    c2.metric("Severity", f"{severity:.1f}%")
    c3.metric("Est. yield loss", f"{loss}%")

    st.image(overlay, caption="Green = healthy tissue, red = diseased tissue",
             use_container_width=True)
    st.bar_chart({c: float(p) for c, p in zip(CLASSES, probs)})
    st.caption("Severity is a rough color-based estimate. Yield loss uses assumed "
               "disease factors, so treat it as an approximation.")