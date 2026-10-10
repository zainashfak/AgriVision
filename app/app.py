import datetime
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image

sys.path.append(str(Path(__file__).parent))
from utils import gradcam

CLASSES = ["BrownRust", "Healthy", "Mildew", "Septoria", "YellowRust"]
IMG_SIZE = 224
MODEL_DIR = Path(r"D:\AgriVision\models")
FT_MODEL = MODEL_DIR / "leaf_model_ft.keras"
BASE_MODEL = MODEL_DIR / "leaf_model.keras"
YIELD_MODEL = MODEL_DIR / "yield_model.joblib"

ADVICE = {
    "Healthy": "No disease detected. Keep monitoring the crop regularly.",
    "BrownRust": "Apply a recommended fungicide early (for example a triazole-based product), "
                 "and monitor nearby plants. Consult your local agriculture extension office.",
    "YellowRust": "Act quickly: yellow rust spreads fast in cool, humid weather. Apply a "
                  "recommended fungicide and consult your local agriculture extension office.",
    "Septoria": "Remove infected crop residue, avoid excess moisture, and apply a suitable "
                "fungicide if the infection spreads to upper leaves.",
    "Mildew": "Improve airflow, avoid over-fertilising with nitrogen, and apply a suitable "
              "fungicide if severity increases.",
}


@st.cache_resource
def load_leaf_model():
    return tf.keras.models.load_model(str(FT_MODEL if FT_MODEL.exists() else BASE_MODEL))


@st.cache_resource
def load_yield_model():
    return joblib.load(YIELD_MODEL)


@st.cache_data(ttl=3600)
def fetch_weather(lat, lon):
    """Average temperature, humidity and rainfall over the last ~30 days (NASA POWER)."""
    end = datetime.date.today() - datetime.timedelta(days=7)
    start = end - datetime.timedelta(days=29)
    url = ("https://power.larc.nasa.gov/api/temporal/daily/point"
           f"?parameters=T2M,RH2M,PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}"
           f"&start={start:%Y%m%d}&end={end:%Y%m%d}&format=JSON")
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    p = r.json()["properties"]["parameter"]

    def avg(key):
        return float(np.mean([v for v in p[key].values() if v != -999]))

    return avg("T2M"), avg("RH2M"), avg("PRECTOTCORR")


def estimate_severity(img_rgb):
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
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

    diseased = cv2.bitwise_and(cv2.inRange(hsv, (10, 70, 90), (34, 255, 255)), leaf_mask)
    healthy_px = int(np.count_nonzero(green))
    diseased_px = int(np.count_nonzero(diseased))
    total = healthy_px + diseased_px
    severity = 100 * diseased_px / total if total else 0.0

    overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay[green > 0] = (0, 200, 0)
    overlay[diseased > 0] = (255, 0, 0)
    return severity, overlay


leaf_model = load_leaf_model()
yield_model = load_yield_model()

st.title("AgriVision: Wheat Leaf Diagnosis")
st.write("Upload a wheat leaf photo to get the disease, severity, estimated yield loss "
         "and a recommended action.")

# ---- sidebar: crop, location and weather ----
st.sidebar.header("Settings")
st.sidebar.selectbox("Crop", ["Wheat"])
use_live = st.sidebar.checkbox("Fetch live weather (NASA POWER)", value=True)
lat = st.sidebar.number_input("Latitude", value=33.68, format="%.2f")
lon = st.sidebar.number_input("Longitude", value=73.05, format="%.2f")

st.sidebar.subheader("Manual weather (fallback)")
temp = st.sidebar.slider("Average temperature (C)", 0.0, 40.0, 16.0)
hum = st.sidebar.slider("Average humidity (%)", 20, 95, 55)
rain = st.sidebar.slider("Average rainfall (mm/day)", 0.0, 10.0, 2.1)

if use_live:
    try:
        temp, hum, rain = fetch_weather(lat, lon)
        st.sidebar.success("Live weather loaded")
    except Exception:
        st.sidebar.warning("Could not fetch live weather, using manual values.")
st.sidebar.write(f"Using: {temp:.1f} C, {hum:.0f}% humidity, {rain:.1f} mm/day")

file = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png"])

if file:
    img = np.array(Image.open(file).convert("RGB"))
    st.image(img, caption="Uploaded leaf", use_container_width=True)

    # 1. diagnosis
    x = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32")
    probs = leaf_model.predict(x[None, ...], verbose=0)[0]
    best = int(np.argmax(probs))
    disease = CLASSES[best]

    # 2. severity
    severity, sev_overlay = estimate_severity(img)

    # 3. yield loss (regression model)
    row = pd.DataFrame([{"severity": severity, "temp_c": temp, "humidity_pct": hum,
                         "rain_mm": rain, "disease": disease}])
    loss = 0.0 if disease == "Healthy" else float(np.clip(yield_model.predict(row)[0], 0, 100))

    # 4. explainability
    try:
        cam_overlay = gradcam(leaf_model, img, class_idx=best)
    except Exception as e:
        cam_overlay = None
        st.warning(f"Grad-CAM unavailable: {e}")

    st.subheader(f"Prediction: {disease}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{probs[best] * 100:.1f}%")
    c2.metric("Severity", f"{severity:.1f}%")
    c3.metric("Est. yield loss", f"{loss:.1f}%")

    st.info(f"Recommended action: {ADVICE[disease]}")

    col_a, col_b = st.columns(2)
    col_a.image(sev_overlay, caption="Severity: green = healthy, red = diseased",
                use_container_width=True)
    if cam_overlay is not None:
        col_b.image(cam_overlay, caption="Grad-CAM: areas that influenced the diagnosis",
                    use_container_width=True)

    st.bar_chart({c: float(p) for c, p in zip(CLASSES, probs)})
    st.caption("Decision-support prototype, not a replacement for an agricultural expert. "
               "Severity is a colour-based estimate and yield-loss labels are simulated.")