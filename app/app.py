import datetime
import io
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import tensorflow as tf
from fpdf import FPDF
from PIL import Image

st.set_page_config(page_title="AgriVision", page_icon="🌾", layout="wide")

sys.path.append(str(Path(__file__).parent))
from utils import gradcam

CLASSES = ["BrownRust", "Healthy", "Mildew", "Septoria", "YellowRust"]
IMG_SIZE = 224
MAX_PHOTOS = 10
MODEL_DIR = Path(r"D:\AgriVision\models")
FT_MODEL = MODEL_DIR / "leaf_model_ft.keras"
BASE_MODEL = MODEL_DIR / "leaf_model.keras"
YIELD_MODEL = MODEL_DIR / "yield_model.joblib"

ADVICE = {
    "Healthy": "No disease detected. Keep monitoring the crop regularly.",
    "BrownRust": "Apply a recommended fungicide early (for example a triazole-based product) "
                 "and monitor nearby plants. Consult your local agriculture extension office.",
    "YellowRust": "Act quickly: yellow rust spreads fast in cool, humid weather. Apply a "
                  "recommended fungicide and consult your local agriculture extension office.",
    "Septoria": "Remove infected crop residue, avoid excess moisture, and apply a suitable "
                "fungicide if the infection spreads to upper leaves.",
    "Mildew": "Improve airflow, avoid over-fertilising with nitrogen, and apply a suitable "
              "fungicide if severity increases.",
}

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1250px;}
.hero {background: linear-gradient(135deg, #14532d 0%, #16a34a 100%);
       padding: 1.5rem 2rem; border-radius: 16px; margin-bottom: 1.2rem;}
.hero h1 {color: #ffffff; margin: 0; font-size: 2.1rem; font-weight: 700;}
.hero p {color: #dcfce7; margin: .35rem 0 0 0; font-size: 1.02rem;}
.card {border: 1px solid rgba(100,116,139,.25); border-radius: 14px;
       padding: 1rem 1.2rem; background: #f8fafc; height: 100%;}
.card h4 {margin: 0 0 .3rem 0; color: #14532d;}
.card p {margin: 0; color: #475569; font-size: .93rem;}
div[data-testid="stMetric"] {background: #f0fdf4; border: 1px solid #bbf7d0;
       border-radius: 12px; padding: .7rem 1rem;}
.band {display: inline-block; padding: .15rem .7rem; border-radius: 999px;
       font-weight: 600; font-size: .85rem; color: #ffffff;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🌾 AgriVision</h1>
  <p>Wheat leaf disease diagnosis, severity and yield-loss estimation from a single photo.</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    leaf = tf.keras.models.load_model(str(FT_MODEL if FT_MODEL.exists() else BASE_MODEL))
    leaf.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32"), verbose=0)  # warm-up
    return leaf, joblib.load(YIELD_MODEL)


leaf_model, yield_model = load_models()


@st.cache_data(ttl=3600, show_spinner=False)
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


def read_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img.thumbnail((1024, 1024))
    return np.array(img)


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


@st.cache_data(show_spinner=False, max_entries=60)
def classify(file_bytes):
    img = read_image(file_bytes)
    x = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32")
    probs = leaf_model.predict(x[None, ...], verbose=0)[0]
    severity, overlay = estimate_severity(img)
    return {"img": img, "probs": probs, "best": int(np.argmax(probs)),
            "severity": severity, "overlay": overlay}


@st.cache_data(show_spinner="Computing Grad-CAM...", max_entries=60)
def cam_for(file_bytes, class_idx):
    return gradcam(leaf_model, read_image(file_bytes), class_idx=class_idx)


def estimate_loss(disease, severity, temp, hum, rain):
    if disease == "Healthy":
        return 0.0
    row = pd.DataFrame([{"severity": severity, "temp_c": temp, "humidity_pct": hum,
                         "rain_mm": rain, "disease": disease}])
    return float(np.clip(yield_model.predict(row)[0], 0, 100))


def band(sev):
    if sev < 10:
        return "Low", "#16a34a"
    if sev < 30:
        return "Moderate", "#d97706"
    return "High", "#dc2626"


# ---------------------------------------------------------------- PDF report
def _t(s):
    """Make text safe for the built-in PDF fonts (latin-1 only)."""
    return str(s).encode("latin-1", "replace").decode("latin-1")


def _pil(arr, max_side=700):
    im = Image.fromarray(arr)
    im.thumbnail((max_side, max_side))
    return im


def build_pdf(results, summary, temp, hum, rain, include_cam):
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # banner
    pdf.set_fill_color(20, 83, 45)
    pdf.rect(0, 0, 210, 32, style="F")
    pdf.set_xy(10, 8)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 9, "AgriVision Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(10)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Wheat leaf disease diagnosis, severity and yield-loss estimation",
             new_x="LMARGIN", new_y="NEXT")

    pdf.set_text_color(31, 41, 55)
    pdf.set_y(38)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Weather used: {temp:.1f} C, {hum:.0f}% humidity, {rain:.1f} mm/day",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Photos analysed: {len(results)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # summary table
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Summary", new_x="LMARGIN", new_y="NEXT")
    cols = [("File", 62), ("Disease", 32), ("Conf. (%)", 28),
            ("Severity (%)", 30), ("Yield loss (%)", 34)]
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(220, 252, 231)
    for name, w in cols:
        pdf.cell(w, 7, name, border=1, fill=True, new_x="RIGHT", new_y="TOP")
    pdf.ln(7)
    pdf.set_font("Helvetica", "", 9)
    for _, row in summary.iterrows():
        vals = [_t(str(row["File"])[:34]), row["Disease"], f"{row['Confidence (%)']:.1f}",
                f"{row['Severity (%)']:.1f}", f"{row['Est. yield loss (%)']:.1f}"]
        for (name, w), v in zip(cols, vals):
            pdf.cell(w, 7, str(v), border=1, new_x="RIGHT", new_y="TOP")
        pdf.ln(7)

    # one page per photo
    for r in results:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, _t(r["name"]), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(20, 83, 45)
        pdf.cell(0, 7, f"Diagnosis: {r['disease']} ({r['confidence']:.1f}% confidence)",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(31, 41, 55)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 6, f"Severity index: {r['severity']:.1f}% ({band(r['severity'])[0]})",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, f"Estimated yield loss: {r['loss']:.1f}%",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        y = pdf.get_y()
        panels = [("Photo", r["img"]), ("Severity map", r["overlay"])]
        if include_cam:
            panels.append(("Grad-CAM", cam_for(r["bytes"], r["best"])))
        x = 10
        for title, arr in panels:
            pdf.set_xy(x, y)
            pdf.set_font("Helvetica", "I", 9)
            pdf.cell(58, 5, title, new_x="RIGHT", new_y="TOP")
            pdf.image(_pil(arr), x=x, y=y + 6, w=58, h=78, keep_aspect_ratio=True)
            x += 63
        pdf.set_y(y + 90)

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Class probabilities", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for c, p in sorted(zip(CLASSES, r["probs"]), key=lambda t: -t[1]):
            pdf.cell(0, 5, f"{c}: {p * 100:.1f}%", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Recommended action", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5.5, _t(ADVICE[r["disease"]]), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(0, 4.5, _t(
        "Decision-support prototype, not a replacement for an agricultural expert. "
        "Severity is a rough colour-based index that tends to overestimate, and "
        "yield-loss labels are simulated."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Settings")
st.sidebar.selectbox("Crop", ["Wheat"])
show_cam = st.sidebar.toggle("Show Grad-CAM explanation", value=True,
                             help="Turn off for faster results on large batches.")
use_live = st.sidebar.toggle("Use live weather (NASA POWER)", value=True)

with st.sidebar.expander("Location"):
    lat = st.number_input("Latitude", value=33.68, format="%.2f")
    lon = st.number_input("Longitude", value=73.05, format="%.2f")

with st.sidebar.expander("Manual weather (fallback)"):
    temp = st.slider("Average temperature (C)", 0.0, 40.0, 16.0)
    hum = st.slider("Average humidity (%)", 20, 95, 55)
    rain = st.slider("Average rainfall (mm/day)", 0.0, 10.0, 2.1)

if use_live:
    try:
        temp, hum, rain = fetch_weather(lat, lon)
        st.sidebar.success("Live weather loaded")
    except Exception:
        st.sidebar.warning("Live weather unavailable, using manual values.")
st.sidebar.metric("Temperature", f"{temp:.1f} C")
st.sidebar.metric("Humidity", f"{hum:.0f} %")
st.sidebar.metric("Rainfall", f"{rain:.1f} mm/day")

# ---------------------------------------------------------------- upload
files = st.file_uploader("Upload wheat leaf photos (up to 10)",
                         type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if not files:
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="card"><h4>1. Upload</h4><p>Drop one or more close-up photos '
                'of wheat leaves.</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="card"><h4>2. Analyse</h4><p>A MobileNetV2 model diagnoses the '
                'disease, and image analysis estimates severity.</p></div>',
                unsafe_allow_html=True)
    c3.markdown('<div class="card"><h4>3. Act</h4><p>Get the estimated yield loss, a '
                'recommended action, a Grad-CAM explanation and a PDF report.</p></div>',
                unsafe_allow_html=True)
    st.stop()

if len(files) > MAX_PHOTOS:
    st.warning(f"Only the first {MAX_PHOTOS} photos are analysed.")
    files = files[:MAX_PHOTOS]

# ---------------------------------------------------------------- analysis
results = []
progress = st.progress(0.0, text="Analysing photos...")
for i, f in enumerate(files):
    data = f.getvalue()
    r = classify(data)
    disease = CLASSES[r["best"]]
    r.update({"name": f.name, "bytes": data, "disease": disease,
              "confidence": float(r["probs"][r["best"]]) * 100,
              "loss": estimate_loss(disease, r["severity"], temp, hum, rain)})
    results.append(r)
    progress.progress((i + 1) / len(files), text=f"Analysed {i + 1} of {len(files)}")
progress.empty()

summary = pd.DataFrame([{
    "File": r["name"], "Disease": r["disease"],
    "Confidence (%)": round(r["confidence"], 1),
    "Severity (%)": round(r["severity"], 1),
    "Est. yield loss (%)": round(r["loss"], 1)} for r in results])

# ---------------------------------------------------------------- summary + downloads
if len(results) > 1:
    st.subheader("Summary")
    k1, k2, k3 = st.columns(3)
    k1.metric("Photos analysed", len(results))
    k2.metric("Diseased leaves", int((summary["Disease"] != "Healthy").sum()))
    k3.metric("Average yield loss", f"{summary['Est. yield loss (%)'].mean():.1f}%")
    st.dataframe(summary, use_container_width=True, hide_index=True, column_config={
        "Confidence (%)": st.column_config.ProgressColumn(
            "Confidence (%)", min_value=0, max_value=100, format="%.1f"),
        "Severity (%)": st.column_config.ProgressColumn(
            "Severity (%)", min_value=0, max_value=100, format="%.1f"),
        "Est. yield loss (%)": st.column_config.ProgressColumn(
            "Est. yield loss (%)", min_value=0, max_value=100, format="%.1f"),
    })

st.subheader("Download")
d1, d2, d3 = st.columns([1, 1, 2])
d1.download_button("Download results (CSV)", summary.to_csv(index=False).encode("utf-8"),
                   "agrivision_results.csv", "text/csv", use_container_width=True)

sig = (tuple(f"{r['name']}-{len(r['bytes'])}" for r in results),
       round(temp, 1), round(hum), round(rain, 1), show_cam)
if d2.button("Prepare PDF report", type="primary", use_container_width=True):
    with st.spinner("Building PDF report..."):
        st.session_state["pdf"] = (sig, build_pdf(results, summary, temp, hum, rain, show_cam))
pdf_state = st.session_state.get("pdf")
if pdf_state and pdf_state[0] == sig:
    d3.download_button("Download PDF report", pdf_state[1], "agrivision_report.pdf",
                       "application/pdf", use_container_width=True)

st.subheader("Details")

# ---------------------------------------------------------------- per-photo results
for i, r in enumerate(results):
    title = f"{r['name']}  |  {r['disease']}  |  {r['confidence']:.0f}% confidence"
    with st.expander(title, expanded=(i == 0)):
        left, right = st.columns([1, 1.4], gap="large")
        with left:
            st.image(r["img"], use_container_width=True)
        with right:
            st.subheader(r["disease"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", f"{r['confidence']:.1f}%")
            m2.metric("Severity index", f"{r['severity']:.1f}%")
            m3.metric("Est. yield loss", f"{r['loss']:.1f}%")

            label, colour = band(r["severity"])
            st.markdown(f'Severity level: <span class="band" style="background:{colour}">'
                        f'{label}</span>', unsafe_allow_html=True)
            st.progress(min(int(r["severity"]), 100))

            if r["confidence"] < 50:
                st.warning("Low confidence. Retake the photo in good light, close to one leaf.")
            st.info(f"**Recommended action:** {ADVICE[r['disease']]}")

        t1, t2, t3 = st.tabs(["Severity map", "Grad-CAM", "Class probabilities"])
        with t1:
            st.image(r["overlay"], caption="Green = healthy tissue, red = diseased tissue",
                     use_container_width=True)
        with t2:
            if show_cam:
                st.image(cam_for(r["bytes"], r["best"]),
                         caption="Warmer colours = regions that influenced the diagnosis",
                         use_container_width=True)
            else:
                st.info("Grad-CAM is switched off in the sidebar.")
        with t3:
            st.bar_chart(pd.DataFrame({"probability": r["probs"]}, index=CLASSES),
                         horizontal=True)

st.caption("Decision-support prototype, not a replacement for an agricultural expert. "
           "Severity is a rough colour-based index, and yield-loss labels are simulated.")