"""
AI Disease Prediction System
Streamlit Cloud-ready version
- Zero external CDN dependencies (no Google Fonts, no Font Awesome)
- Zero JavaScript (pure CSS animations only)
- All styling via st.markdown unsafe_allow_html
- Drop-in replacement for the original disease_prediction.py
"""

import random
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Reduce TensorFlow/absl noise in Streamlit reruns.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model

tf.get_logger().setLevel("ERROR")

# ── PAGE CONFIG  (must be the very first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="AI Disease Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── ALL CSS — self-contained, zero external dependencies ──────────────────────
GLOBAL_CSS = """
<style>
/* ── CSS VARIABLES ── */
:root {
    --blue:    #4A90E2;
    --blue-dk: #2d6dbf;
    --teal:    #2ECC71;
    --teal-dk: #25a85d;
    --card:    rgba(255,255,255,0.96);
    --shadow:  0 8px 32px rgba(74,144,226,0.14);
    --r:       16px;
    --txt:     #1a2a3a;
    --muted:   #6b8299;
}

/* ── BASE ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    background: linear-gradient(150deg, #ddeefa 0%, #f5faff 60%, #ffffff 100%) !important;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}
[data-testid="stMainBlockContainer"] {
    background: transparent !important;
}

/* hide Streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden !important; }
[data-testid="stSidebar"] { display: none !important; }
.block-container {
    padding-top: 0 !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 1200px !important;
}

/* ── FADE-IN ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0);    }
}
.fadein { animation: fadeUp 0.7s ease both; }

/* ── FLOATING BACKGROUND BLOBS ── */
@keyframes drift {
    0%,100% { transform: translateY(0)    rotate(0deg);  }
    40%     { transform: translateY(-22px) rotate(7deg);  }
    70%     { transform: translateY(12px)  rotate(-5deg); }
}
.bg-shapes {
    position: fixed; inset: 0;
    pointer-events: none; z-index: 0; overflow: hidden;
}
.sh {
    position: absolute; border-radius: 50%; opacity: 0.07;
    animation: drift linear infinite;
}
.sh1 { width:220px; height:220px; background:var(--blue);  top:4%;  left:1%;  animation-duration:15s; }
.sh2 { width:160px; height:160px; background:var(--teal);  top:9%;  right:3%; animation-duration:12s; animation-delay:2s; }
.sh3 { width:300px; height:300px; background:var(--blue);  top:48%; left:0%;  animation-duration:19s; animation-delay:1s; }
.sh4 { width:120px; height:120px; background:var(--teal);  top:63%; right:2%; animation-duration:14s; animation-delay:3s; }
.sh5 { width:200px; height:200px; background:#9b59b6;      top:78%; left:38%; animation-duration:17s; animation-delay:4s; }

/* ── HEADER ── */
.hdr {
    background: linear-gradient(125deg, #102f5e 0%, #1d4f94 40%, #1aa77a 100%);
    padding: 40px 44px 26px;
    border-radius: 0 0 36px 36px;
    box-shadow: 0 10px 42px rgba(14,51,97,0.30);
    position: relative; overflow: hidden;
    animation: fadeUp 0.6s ease both;
    margin-bottom: 4px;
}
.hdr::after {
    content: '';
    position: absolute;
    width: 420px;
    height: 420px;
    right: -130px;
    top: -170px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.26), rgba(255,255,255,0));
    animation: pulseGlow 6s ease-in-out infinite;
}
.hdr::before {
    content: '';
    position: absolute; inset: 0;
    background: repeating-linear-gradient(
        45deg,
        rgba(255,255,255,0.03) 0px, rgba(255,255,255,0.03) 1px,
        transparent 1px, transparent 12px
    );
}
@keyframes pulseGlow {
    0%, 100% { transform: scale(1); opacity: 0.75; }
    50% { transform: scale(1.08); opacity: 1; }
}
.hdr-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.28);
    border-radius: 30px; padding: 5px 16px;
    font-size: 0.78rem; font-weight: 600; color: #fff;
    letter-spacing: 0.4px; margin-bottom: 14px;
}
.hdr-title {
    font-size: 2.4rem; font-weight: 800; color: #fff;
    margin: 0 0 6px; letter-spacing: -0.5px;
    text-shadow: 0 2px 14px rgba(0,0,0,0.2);
}
.hdr-sub {
    font-size: 1rem; color: rgba(255,255,255,0.82);
    margin: 0 0 22px; font-weight: 400;
}
.hdr-grid {
    display: grid;
    grid-template-columns: 1.5fr 1fr;
    gap: 18px;
    align-items: center;
    position: relative;
    z-index: 2;
}
.hdr-panel {
    background: rgba(255,255,255,0.11);
    border: 1px solid rgba(255,255,255,0.24);
    border-radius: 14px;
    padding: 12px 14px;
    backdrop-filter: blur(3px);
}
.hdr-panel-title {
    font-size: 0.74rem;
    color: rgba(255,255,255,0.85);
    letter-spacing: 1px;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.hdr-pills { display: flex; flex-wrap: wrap; gap: 8px; }
.hdr-pill {
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.28);
    color: #f2fbff;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 600;
}
.thought-wrap {
    position: relative;
    overflow: hidden;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.22);
    background: rgba(7, 26, 52, 0.35);
}
.thought-track {
    display: flex;
    width: max-content;
    animation: thoughtSlide 16s linear infinite;
}
.thought-item {
    color: rgba(236,247,255,0.93);
    font-size: 0.82rem;
    font-weight: 500;
    padding: 9px 16px;
    white-space: nowrap;
}
@keyframes thoughtSlide {
    from { transform: translateX(0); }
    to { transform: translateX(-50%); }
}

/* ── ECG ANIMATION ── */
@keyframes ecgDraw {
    0%  { stroke-dashoffset: 920; opacity: 1; }
    85% { stroke-dashoffset: 0;   opacity: 1; }
    100%{ stroke-dashoffset: 0;   opacity: 0; }
}
.ecg-svg  { width:100%; height:44px; overflow:hidden; display:block; }
.ecg-path {
    stroke: rgba(255,255,255,0.5); stroke-width: 2.2; fill: none;
    stroke-dasharray: 920; stroke-dashoffset: 920;
    animation: ecgDraw 3.2s linear infinite;
}

/* ── STEP INDICATORS ── */
.steps {
    display: flex; align-items: center;
    margin: 26px 0 20px;
    animation: fadeUp 0.7s 0.1s ease both; opacity: 0;
}
.step-item {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; position: relative;
}
.step-item:not(:last-child)::after {
    content:''; position:absolute; top:17px; left:55%;
    width:90%; height:2px;
    background: linear-gradient(90deg, var(--blue), rgba(74,144,226,0.12));
}
.step-dot {
    width:34px; height:34px; border-radius:50%;
    background: linear-gradient(135deg, var(--blue), var(--teal));
    color:#fff; font-size:0.82rem; font-weight:700;
    display:flex; align-items:center; justify-content:center;
    box-shadow: 0 3px 12px rgba(74,144,226,0.38);
    position:relative; z-index:1;
}
.step-lbl {
    font-size:0.7rem; color:var(--muted);
    margin-top:6px; font-weight:600; text-align:center;
}

/* ── CARD ── */
.card {
    background: var(--card);
    border-radius: var(--r); box-shadow: var(--shadow);
    padding: 26px 28px; border: 1px solid rgba(74,144,226,0.09);
    animation: fadeUp 0.7s 0.15s ease both; opacity: 0;
}
.slabel {
    font-size:0.68rem; font-weight:700; letter-spacing:2px;
    text-transform:uppercase; color:var(--blue); margin-bottom:4px;
}
.stitle {
    font-size:1.25rem; font-weight:700; color:var(--txt); margin:0 0 18px;
}

/* ── SELECTBOX ── */
[data-testid="stSelectbox"] label {
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    color: var(--txt) !important;
}
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #f0f7ff !important;
    border: 1.5px solid rgba(74,144,226,0.22) !important;
    border-radius: 10px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(74,144,226,0.12) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] div {
    color: var(--txt) !important;
    opacity: 1 !important;
}
[data-baseweb="popover"] {
    background: #ffffff !important;
}
[data-baseweb="popover"] > div {
    background: #ffffff !important;
    border: 1px solid rgba(74,144,226,0.2) !important;
}
[data-baseweb="menu"] {
    background: #ffffff !important;
}
[data-baseweb="popover"] [role="option"] {
    color: var(--txt) !important;
    background: #ffffff !important;
}
[data-baseweb="popover"] [role="option"][aria-selected="true"] {
    background: #eaf4ff !important;
    color: #1a2a3a !important;
}
[data-baseweb="popover"] [role="option"]:hover {
    background: #f3f9ff !important;
    color: #1a2a3a !important;
}

/* ── PRIMARY BUTTON ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #4A90E2 0%, #2d6dbf 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important; font-size: 0.95rem !important;
    padding: 13px 36px !important; width: 100% !important;
    margin-top: 12px !important;
    box-shadow: 0 4px 18px rgba(74,144,226,0.36) !important;
    transition: transform 0.18s, box-shadow 0.18s !important;
    letter-spacing: 0.2px !important;
}
[data-testid="stButton"] > button:hover:not(:disabled) {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(74,144,226,0.48) !important;
}
[data-testid="stButton"] > button:disabled {
    background: linear-gradient(135deg, #a8c4e4 0%, #8eaed0 100%) !important;
    box-shadow: none !important; transform: none !important;
    cursor: not-allowed !important; opacity: 0.72 !important;
}

/* ── RESULT CARD ── */
@keyframes resultPop {
    from { opacity:0; transform:scale(0.94) translateY(10px); }
    to   { opacity:1; transform:scale(1)    translateY(0);    }
}
.result-card {
    background: linear-gradient(135deg, #edfbf3 0%, #e8f4fd 100%);
    border: 2px solid rgba(46,204,113,0.3);
    border-radius: var(--r); padding: 26px 28px;
    animation: resultPop 0.55s cubic-bezier(0.34,1.56,0.64,1) both;
    box-shadow: 0 6px 28px rgba(46,204,113,0.13);
    margin-bottom: 18px;
}
.r-label { font-size:0.68rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:var(--teal-dk); margin-bottom:4px; }
.r-name  { font-size:1.9rem; font-weight:800; color:var(--txt); line-height:1.15; margin-bottom:4px; }
.r-sub   { font-size:0.82rem; color:var(--muted); margin-bottom:22px; }

/* ── PROGRESS BARS ── */
@keyframes fillIn {
    from { width: 0%; }
}
.pb-wrap { margin-bottom:12px; }
.pb-head { display:flex; justify-content:space-between; font-size:0.8rem; font-weight:600; color:var(--txt); margin-bottom:5px; }
.pb-track { background:#ddeefa; border-radius:8px; height:9px; overflow:hidden; }
.pb-fill  { height:100%; border-radius:8px; animation:fillIn 1.1s cubic-bezier(0.4,0,0.2,1) both; }

/* ── DISEASE PILLS ── */
@keyframes pillIn {
    from { opacity:0; transform:translateY(5px); }
    to   { opacity:1; transform:translateY(0);   }
}
.pill-wrap { display:flex; flex-wrap:wrap; gap:7px; margin-top:8px; }
.pill {
    background: rgba(74,144,226,0.09);
    border: 1px solid rgba(74,144,226,0.2);
    border-radius:20px; padding:4px 13px;
    font-size:0.78rem; color:#2d5f9e; font-weight:600;
    animation:pillIn 0.4s ease both;
}

/* ── HEALTH TIPS ── */
@keyframes tipSlide {
    from { opacity:0; transform:translateX(-10px); }
    to   { opacity:0.93; transform:translateX(0); }
}
.tips-card {
    background: linear-gradient(135deg, #1a3f7a 0%, #4A90E2 100%);
    border-radius:var(--r); padding:22px 26px; color:#fff;
    margin-top:20px;
    animation:fadeUp 0.7s 0.25s ease both; opacity:0;
}
.tips-title { font-size:1rem; font-weight:700; margin-bottom:14px; }
.tip-row {
    display:flex; align-items:flex-start; gap:10px;
    padding:9px 0; border-bottom:1px solid rgba(255,255,255,0.1);
    font-size:0.85rem; line-height:1.5;
    animation:tipSlide 0.5s ease both; opacity:0;
}
.tip-row:last-child { border-bottom:none; }
.tip-row:nth-child(2) { animation-delay:0.05s; }
.tip-row:nth-child(3) { animation-delay:0.12s; }
.tip-row:nth-child(4) { animation-delay:0.19s; }
.tip-row:nth-child(5) { animation-delay:0.26s; }
.tip-ico { flex-shrink:0; margin-top:1px; }

/* ── DISCLAIMER ── */
.disc {
    background:rgba(74,144,226,0.06);
    border-left:4px solid var(--blue);
    border-radius:0 10px 10px 0;
    padding:12px 18px;
    font-size:0.79rem; color:var(--muted);
    margin-top:24px; line-height:1.6;
}
.disc strong { color:var(--blue-dk); }

/* ── WARNING HINT ── */
.warn-hint {
    background:rgba(230,126,34,0.08);
    border-left:3px solid #e67e22;
    border-radius:0 8px 8px 0;
    padding:8px 14px; font-size:0.8rem;
    color:#a0522d; margin-top:10px; font-weight:500;
}

/* ── PLOTLY ── */
[data-testid="stPlotlyChart"] { border-radius:14px; overflow:hidden; }
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── DATASET LOADING ───────────────────────────────────────────────────────────
try:
    df = pd.read_csv("resources/dataset_kaggle.csv")
except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.stop()

# ── SYMPTOMS ──────────────────────────────────────────────────────────────────
SYMPTOMS = [
    "Anemia", "Anxiety", "Aura", "Belching", "Bladder issues", "Bleeding mole",
    "Blisters", "Bloating", "Blood in stool", "Body aches", "Bone fractures",
    "Bone pain", "Bowel issues", "Burning", "Butterfly-shaped rash",
    "Change in bowel habits", "Change in existing mole", "Chest discomfort",
    "Chest pain", "Congestion", "Constipation", "Coughing up blood", "Depression",
    "Diarrhea", "Difficulty performing familiar tasks", "Difficulty sleeping",
    "Difficulty swallowing", "Difficulty thinking", "Difficulty walking",
    "Double vision", "Easy bruising", "Fatigue", "Fear", "Frequent infections",
    "Frequent urination", "Fullness", "Gas", "Hair loss", "Hard lumps", "Headache",
    "Hunger", "Inability to defecate", "Increased mucus production",
    "Increased thirst", "Irregular heartbeat", "Irritability", "Itching",
    "Jaw pain", "Limited range of motion", "Loss of automatic movements",
    "Loss of height", "Loss of smell", "Loss of taste", "Lump or swelling",
    "Mild fever", "Misplacing things", "Morning stiffness", "Mouth sores",
    "Mucus production", "Nausea", "Neck stiffness", "Nosebleeds", "Numbness",
    "Pain during urination", "Pale skin", "Persistent cough", "Persistent pain",
    "Pigment spread", "Pneumonia", "Poor judgment", "Problems with words",
    "Rapid pulse", "Rash", "Receding gums", "Redness", "Redness in joints",
    "Reduced appetite", "Seizures", "Sensitivity to light", "Severe headache",
    "Shortness of breath", "Skin changes", "Skin infections", "Slight fever",
    "Sneezing", "Sore that doesn't heal", "Soreness", "Staring spells",
    "Stiff joints", "Stooped posture", "Swelling", "Swelling in ankles",
    "Swollen joints", "Swollen lymph nodes", "Tender abdomen", "Tenderness",
    "Thickened skin", "Throbbing pain", "Tophi", "Tremor", "Unconsciousness",
    "Unexplained bleeding", "Unexplained fevers", "Vomiting", "Weakness",
    "Withdrawal from work", "Writing changes",
]

SYMPTOM_ICONS = {
    "Headache": "🤕", "Fatigue": "😴", "Nausea": "🤢", "Vomiting": "🤮",
    "Chest pain": "💔", "Shortness of breath": "😮‍💨", "Diarrhea": "🚽",
    "Rash": "🔴", "Seizures": "⚡", "Tremor": "〰️", "Swelling": "🫧",
    "Weakness": "💪", "Coughing up blood": "🩸", "Anxiety": "😰",
    "Depression": "😞", "Hair loss": "👤", "Sneezing": "🤧", "Numbness": "🫥",
}

HEALTH_TIPS = [
    ("💧", "Stay well-hydrated — aim for at least 8 glasses of water daily."),
    ("🚶", "30 minutes of moderate walking daily significantly reduces cardiovascular risk."),
    ("🌙", "Quality sleep (7–9 hours) strengthens immune resilience and cognition."),
    ("🥦", "A balanced diet rich in fibre and micronutrients aids disease prevention."),
]

BAR_COLORS = ["#4A90E2", "#2ECC71", "#f39c12", "#e74c3c", "#9b59b6"]


def _normalize_text(text: str) -> str:
    return text.replace("’", "'").strip()


def train_and_save_model(output_path: Path | None = None):
    train_df = pd.read_csv("resources/dataset_kaggle.csv")
    symptom_cols = [c for c in train_df.columns if c.startswith("Symptom_")]

    symptom_index = {_normalize_text(s): i for i, s in enumerate(SYMPTOMS)}
    diseases = sorted(train_df["Disease"].unique())
    disease_to_index = {d: i for i, d in enumerate(diseases)}

    x = np.zeros((len(train_df), len(SYMPTOMS)), dtype=np.float32)
    y = np.zeros((len(train_df),), dtype=np.int32)

    for row_i, (_, row) in enumerate(train_df.iterrows()):
        for col in symptom_cols:
            sym = _normalize_text(str(row[col]))
            if sym in symptom_index:
                x[row_i, symptom_index[sym]] = 1.0
        y[row_i] = disease_to_index[row["Disease"]]

    y_cat = tf.keras.utils.to_categorical(y, num_classes=len(diseases))

    model = Sequential(
        [
            Dense(256, activation="relu", input_shape=(len(SYMPTOMS),)),
            Dropout(0.25),
            Dense(128, activation="relu"),
            Dropout(0.2),
            Dense(len(diseases), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        x,
        y_cat,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
    )

    if output_path is None:
        output_path = Path("resources/mlp_model.h5")
    model.save(str(output_path))
    metrics = {
        "train_accuracy": float(history.history["accuracy"][-1]),
        "val_accuracy": float(history.history["val_accuracy"][-1]),
        "train_loss": float(history.history["loss"][-1]),
        "val_loss": float(history.history["val_loss"][-1]),
    }
    return output_path, metrics


@st.cache_resource(show_spinner=False)
def load_model_cached(model_path: str, model_mtime: float):
    _ = model_mtime  # included in cache key to invalidate stale loads
    return load_model(model_path, compile=False)


def resolve_model_path() -> tuple[Path, bool]:
    model_candidates = [Path("disease_model.h5"), Path("resources/mlp_model.h5")]
    for model_path in model_candidates:
        if model_path.exists():
            return model_path, False

    trained_path, trained_metrics = train_and_save_model(Path("resources/mlp_model.h5"))
    st.session_state.last_train_metrics = trained_metrics
    return trained_path, True


try:
    active_model_path, model_was_trained = resolve_model_path()
except Exception as exc:
    st.error(f"Model could not be loaded or trained: {exc}")
    st.stop()

if model_was_trained:
    st.success(f"No pretrained model found. Trained a new model at: {active_model_path}")

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "selected_symptoms" not in st.session_state:
    st.session_state.selected_symptoms = ["Please Select"] * 5
if "active_model_path" not in st.session_state:
    st.session_state.active_model_path = str(active_model_path)
if "retrain_status" not in st.session_state:
    st.session_state.retrain_status = ""
if "last_train_metrics" not in st.session_state:
    st.session_state.last_train_metrics = None

if active_model_path.exists():
    current_model_path = Path(st.session_state.active_model_path)
    if not current_model_path.exists():
        current_model_path = active_model_path
        st.session_state.active_model_path = str(active_model_path)
    model = load_model_cached(str(current_model_path), current_model_path.stat().st_mtime)
else:
    st.error("Model file is missing after setup. Please re-train the model.")
    st.stop()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="bg-shapes">
  <div class="sh sh1"></div><div class="sh sh2"></div>
  <div class="sh sh3"></div><div class="sh sh4"></div><div class="sh sh5"></div>
</div>

<div class="hdr fadein">
  <div class="hdr-grid">
    <div>
      <div class="hdr-badge">🛡️ &nbsp;AI-Powered Clinical Decision Support</div>
      <div class="hdr-title">🩺 AI Disease Prediction System</div>
      <div class="hdr-sub">
        Bridging clinical expertise with machine intelligence for faster, smarter first-line triage.
      </div>
    </div>
    <div class="hdr-panel">
      <div class="hdr-panel-title">AI + Medical Intelligence</div>
      <div class="hdr-pills">
        <span class="hdr-pill">🧠 Pattern Learning</span>
        <span class="hdr-pill">🫀 Symptom Mapping</span>
        <span class="hdr-pill">📊 Risk Ranking</span>
        <span class="hdr-pill">⚕️ Clinical Support</span>
      </div>
    </div>
  </div>

  <div class="thought-wrap">
    <div class="thought-track">
      <div class="thought-item">"Early insight saves critical time in care pathways."</div>
      <div class="thought-item">"Responsible AI augments doctors, it never replaces them."</div>
      <div class="thought-item">"Better symptom intelligence leads to better triage outcomes."</div>
      <div class="thought-item">"Data-guided screening can improve consistency in low-resource settings."</div>
      <div class="thought-item">"Early insight saves critical time in care pathways."</div>
      <div class="thought-item">"Responsible AI augments doctors, it never replaces them."</div>
      <div class="thought-item">"Better symptom intelligence leads to better triage outcomes."</div>
      <div class="thought-item">"Data-guided screening can improve consistency in low-resource settings."</div>
    </div>
  </div>

  <svg class="ecg-svg" viewBox="0 0 900 44" preserveAspectRatio="none"
       xmlns="http://www.w3.org/2000/svg">
    <polyline class="ecg-path"
      points="0,22 55,22 75,22 85,3 96,42 110,3 122,42 133,22 158,22
              218,22 238,22 248,3 259,42 273,3 285,42 296,22 321,22
              381,22 401,22 411,3 422,42 436,3 448,42 459,22 484,22
              544,22 564,22 574,3 585,42 599,3 611,42 622,22 647,22
              707,22 727,22 737,3 748,42 762,3 774,42 785,22 810,22 900,22"/>
  </svg>
</div>
""", unsafe_allow_html=True)

# ── STEP INDICATORS ───────────────────────────────────────────────────────────
st.markdown("""
<div class="steps">
  <div class="step-item">
    <div class="step-dot">1</div><div class="step-lbl">Select Symptoms</div>
  </div>
  <div class="step-item">
    <div class="step-dot">2</div><div class="step-lbl">Run Prediction</div>
  </div>
  <div class="step-item">
    <div class="step-dot">3</div><div class="step-lbl">Review Results</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── TWO-COLUMN LAYOUT ─────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

# ═══════════════════════════════════
# LEFT — symptom selection
# ═══════════════════════════════════
with col1:
    st.markdown('<div class="slabel">📋 Step 1</div>', unsafe_allow_html=True)
    st.markdown('<div class="stitle">Select Patient Symptoms</div>', unsafe_allow_html=True)

    for i in range(len(st.session_state.selected_symptoms)):
        others = (
            st.session_state.selected_symptoms[:i]
            + st.session_state.selected_symptoms[i + 1:]
        )
        options = ["Please Select"] + sorted(set(SYMPTOMS) - set(others))
        current = st.session_state.selected_symptoms[i]
        icon = SYMPTOM_ICONS.get(current, "🩹")

        chosen = st.selectbox(
            f"{icon} Symptom {i + 1}",
            options=options,
            index=options.index(current) if current in options else 0,
            key=f"sym_{i}",
        )
        st.session_state.selected_symptoms[i] = chosen

    # "Add symptom" button — styled as ghost via CSS on [data-testid="stButton"]
    # We use a unique key so we can target it if needed, but styling is global
    if len(st.session_state.selected_symptoms) < 17:
        if st.button("＋  Add Another Symptom", key="add_sym"):
            st.session_state.selected_symptoms.append("Please Select")
            st.rerun()

    # Compute valid selections
    final_selected = [
        s for s in st.session_state.selected_symptoms
        if s != "Please Select" and s in SYMPTOMS
    ]

    # Hint
    remaining = max(0, 5 - len(final_selected))
    if remaining > 0:
        st.markdown(
            f'<div class="warn-hint">⚠️ Select at least '
            f'<strong>{remaining}</strong> more symptom(s) to enable prediction.</div>',
            unsafe_allow_html=True,
        )

    # Predict button
    predict_disabled = not (5 <= len(final_selected) <= 17)
    predict_clicked = st.button(
        "🔍  Predict Disease",
        disabled=predict_disabled,
        key="predict_btn",
    )

    if st.button("🧠  Train / Re-train Model", key="retrain_btn"):
        with st.spinner("Training model... this may take around 20-60 seconds."):
            try:
                trained_path, trained_metrics = train_and_save_model(Path("resources/mlp_model.h5"))
                st.session_state.active_model_path = str(trained_path)
                st.session_state.last_train_metrics = trained_metrics
                st.session_state.retrain_status = f"Model re-trained successfully: {trained_path}"
                st.rerun()
            except Exception as exc:
                st.session_state.retrain_status = f"Model training failed: {exc}"

    if st.session_state.retrain_status:
        if st.session_state.retrain_status.startswith("Model re-trained successfully"):
            st.success(st.session_state.retrain_status)
        else:
            st.error(st.session_state.retrain_status)

    if st.session_state.last_train_metrics:
        m = st.session_state.last_train_metrics
        st.markdown(
            f"""
            <div class="warn-hint" style="background:rgba(46,204,113,0.1);border-left-color:#2ECC71;color:#1e6b43;">
              <strong>📈 Latest Training Metrics</strong><br/>
              Train Accuracy: <strong>{m["train_accuracy"] * 100:.2f}%</strong> |
              Validation Accuracy: <strong>{m["val_accuracy"] * 100:.2f}%</strong><br/>
              Train Loss: <strong>{m["train_loss"]:.4f}</strong> |
              Validation Loss: <strong>{m["val_loss"]:.4f}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Health tips
    tips_rows = "".join(
        f'<div class="tip-row"><span class="tip-ico">{ico}</span><span>{tip}</span></div>'
        for ico, tip in HEALTH_TIPS
    )
    st.markdown(
        f'<div class="tips-card"><div class="tips-title">💡  Health Tips</div>{tips_rows}</div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════
# RIGHT — prediction output
# ═══════════════════════════════════
with col2:
    st.markdown('<div class="slabel">📊 Step 2 &amp; 3</div>', unsafe_allow_html=True)
    st.markdown('<div class="stitle">Prediction &amp; Analysis</div>', unsafe_allow_html=True)

    def _placeholder_chart(title: str):
        fig = px.pie(
            names=[title], values=[100],
            color_discrete_sequence=["#d0e4f7"],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title_font_size=13, showlegend=False,
            margin=dict(t=40, b=10, l=10, r=10), height=290,
        )
        fig.update_traces(textinfo="none", hoverinfo="none")
        st.plotly_chart(fig, use_container_width=True)

    # ── Not enough symptoms ──
    if len(final_selected) < 5:
        _placeholder_chart("Select \u2265 5 symptoms to generate a prediction")

    # ── Predict clicked ──
    elif predict_clicked:
        if len(final_selected) > 17:
            st.warning("Maximum 17 symptoms. Using the first 17.")
            final_selected = final_selected[:17]

        with st.spinner("Analysing symptoms with AI model…"):
            # Encode
            encoded = np.zeros(len(SYMPTOMS))
            for s in final_selected:
                if s in SYMPTOMS:
                    encoded[SYMPTOMS.index(s)] = 1

            input_dim = int(model.input_shape[-1])
            inp = np.zeros((1, input_dim))
            inp[0, : min(len(encoded), input_dim)] = encoded[:input_dim]

            predictions = model.predict(inp, verbose=0)

            # Match-score boosts (identical to original logic)
            scores = {}
            for _, row in df.iterrows():
                d_enc = np.array(
                    [1 if sym in row[1:].values else 0 for sym in SYMPTOMS]
                )
                scores[row["Disease"]] = int(np.sum(encoded == d_enc))

            if any(np.array_equal(encoded, df.iloc[i, 1:].values) for i in range(len(df))):
                em = next(
                    df["Disease"].iloc[i] for i in range(len(df))
                    if np.array_equal(encoded, df.iloc[i, 1:].values)
                )
                idx = df[df["Disease"] == em].index[0]
                if idx < len(predictions[0]):
                    predictions[0][idx] *= 2.0
            elif any(v >= 10 for v in scores.values()):
                pm = max(scores, key=scores.get)
                idx = df[df["Disease"] == pm].index[0]
                if idx < len(predictions[0]):
                    predictions[0][idx] *= 1.5
            else:
                bm = max(scores, key=scores.get)
                idx = df[df["Disease"] == bm].index[0]
                if idx < len(predictions[0]):
                    predictions[0][idx] *= 1.2

            predictions = predictions / predictions.sum() * 100
            diseases = df["Disease"].unique()
            pred_df = pd.DataFrame(predictions, columns=diseases).T
            pred_df.columns = ["Probability"]
            pred_df = pred_df.sort_values("Probability", ascending=False)
            top5 = pred_df.head(5).copy()
            top5["Probability"] = top5["Probability"] / top5["Probability"].sum() * 100

        # Primary result card
        top_disease = top5.index[0]
        top_prob = float(top5["Probability"].iloc[0])
        st.markdown(
            f"""
            <div class="result-card">
              <div class="r-label">🩺 Primary Prediction</div>
              <div class="r-name">{top_disease}</div>
              <div class="r-sub">Based on {len(final_selected)} selected symptom(s)</div>
              <div class="pb-wrap">
                <div class="pb-head">
                  <span>Model Confidence</span><span>{top_prob:.1f}%</span>
                </div>
                <div class="pb-track">
                  <div class="pb-fill"
                       style="width:{top_prob:.1f}%;
                              background:linear-gradient(90deg,#2ECC71,#4A90E2);">
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Pie chart
        fig = px.pie(
            top5, values="Probability", names=top5.index,
            title="Top 5 Probable Diseases",
            color_discrete_sequence=BAR_COLORS,
        )
        fig.update_traces(
            textposition="inside", textinfo="percent+label",
            pull=[0.08, 0, 0, 0, 0],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title_font_size=14,
            legend=dict(orientation="v", font=dict(size=11)),
            margin=dict(t=40, b=10, l=10, r=10), height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Per-disease probability bars
        for i, (disease, row) in enumerate(top5.iterrows()):
            pct = float(row["Probability"])
            color = BAR_COLORS[i]
            st.markdown(
                f"""
                <div class="pb-wrap">
                  <div class="pb-head">
                    <span>{disease}</span><span>{pct:.1f}%</span>
                  </div>
                  <div class="pb-track">
                    <div class="pb-fill" style="width:{pct:.1f}%; background:{color};"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Additional disease pills
        rest = pred_df.iloc[5:].index.tolist()
        if rest:
            extra = random.sample(rest, min(6, len(rest)))
            pills = "".join(f'<span class="pill">{d}</span>' for d in extra)
            st.markdown(
                f"""
                <div style="margin-top:20px;">
                  <div class="slabel">Also consider</div>
                  <p style="font-size:0.78rem;color:var(--muted);margin-bottom:8px;">
                    These may warrant further clinical investigation:
                  </p>
                  <div class="pill-wrap">{pills}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Valid symptoms but not clicked yet ──
    else:
        _placeholder_chart("Ready — click Predict")
        st.markdown(
            '<p style="text-align:center;color:#6b8299;font-size:0.88rem;">'
            "Click <strong>🔍 Predict Disease</strong> to generate results.</p>",
            unsafe_allow_html=True,
        )

# ── DISCLAIMER ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="disc">
      <strong>⚠️ Clinical Disclaimer:</strong>
      These AI-generated predictions are intended solely as a
      <strong>decision-support aid</strong> for trained healthcare providers.
      They do not constitute a definitive diagnosis. All clinical decisions must be
      grounded in comprehensive patient evaluation, laboratory investigations, and
      professional medical judgment. Data sourced from CDC research studies.
    </div>
    """,
    unsafe_allow_html=True,
)
