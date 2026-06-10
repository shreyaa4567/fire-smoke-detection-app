"""Fire & Smoke Detection System
=============================
A real-time fire and smoke detection application built with YOLOv8 and Streamlit.
Features five analysis modes: image upload, quick test samples, video upload with
annotated re-encoding, webcam capture, and an auto-cycling demo mode.

Author: Shreya Singh
License: MIT
Repository: https://github.com/shreyaa4567/fire-smoke-detection-app
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
import os
import io
from datetime import datetime

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ═══════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════

MODEL_PATH = "best_shreya_v2.pt"
CONFIDENCE_THRESHOLD = 0.15
IOU_THRESHOLD = 0.45
DEMO_INTERVAL_SECONDS = 2
DEMO_IMAGES_DIR = "demo_images"
SAMPLE_IMAGES_DIR = "samples"
SUPPORTED_IMAGE_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")
SUPPORTED_VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv"]

# ═══════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════

st.set_page_config(
    page_title="Fire & Smoke Detection",
    page_icon="🔥",
    layout="wide"
)

# ═══════════════════════════════════════════
# CUSTOM CSS — Premium Dark Industrial Dashboard
# ═══════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #080808;
        color: #e8e8e8;
    }

    /* ── HEADER ── */
    .header-container {
        background: linear-gradient(160deg, #140500 0%, #1f0800 40%, #140500 100%);
        border: 1px solid rgba(255, 80, 0, 0.35);
        border-radius: 20px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.75rem;
        box-shadow: 0 0 60px rgba(255, 60, 0, 0.1), inset 0 1px 0 rgba(255,255,255,0.03);
        text-align: center;
    }

    .header-eyebrow {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: rgba(255, 100, 40, 0.6);
        margin-bottom: 0.5rem;
    }

    .header-title {
        font-size: 1.9rem;
        font-weight: 700;
        color: #ff6535;
        margin: 0;
        letter-spacing: -0.8px;
        line-height: 1.1;
    }

    .header-subtitle {
        font-size: 0.88rem;
        color: #666;
        margin-top: 0.4rem;
        font-weight: 400;
    }

    .header-badges {
        margin-top: 0.9rem;
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        flex-wrap: wrap;
    }

    .badge {
        display: inline-block;
        background: rgba(255, 69, 0, 0.1);
        border: 1px solid rgba(255, 69, 0, 0.25);
        color: #cc5522;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.2rem 0.65rem;
        border-radius: 20px;
        letter-spacing: 0.5px;
        font-family: 'JetBrains Mono', monospace;
    }

    .badge.green {
        background: rgba(34, 197, 94, 0.08);
        border-color: rgba(34, 197, 94, 0.2);
        color: #4ade80;
    }

    /* ── INFO CARDS ── */
    .info-card {
        background: #111;
        border: 1px solid #1e1e1e;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
        transition: border-color 0.2s, box-shadow 0.2s;
        position: relative;
        overflow: hidden;
    }

    .info-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255,80,0,0.4), transparent);
    }

    .info-card:hover {
        border-color: rgba(255, 80, 0, 0.2);
        box-shadow: 0 4px 20px rgba(255, 60, 0, 0.05);
    }

    .info-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ff6535;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
    }

    .info-label {
        font-size: 0.7rem;
        color: #555;
        margin-top: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* ── IMAGE CONTAINERS ── */
    .img-container {
        background: #0d0d0d;
        border: 1px solid #1a1a1a;
        border-radius: 14px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }

    .img-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.6rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #1a1a1a;
    }

    .img-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #ff4500;
    }

    .img-dot.grey { background: #444; }

    .img-title {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #555;
    }

    .img-caption {
        font-size: 0.72rem;
        color: #333;
        text-align: center;
        margin-top: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── STAT CARDS ── */
    .stat-card {
        background: #0e0e0e;
        border: 1px solid #1c1c1c;
        border-radius: 14px;
        padding: 1.1rem 1rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: transform 0.15s, box-shadow 0.15s;
    }

    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    }

    .stat-card.fire::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #ff4500, #ff8c00);
    }

    .stat-card.smoke::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #555, #888);
    }

    .stat-card.total::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #7c3aed, #a855f7);
    }

    .stat-card.conf::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #0ea5e9, #38bdf8);
    }

    .stat-icon {
        font-size: 1.4rem;
        margin-bottom: 0.4rem;
        line-height: 1;
    }

    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f0f0f0;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
    }

    .stat-label {
        font-size: 0.68rem;
        color: #444;
        margin-top: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* ── DETECTION LIST ── */
    .det-header {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #444;
        margin-bottom: 0.6rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #181818;
    }

    .det-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        background: #0e0e0e;
        border: 1px solid #1a1a1a;
        border-left: 3px solid #ff4500;
        border-radius: 0 10px 10px 0;
        padding: 0.55rem 0.9rem;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
        color: #ccc;
        transition: background 0.15s;
    }

    .det-item:hover { background: #141414; }

    .det-item.smoke { border-left-color: #666; }

    .det-conf {
        margin-left: auto;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #555;
    }

    .det-empty {
        background: #0e0e0e;
        border: 1px dashed #1e1e1e;
        border-radius: 10px;
        padding: 1.25rem;
        text-align: center;
        color: #333;
        font-size: 0.85rem;
    }

    /* ── RISK + ACTION PANEL ── */
    .risk-panel {
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
        text-align: center;
    }

    .risk-panel.high {
        background: rgba(220, 38, 38, 0.08);
        border: 1px solid rgba(220, 38, 38, 0.3);
    }

    .risk-panel.medium {
        background: rgba(234, 88, 12, 0.08);
        border: 1px solid rgba(234, 88, 12, 0.3);
    }

    .risk-panel.low {
        background: rgba(22, 163, 74, 0.08);
        border: 1px solid rgba(22, 163, 74, 0.3);
    }

    .risk-panel.safe {
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    .risk-indicator {
        font-size: 2rem;
        line-height: 1;
        margin-bottom: 0.4rem;
    }

    .risk-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .risk-panel.high .risk-label { color: #f87171; }
    .risk-panel.medium .risk-label { color: #fb923c; }
    .risk-panel.low .risk-label { color: #4ade80; }
    .risk-panel.safe .risk-label { color: #60a5fa; }

    .risk-conf {
        font-size: 0.78rem;
        color: #444;
        font-family: 'JetBrains Mono', monospace;
    }

    .action-panel {
        border-radius: 12px;
        padding: 1rem 1.25rem;
        border: 1px solid #1e1e1e;
        background: #0c0c0c;
    }

    .action-title {
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #444;
        margin-bottom: 0.6rem;
    }

    .action-text {
        font-size: 0.9rem;
        font-weight: 500;
        color: #bbb;
    }

    /* ── EMPTY STATE ── */
    .empty-state {
        background: #0a0a0a;
        border: 1px dashed #1e1e1e;
        border-radius: 20px;
        padding: 5rem 2rem;
        text-align: center;
        margin-top: 1rem;
    }

    .empty-icon {
        font-size: 3.5rem;
        margin-bottom: 1.25rem;
        opacity: 0.3;
    }

    .empty-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.4rem;
    }

    .empty-sub {
        font-size: 0.82rem;
        color: #2a2a2a;
    }

    .empty-formats {
        display: flex;
        justify-content: center;
        gap: 0.4rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }

    .fmt-tag {
        background: #111;
        border: 1px solid #1e1e1e;
        color: #333;
        font-size: 0.68rem;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── SECTION LABEL ── */
    .section-label {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #444;
        margin-bottom: 0.65rem;
    }

    /* ── DIVIDER ── */
    hr { border-color: #141414; margin: 1.5rem 0; }

    /* ── HIDE STREAMLIT UI ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* ── MODE SELECTOR ── */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        gap: 0.35rem;
        justify-content: center;
    }

    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: #111 !important;
        border: 1px solid #1e1e1e !important;
        border-radius: 10px !important;
        padding: 0.55rem 1rem !important;
        font-size: 0.82rem !important;
        color: #666 !important;
        transition: all 0.2s !important;
        cursor: pointer;
    }

    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        border-color: rgba(255, 80, 0, 0.25) !important;
        color: #999 !important;
    }

    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] {
        background: rgba(255, 69, 0, 0.1) !important;
        border-color: rgba(255, 69, 0, 0.35) !important;
        color: #ff6535 !important;
    }

    /* ── DEMO STATUS ── */
    .demo-status {
        background: rgba(255, 69, 0, 0.08);
        border: 1px solid rgba(255, 69, 0, 0.2);
        border-radius: 10px;
        padding: 0.65rem 1.2rem;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #ff6535;
        margin-bottom: 1rem;
        animation: pulse-border 2s ease-in-out infinite;
    }

    @keyframes pulse-border {
        0%, 100% { border-color: rgba(255, 69, 0, 0.2); }
        50% { border-color: rgba(255, 69, 0, 0.5); }
    }

    /* ── DETECTION TIMELINE ── */
    .timeline-container {
        background: #0d0d0d;
        border: 1px solid #1a1a1a;
        border-radius: 14px;
        padding: 1rem;
        max-height: 350px;
        overflow-y: auto;
    }

    .timeline-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.45rem 0.7rem;
        border-left: 2px solid #222;
        margin-left: 0.5rem;
        margin-bottom: 0.2rem;
        font-size: 0.8rem;
        color: #888;
        transition: background 0.15s;
    }

    .timeline-item:hover { background: rgba(255, 69, 0, 0.04); }
    .timeline-item.fire { border-left-color: #ff4500; }
    .timeline-item.smoke { border-left-color: #666; }

    .timeline-time {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #555;
        min-width: 55px;
    }

    .timeline-label {
        font-weight: 600;
        color: #ccc;
    }

    .timeline-conf {
        margin-left: auto;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #444;
    }

    /* ── FOOTER ── */
    .footer-container {
        background: linear-gradient(160deg, #0a0a0a 0%, #111 50%, #0a0a0a 100%);
        border: 1px solid #1a1a1a;
        border-radius: 16px;
        padding: 2rem;
        margin-top: 3rem;
        text-align: center;
    }

    .footer-title {
        font-size: 1rem;
        font-weight: 600;
        color: #ff6535;
        margin-bottom: 0.3rem;
    }

    .footer-subtitle {
        font-size: 0.78rem;
        color: #444;
        margin-bottom: 1rem;
    }

    .footer-stats {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }

    .footer-stat {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #555;
        background: #0e0e0e;
        border: 1px solid #1a1a1a;
        padding: 0.3rem 0.7rem;
        border-radius: 8px;
    }

    .footer-author {
        font-size: 0.78rem;
        color: #333;
        margin-top: 0.5rem;
    }

    .footer-author a {
        color: #ff6535;
        text-decoration: none;
    }

    .footer-author a:hover {
        text-decoration: underline;
    }

    /* ── VIDEO CONTAINER ── */
    .video-container {
        background: #0d0d0d;
        border: 1px solid #1a1a1a;
        border-radius: 14px;
        padding: 0.75rem;
        margin-bottom: 1rem;
    }

    /* ── SAMPLE GRID ── */
    .sample-hint {
        font-size: 0.8rem;
        color: #444;
        margin-bottom: 1rem;
        text-align: center;
    }

</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════

st.markdown("""
<div class="header-container">
    <div class="header-eyebrow">Industrial Safety · AI Vision System</div>
    <div class="header-title">🔥 Fire &amp; Smoke Detection</div>
    <div class="header-subtitle">Real-time detection powered by YOLOv8 deep learning</div>
    <div class="header-badges">
        <span class="badge">YOLOv8n</span>
        <span class="badge">Precision 0.760</span>
        <span class="badge">Recall 0.718</span>
        <span class="badge">mAP50 0.776</span>
        <span class="badge green">● Live Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════

@st.cache_resource
def load_model():
    """Load and cache the YOLOv8 model to avoid reloading on every Streamlit rerun."""
    return YOLO(MODEL_PATH)

model = load_model()

# ═══════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════

if "demo_running" not in st.session_state:
    st.session_state.demo_running = False
if "demo_index" not in st.session_state:
    st.session_state.demo_index = 0
if "demo_timeline" not in st.session_state:
    st.session_state.demo_timeline = []
if "demo_last_logged" not in st.session_state:
    st.session_state.demo_last_logged = -1
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "current_mode" not in st.session_state:
    st.session_state.current_mode = None

# ═══════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════

def get_sample_icon(name):
    """Map a sample image filename to an appropriate emoji icon.

    Scans the filename for keywords (fire, smoke, forest, safe, etc.)
    and returns a matching emoji. Falls back to a camera icon.

    Args:
        name: The filename (without extension) to analyze.

    Returns:
        A single emoji string representing the detected category.
    """
    n = name.lower()
    if any(w in n for w in ["fire", "wildfire", "flame", "blaze"]):
        return "🔥"
    elif any(w in n for w in ["smoke", "industrial", "factory", "chimney"]):
        return "🏭"
    elif any(w in n for w in ["forest", "tree", "bush", "wood"]):
        return "🌲"
    elif any(w in n for w in ["safe", "clear", "normal", "clean", "no_fire"]):
        return "✅"
    elif any(w in n for w in ["car", "vehicle", "road"]):
        return "🚗"
    elif any(w in n for w in ["building", "house", "room", "indoor"]):
        return "🏢"
    return "📸"


def get_image_files(directory):
    """Get a sorted list of supported image files from a directory.

    Args:
        directory: Path to the directory to scan.

    Returns:
        A sorted list of absolute file paths for supported image formats.
        Returns an empty list if the directory does not exist.
    """
    if not os.path.exists(directory):
        return []
    files = [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.lower().endswith(SUPPORTED_IMAGE_FORMATS) and not f.startswith(".")
    ]
    return files


# ═══════════════════════════════════════════
# REUSABLE: RENDER DETECTION RESULTS
# ═══════════════════════════════════════════

def render_detection_results(image, results, inference_time_ms, key_prefix="det"):
    """
    Render the complete detection results UI for a single image.

    Args:
        image: PIL Image (RGB).
        results: YOLO prediction results.
        inference_time_ms: Inference time in milliseconds.
        key_prefix: Unique prefix for Streamlit widget keys.

    Returns:
        dict with fire_count, smoke_count, max_confidence, avg_confidence, detection_items.
    """

    # Ensure PIL Image
    if isinstance(image, np.ndarray):
        if CV2_AVAILABLE and len(image.shape) == 3 and image.shape[2] == 3:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image = Image.fromarray(image)
    if image.mode != "RGB":
        image = image.convert("RGB")

    result_image = results[0].plot()
    boxes = results[0].boxes

    # ── PARSE DETECTIONS ──
    fire_count = 0
    smoke_count = 0
    max_confidence = 0.0
    total_conf = 0.0
    detection_items = []

    if len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[cls_id]
            max_confidence = max(max_confidence, confidence)
            total_conf += confidence
            if label.lower() == "fire":
                fire_count += 1
            elif label.lower() == "smoke":
                smoke_count += 1
            detection_items.append((label, confidence))

    total_detections = fire_count + smoke_count
    avg_confidence = (total_conf / total_detections) if total_detections > 0 else 0.0

    # ── DETERMINE RISK ──
    if max_confidence >= 0.80:
        risk_class, risk_icon, risk_text = "high", "🚨", "HIGH RISK"
        risk_card = "🔴 HIGH"
        action_icon, action_text = "🚨", "Emergency Response Recommended"
    elif max_confidence >= 0.50:
        risk_class, risk_icon, risk_text = "medium", "⚠️", "MEDIUM RISK"
        risk_card = "🟠 MEDIUM"
        action_icon, action_text = "⚠️", "Inspect Area Immediately"
    elif max_confidence > 0:
        risk_class, risk_icon, risk_text = "low", "👁️", "LOW RISK"
        risk_card = "🟢 LOW"
        action_icon, action_text = "👀", "Continue Monitoring"
    else:
        risk_class, risk_icon, risk_text = "safe", "✅", "ALL CLEAR"
        risk_card = "🔵 CLEAR"
        action_icon, action_text = "✅", "No Action Required"

    # ── INFO CARDS (Inference Time | Detections | Confidence | Risk) ──
    i1, i2, i3, i4 = st.columns(4)
    conf_pct = f"{max_confidence * 100:.1f}%" if max_confidence > 0 else "—"

    with i1:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-value">{inference_time_ms:.0f}ms</div>
            <div class="info-label">Inference Time</div>
        </div>""", unsafe_allow_html=True)

    with i2:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-value">{total_detections}</div>
            <div class="info-label">Detections Found</div>
        </div>""", unsafe_allow_html=True)

    with i3:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-value">{conf_pct}</div>
            <div class="info-label">Highest Confidence</div>
        </div>""", unsafe_allow_html=True)

    with i4:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-value">{risk_card}</div>
            <div class="info-label">Risk Level</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── IMAGES SIDE BY SIDE ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="img-container">
            <div class="img-header">
                <div class="img-dot grey"></div>
                <div class="img-title">Original Image</div>
            </div>
        """, unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown(
            f'<div class="img-caption">Input · {image.size[0]} × {image.size[1]} px</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="img-container">
            <div class="img-header">
                <div class="img-dot"></div>
                <div class="img-title">Detection Result</div>
            </div>
        """, unsafe_allow_html=True)
        st.image(result_image, use_container_width=True)
        st.markdown(
            f'<div class="img-caption">YOLOv8n · conf={CONFIDENCE_THRESHOLD} · iou={IOU_THRESHOLD}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── STAT CARDS ──
    st.markdown('<div class="section-label">Detection Summary</div>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.markdown(f"""
        <div class="stat-card fire">
            <div class="stat-icon">🔥</div>
            <div class="stat-value">{fire_count}</div>
            <div class="stat-label">Fire Regions</div>
        </div>""", unsafe_allow_html=True)

    with s2:
        st.markdown(f"""
        <div class="stat-card smoke">
            <div class="stat-icon">🌫</div>
            <div class="stat-value">{smoke_count}</div>
            <div class="stat-label">Smoke Regions</div>
        </div>""", unsafe_allow_html=True)

    with s3:
        st.markdown(f"""
        <div class="stat-card total">
            <div class="stat-icon">📦</div>
            <div class="stat-value">{total_detections}</div>
            <div class="stat-label">Total Detections</div>
        </div>""", unsafe_allow_html=True)

    with s4:
        st.markdown(f"""
        <div class="stat-card conf">
            <div class="stat-icon">⚡</div>
            <div class="stat-value">{conf_pct}</div>
            <div class="stat-label">Max Confidence</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── DETECTIONS LIST + RISK PANEL ──
    det_col, right_col = st.columns([1.6, 1])

    with det_col:
        st.markdown('<div class="det-header">Detected Objects</div>', unsafe_allow_html=True)
        if len(detection_items) == 0:
            st.markdown(
                '<div class="det-empty">No fire or smoke detected in this image</div>',
                unsafe_allow_html=True,
            )
        else:
            for label, conf in detection_items:
                css = "det-item smoke" if label.lower() == "smoke" else "det-item"
                icon = "🌫" if label.lower() == "smoke" else "🔥"
                st.markdown(
                    f'<div class="{css}">{icon} <strong>{label.upper()}</strong>'
                    f'<span class="det-conf">{conf * 100:.1f}%</span></div>',
                    unsafe_allow_html=True,
                )

    with right_col:
        conf_str = (
            f"{max_confidence * 100:.1f}% confidence"
            if max_confidence > 0
            else "No detections"
        )
        st.markdown(f"""
        <div class="risk-panel {risk_class}">
            <div class="risk-indicator">{risk_icon}</div>
            <div class="risk-label">{risk_text}</div>
            <div class="risk-conf">{conf_str}</div>
        </div>
        <div class="action-panel">
            <div class="action-title">Recommended Action</div>
            <div class="action-text">{action_icon} {action_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── DOWNLOAD ──
    result_pil = Image.fromarray(result_image)
    buf = io.BytesIO()
    result_pil.save(buf, format="JPEG")
    buf.seek(0)
    st.download_button(
        label="⬇ Download Detection Result",
        data=buf.getvalue(),
        file_name="fire_smoke_result.jpg",
        mime="image/jpeg",
        use_container_width=True,
        key=f"{key_prefix}_download",
    )

    return {
        "detection_items": detection_items,
        "fire_count": fire_count,
        "smoke_count": smoke_count,
        "max_confidence": max_confidence,
        "avg_confidence": avg_confidence,
    }


# ═══════════════════════════════════════════
# REUSABLE: RENDER DETECTION TIMELINE
# ═══════════════════════════════════════════

def render_timeline(timeline_entries, max_entries=30):
    """Render a scrollable, color-coded detection timeline.

    Displays timestamped detection entries with fire/smoke icons and
    confidence values. Limits display to the most recent entries.

    Args:
        timeline_entries: List of dicts with 'time', 'label', and 'confidence' keys.
        max_entries: Maximum number of entries to display (default: 30).
    """
    if not timeline_entries:
        return

    st.markdown('<div class="section-label">Detection Timeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="timeline-container">', unsafe_allow_html=True)

    entries = timeline_entries[-max_entries:]
    for entry in entries:
        icon = "🔥" if entry["label"].lower() == "fire" else "🌫"
        css_class = "fire" if entry["label"].lower() == "fire" else "smoke"
        st.markdown(f"""
        <div class="timeline-item {css_class}">
            <span class="timeline-time">{entry['time']}</span>
            {icon} <span class="timeline-label">{entry['label'].upper()}</span>
            <span class="timeline-conf">{entry['confidence'] * 100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# REUSABLE: RENDER DETECTION STATISTICS
# ═══════════════════════════════════════════

def render_statistics(fire_total, smoke_total, avg_conf, peak_conf):
    """Render an aggregate detection statistics panel.

    Displays four stat cards showing total fire detections, total smoke
    detections, average confidence, and peak confidence across all
    processed frames or images.

    Args:
        fire_total: Total number of fire detections.
        smoke_total: Total number of smoke detections.
        avg_conf: Average confidence percentage (0-100).
        peak_conf: Peak confidence percentage (0-100).
    """
    st.markdown('<div class="section-label">Detection Statistics</div>', unsafe_allow_html=True)

    v1, v2, v3, v4 = st.columns(4)
    with v1:
        st.markdown(f"""
        <div class="stat-card fire">
            <div class="stat-icon">🔥</div>
            <div class="stat-value">{fire_total}</div>
            <div class="stat-label">Fire Detections</div>
        </div>""", unsafe_allow_html=True)

    with v2:
        st.markdown(f"""
        <div class="stat-card smoke">
            <div class="stat-icon">🌫</div>
            <div class="stat-value">{smoke_total}</div>
            <div class="stat-label">Smoke Detections</div>
        </div>""", unsafe_allow_html=True)

    with v3:
        st.markdown(f"""
        <div class="stat-card conf">
            <div class="stat-icon">📊</div>
            <div class="stat-value">{avg_conf:.1f}%</div>
            <div class="stat-label">Avg Confidence</div>
        </div>""", unsafe_allow_html=True)

    with v4:
        st.markdown(f"""
        <div class="stat-card total">
            <div class="stat-icon">⚡</div>
            <div class="stat-value">{peak_conf:.1f}%</div>
            <div class="stat-label">Peak Confidence</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# MODE SELECTOR
# ═══════════════════════════════════════════

MODES = [
    "📷 Image Upload",
    "🔥 Quick Test",
    "🎥 Video Upload",
    "📹 Webcam",
    "🎬 Demo Mode",
]


def on_mode_change():
    """Callback to stop demo mode when the user switches to a different mode."""
    st.session_state.demo_running = False


mode = st.radio(
    "mode_selector",
    options=MODES,
    horizontal=True,
    label_visibility="collapsed",
    on_change=on_mode_change,
)

# Track mode changes for cleanup
if st.session_state.current_mode != mode:
    st.session_state.current_mode = mode
    if mode != "🎬 Demo Mode":
        st.session_state.demo_running = False

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# TAB 1: IMAGE UPLOAD
# ═══════════════════════════════════════════

if mode == "📷 Image Upload":

    st.markdown('<div class="section-label">Upload Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag and drop or browse",
        type=[ext.lstrip(".") for ext in SUPPORTED_IMAGE_FORMATS],
        label_visibility="collapsed",
        key="image_uploader",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        if image.mode == "RGBA":
            image = image.convert("RGB")

        with st.spinner("Analyzing..."):
            start_time = time.time()
            results = model.predict(image, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
            inference_ms = (time.time() - start_time) * 1000

        render_detection_results(image, results, inference_ms, key_prefix="upload")

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🔍</div>
            <div class="empty-title">No Image Loaded</div>
            <div class="empty-sub">Upload an image above to run fire and smoke detection</div>
            <div class="empty-formats">
                <span class="fmt-tag">.jpg</span>
                <span class="fmt-tag">.png</span>
                <span class="fmt-tag">.bmp</span>
                <span class="fmt-tag">.webp</span>
                <span class="fmt-tag">.tiff</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# TAB 2: QUICK TEST SAMPLES
# ═══════════════════════════════════════════

elif mode == "🔥 Quick Test":

    st.markdown('<div class="section-label">Quick Test Samples</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sample-hint">Click any sample below to instantly test the model — no upload needed</div>',
        unsafe_allow_html=True,
    )

    sample_dir = SAMPLE_IMAGES_DIR
    sample_files = get_image_files(sample_dir)

    if not sample_files:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📁</div>
            <div class="empty-title">No Sample Images Found</div>
            <div class="empty-sub">Add images to the <code>samples/</code> folder to enable Quick Test</div>
            <div class="empty-formats">
                <span class="fmt-tag">wildfire.jpg</span>
                <span class="fmt-tag">industrial_smoke.jpg</span>
                <span class="fmt-tag">forest_fire.jpg</span>
                <span class="fmt-tag">safe_scene.jpg</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Build button grid (up to 4 per row)
        num_cols = min(len(sample_files), 4)
        cols = st.columns(num_cols)
        for i, fpath in enumerate(sample_files):
            fname = os.path.splitext(os.path.basename(fpath))[0]
            display_name = fname.replace("_", " ").replace("-", " ").title()
            icon = get_sample_icon(fname)
            with cols[i % num_cols]:
                if st.button(
                    f"{icon} {display_name}",
                    use_container_width=True,
                    key=f"sample_btn_{i}",
                ):
                    st.session_state.selected_sample = fpath

        # Process & display selected sample
        selected = st.session_state.get("selected_sample")
        if selected and os.path.exists(selected):
            st.markdown("<br>", unsafe_allow_html=True)
            image = Image.open(selected)
            if image.mode == "RGBA":
                image = image.convert("RGB")

            with st.spinner("Analyzing..."):
                start_time = time.time()
                results = model.predict(image, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
                inference_ms = (time.time() - start_time) * 1000

            render_detection_results(image, results, inference_ms, key_prefix="sample")


# ═══════════════════════════════════════════
# TAB 3: VIDEO UPLOAD
# ═══════════════════════════════════════════

elif mode == "🎥 Video Upload":

    if not CV2_AVAILABLE:
        st.error("📦 OpenCV is required for video processing.")
        st.code("pip install opencv-python", language="bash")
    else:
        st.markdown('<div class="section-label">Upload Video File</div>', unsafe_allow_html=True)
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=SUPPORTED_VIDEO_FORMATS,
            label_visibility="collapsed",
            key="video_uploader",
        )

        if uploaded_video is not None:
            # Detect if this is a new video
            vid_id = f"{uploaded_video.name}_{uploaded_video.size}"
            if st.session_state.video_id != vid_id:
                st.session_state.video_processed = False
                st.session_state.video_id = vid_id

            if not st.session_state.video_processed:
                # Show preview
                st.markdown(
                    '<div class="section-label">Video Preview</div>',
                    unsafe_allow_html=True,
                )
                st.video(uploaded_video)

                if st.button("🔍 Process Video", use_container_width=True, key="process_video_btn"):
                    # ── VIDEO PROCESSING PIPELINE ──
                    import tempfile

                    progress_bar = st.progress(0, text="Initializing video processing...")
                    status_area = st.empty()

                    # Save to temp file
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(uploaded_video.read())
                    tfile.close()

                    cap = cv2.VideoCapture(tfile.name)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    # Output video
                    out_path = os.path.join(
                        tempfile.gettempdir(), "fire_smoke_detected.mp4"
                    )
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

                    fire_total = 0
                    smoke_total = 0
                    all_confs = []
                    timeline = []
                    frame_count = 0
                    process_start = time.time()

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Convert BGR -> RGB for YOLO, then back for output
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = model.predict(
                            frame_rgb, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False
                        )
                        annotated_rgb = results[0].plot()
                        annotated_bgr = cv2.cvtColor(
                            annotated_rgb, cv2.COLOR_RGB2BGR
                        )
                        out.write(annotated_bgr)

                        # Aggregate stats
                        for box in results[0].boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            label = model.names[cls_id]
                            all_confs.append(conf)
                            if label.lower() == "fire":
                                fire_total += 1
                            elif label.lower() == "smoke":
                                smoke_total += 1

                        # Timeline: log once per second (highest conf detection)
                        if (
                            frame_count % max(1, int(fps)) == 0
                            and len(results[0].boxes) > 0
                        ):
                            ts = frame_count / fps
                            mins = int(ts // 60)
                            secs = int(ts % 60)
                            best_conf = 0
                            best_label = ""
                            for box in results[0].boxes:
                                c = float(box.conf[0])
                                if c > best_conf:
                                    best_conf = c
                                    best_label = model.names[int(box.cls[0])]
                            timeline.append(
                                {
                                    "time": f"{mins:02d}:{secs:02d}",
                                    "label": best_label,
                                    "confidence": best_conf,
                                }
                            )

                        frame_count += 1
                        pct = min(frame_count / max(total_frames, 1), 1.0)
                        progress_bar.progress(
                            pct,
                            text=f"Processing frame {frame_count}/{total_frames}",
                        )

                    cap.release()
                    out.release()
                    process_time = time.time() - process_start

                    # Read output video bytes
                    with open(out_path, "rb") as f:
                        video_bytes = f.read()

                    # Clean up temp files
                    try:
                        os.unlink(tfile.name)
                        os.unlink(out_path)
                    except OSError:
                        pass

                    # Store results in session state
                    st.session_state.video_processed = True
                    st.session_state.video_bytes = video_bytes
                    st.session_state.video_fire_total = fire_total
                    st.session_state.video_smoke_total = smoke_total
                    st.session_state.video_all_confs = all_confs
                    st.session_state.video_timeline = timeline
                    st.session_state.video_total_frames = frame_count
                    st.session_state.video_fps = fps
                    st.session_state.video_process_time = process_time

                    progress_bar.empty()
                    status_area.empty()
                    st.rerun()

            # ── DISPLAY PROCESSED VIDEO RESULTS ──
            if st.session_state.video_processed:
                # Annotated video playback
                st.markdown(
                    '<div class="section-label">Annotated Video Output</div>',
                    unsafe_allow_html=True,
                )
                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                st.video(st.session_state.video_bytes)
                st.markdown("</div>", unsafe_allow_html=True)

                # Processing info cards
                total_frames = st.session_state.video_total_frames
                proc_time = st.session_state.video_process_time
                vid_fps = st.session_state.video_fps
                fire_t = st.session_state.video_fire_total
                smoke_t = st.session_state.video_smoke_total
                confs = st.session_state.video_all_confs

                p1, p2, p3, p4 = st.columns(4)
                with p1:
                    st.markdown(f"""
                    <div class="info-card">
                        <div class="info-value">{total_frames}</div>
                        <div class="info-label">Frames Processed</div>
                    </div>""", unsafe_allow_html=True)
                with p2:
                    st.markdown(f"""
                    <div class="info-card">
                        <div class="info-value">{proc_time:.1f}s</div>
                        <div class="info-label">Processing Time</div>
                    </div>""", unsafe_allow_html=True)
                with p3:
                    st.markdown(f"""
                    <div class="info-card">
                        <div class="info-value">{vid_fps:.0f}</div>
                        <div class="info-label">Original FPS</div>
                    </div>""", unsafe_allow_html=True)
                with p4:
                    dur = total_frames / vid_fps if vid_fps > 0 else 0
                    st.markdown(f"""
                    <div class="info-card">
                        <div class="info-value">{dur:.1f}s</div>
                        <div class="info-label">Duration</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Detection Statistics
                avg_c = (sum(confs) / len(confs) * 100) if confs else 0
                peak_c = max(confs) * 100 if confs else 0
                render_statistics(fire_t, smoke_t, avg_c, peak_c)

                st.markdown("<br>", unsafe_allow_html=True)

                # Detection Timeline
                render_timeline(st.session_state.video_timeline)

                st.markdown("<hr>", unsafe_allow_html=True)

                # Download annotated video
                st.download_button(
                    label="⬇ Download Annotated Video",
                    data=st.session_state.video_bytes,
                    file_name=f"detected_{uploaded_video.name}",
                    mime="video/mp4",
                    use_container_width=True,
                    key="video_download",
                )

        else:
            st.session_state.video_processed = False
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🎥</div>
                <div class="empty-title">No Video Loaded</div>
                <div class="empty-sub">Upload a video file to run fire and smoke detection on every frame</div>
                <div class="empty-formats">
                    <span class="fmt-tag">.mp4</span>
                    <span class="fmt-tag">.avi</span>
                    <span class="fmt-tag">.mov</span>
                    <span class="fmt-tag">.mkv</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# TAB 4: WEBCAM
# ═══════════════════════════════════════════

elif mode == "📹 Webcam":

    st.markdown(
        '<div class="section-label">Webcam Capture</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<p style="color:#444; font-size:0.82rem; margin-bottom:1rem;">'
        "Capture a photo using your webcam and run instant fire & smoke detection</p>",
        unsafe_allow_html=True,
    )

    camera_photo = st.camera_input("Take a picture", key="webcam_input")

    if camera_photo is not None:
        image = Image.open(camera_photo)
        if image.mode == "RGBA":
            image = image.convert("RGB")

        with st.spinner("Analyzing..."):
            start_time = time.time()
            results = model.predict(image, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
            inference_ms = (time.time() - start_time) * 1000

        render_detection_results(image, results, inference_ms, key_prefix="webcam")

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📹</div>
            <div class="empty-title">Webcam Ready</div>
            <div class="empty-sub">Click "Take Photo" above to capture and analyze</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# TAB 5: DEMO MODE
# ═══════════════════════════════════════════

elif mode == "🎬 Demo Mode":

    demo_dir = DEMO_IMAGES_DIR
    demo_files = get_image_files(demo_dir)

    if not demo_files:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🎬</div>
            <div class="empty-title">No Demo Images Found</div>
            <div class="empty-sub">Add fire and smoke images to the <code>demo_images/</code> folder to enable Demo Mode</div>
            <div class="empty-formats">
                <span class="fmt-tag">.jpg</span>
                <span class="fmt-tag">.png</span>
                <span class="fmt-tag">.bmp</span>
                <span class="fmt-tag">.webp</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── CONTROLS ──
        ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])

        with ctrl1:
            if st.button(
                "▶ Run Demo",
                use_container_width=True,
                key="start_demo",
                disabled=st.session_state.demo_running,
            ):
                st.session_state.demo_running = True
                st.session_state.demo_index = 0
                st.session_state.demo_timeline = []
                st.session_state.demo_last_logged = -1
                st.rerun()

        with ctrl2:
            if st.button(
                "⏹ Stop Demo",
                use_container_width=True,
                key="stop_demo",
                disabled=not st.session_state.demo_running,
            ):
                st.session_state.demo_running = False
                st.rerun()

        with ctrl3:
            if st.session_state.demo_running:
                idx = st.session_state.demo_index % len(demo_files)
                st.markdown(
                    f'<div class="demo-status">🎬 Demo Image {idx + 1} / {len(demo_files)}</div>',
                    unsafe_allow_html=True,
                )

        # ── DISPLAY CURRENT DEMO IMAGE ──
        if st.session_state.demo_running:
            idx = st.session_state.demo_index % len(demo_files)
            current_file = demo_files[idx]

            image = Image.open(current_file)
            if image.mode == "RGBA":
                image = image.convert("RGB")

            start_time = time.time()
            results = model.predict(image, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
            inference_ms = (time.time() - start_time) * 1000

            det_result = render_detection_results(
                image, results, inference_ms, key_prefix=f"demo_{idx}"
            )

            # Log to timeline (only once per index)
            abs_idx = st.session_state.demo_index
            if abs_idx != st.session_state.demo_last_logged:
                st.session_state.demo_last_logged = abs_idx
                now = datetime.now().strftime("%H:%M:%S")
                for label, conf in det_result["detection_items"]:
                    st.session_state.demo_timeline.append(
                        {"time": now, "label": label, "confidence": conf}
                    )

            # Show aggregate stats for all processed demo images
            if st.session_state.demo_timeline:
                st.markdown("<br>", unsafe_allow_html=True)

                fire_t = sum(
                    1
                    for e in st.session_state.demo_timeline
                    if e["label"].lower() == "fire"
                )
                smoke_t = sum(
                    1
                    for e in st.session_state.demo_timeline
                    if e["label"].lower() == "smoke"
                )
                all_c = [e["confidence"] for e in st.session_state.demo_timeline]
                avg_c = (sum(all_c) / len(all_c) * 100) if all_c else 0
                peak_c = max(all_c) * 100 if all_c else 0

                render_statistics(fire_t, smoke_t, avg_c, peak_c)
                st.markdown("<br>", unsafe_allow_html=True)
                render_timeline(st.session_state.demo_timeline)

        elif not st.session_state.demo_running and st.session_state.demo_timeline:
            # Demo stopped but we have history — show final stats
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"Demo completed. Processed through {st.session_state.demo_index} images.")

            fire_t = sum(
                1
                for e in st.session_state.demo_timeline
                if e["label"].lower() == "fire"
            )
            smoke_t = sum(
                1
                for e in st.session_state.demo_timeline
                if e["label"].lower() == "smoke"
            )
            all_c = [e["confidence"] for e in st.session_state.demo_timeline]
            avg_c = (sum(all_c) / len(all_c) * 100) if all_c else 0
            peak_c = max(all_c) * 100 if all_c else 0

            render_statistics(fire_t, smoke_t, avg_c, peak_c)
            st.markdown("<br>", unsafe_allow_html=True)
            render_timeline(st.session_state.demo_timeline)


# ═══════════════════════════════════════════
# AUTO-RERUN FOR DEMO MODE
# ═══════════════════════════════════════════

if st.session_state.get("demo_running", False) and mode == "🎬 Demo Mode":
    demo_files = get_image_files(DEMO_IMAGES_DIR)
    if demo_files:
        time.sleep(DEMO_INTERVAL_SECONDS)
        st.session_state.demo_index += 1
        st.rerun()
    else:
        st.session_state.demo_running = False


# ═══════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div class="footer-container">
    <div class="footer-title">🔥 Fire & Smoke Detection System</div>
    <div class="footer-subtitle">Built with YOLOv8n + Streamlit</div>
    <div class="footer-stats">
        <span class="footer-stat">Dataset: Smoke-Fire-Detection-YOLO</span>
        <span class="footer-stat">Model: YOLOv8n</span>
        <span class="footer-stat">mAP50: 0.776</span>
        <span class="footer-stat">Precision: 0.760</span>
        <span class="footer-stat">Recall: 0.718</span>
    </div>
    <div class="footer-author">
        Built by <strong>Shreya Singh</strong> ·
        <a href="#" target="_blank">GitHub Repository</a>
    </div>
</div>
""", unsafe_allow_html=True)