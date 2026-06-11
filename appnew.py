"""Fire & Smoke Detection System - UX Redesigned
================================================
Tabs-based interface with result-first layout, real-time webcam, 
and mobile-optimized design.

Author: Shreya Singh
License: MIT
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
import os
import cv2
import tempfile

# ═══════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════

MODEL_PATH = "best_shreya_v2.pt"
CONFIDENCE_THRESHOLD = 0.15
IOU_THRESHOLD = 0.45
DEMO_IMAGES_DIR = "demo_images"

# ═══════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════

st.set_page_config(
    page_title="Fire & Smoke Detection",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════
# CSS — Compact, Mobile-First
# ═══════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    * { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #080808; color: #e8e8e8; }

    /* HEADER */
    .header {
        background: linear-gradient(160deg, #140500 0%, #1f0800 40%, #140500 100%);
        border: 1px solid rgba(255, 80, 0, 0.35);
        border-radius: 14px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .header-title { font-size: 1.6rem; font-weight: 700; color: #ff6535; margin: 0; }
    .header-sub { font-size: 0.75rem; color: #666; margin-top: 0.2rem; letter-spacing: 1px; text-transform: uppercase; }
    .header-badges { display: flex; gap: 0.4rem; justify-content: center; flex-wrap: wrap; margin-top: 0.6rem; }
    .badge { background: rgba(255,69,0,0.1); border: 1px solid rgba(255,69,0,0.25); color: #cc5522; font-size: 0.65rem; padding: 0.15rem 0.5rem; border-radius: 12px; font-weight: 600; }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: transparent; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; border: 1px solid #1e1e1e; background: #0e0e0e; padding: 0.5rem 1rem; }
    .stTabs [aria-selected="true"] { border-color: rgba(255,80,0,0.4); background: rgba(255,60,0,0.08); }

    /* RESULT CARD (TOP) */
    .result-card {
        background: #0a0a0a;
        border: 1px solid #1a1a1a;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .result-title { font-size: 0.7rem; color: #444; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem; font-weight: 600; }
    .risk-status { font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem; }
    .risk-high { color: #f87171; }
    .risk-medium { color: #fb923c; }
    .risk-low { color: #4ade80; }
    .risk-safe { color: #60a5fa; }
    .confidence-text { font-size: 0.8rem; color: #555; font-family: 'JetBrains Mono', monospace; }

    /* STAT ROW */
    .stat-row { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 0.5rem; }
    .stat-box {
        background: #0e0e0e;
        border: 1px solid #1a1a1a;
        border-radius: 8px;
        padding: 0.7rem;
        text-align: center;
    }
    .stat-value { font-size: 1.3rem; font-weight: 700; color: #ff6535; font-family: 'JetBrains Mono', monospace; }
    .stat-label { font-size: 0.65rem; color: #444; margin-top: 0.2rem; text-transform: uppercase; }

    /* IMAGE/VIDEO OUTPUT */
    .output-container {
        background: #0e0e0e;
        border: 1px solid #1a1a1a;
        border-radius: 12px;
        padding: 0.8rem;
        margin-bottom: 1rem;
    }
    .output-label { font-size: 0.65rem; color: #444; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.4rem; }

    /* INPUT SECTION */
    .input-section {
        background: #0a0a0a;
        border: 1px dashed #1e1e1e;
        border-radius: 12px;
        padding: 1rem;
    }

    /* DETECTION LIST */
    .det-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #0e0e0e;
        border-left: 3px solid #ff4500;
        border-radius: 0 8px 8px 0;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.3rem;
        font-size: 0.85rem;
    }
    .det-item.smoke { border-left-color: #666; }
    .det-conf { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #555; }

    /* EXPANDABLE SECTION */
    .expandable-title {
        font-size: 0.7rem;
        color: #444;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }

    /* FOOTER */
    .footer {
        text-align: center;
        padding-top: 1rem;
        border-top: 1px solid #1a1a1a;
        margin-top: 1.5rem;
        font-size: 0.7rem;
        color: #333;
    }
    .footer a { color: #ff6535; text-decoration: none; }

    /* MOBILE RESPONSIVE */
    @media (max-width: 600px) {
        .header { padding: 0.8rem 1rem; }
        .header-title { font-size: 1.3rem; }
        .stat-row { grid-template-columns: 1fr; }
        .stTabs [data-baseweb="tab"] { padding: 0.4rem 0.8rem; font-size: 0.8rem; }
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ═══════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════

st.markdown("""
<div class="header">
    <div class="header-title">🔥 Fire & Smoke Detection</div>
    <div class="header-sub">Real-time Detection Powered by YOLOv8</div>
    <div class="header-badges">
        <span class="badge">YOLOv8n</span>
        <span class="badge">mAP50: 0.776</span>
        <span class="badge">Recall: 0.718</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════

def analyze_detections(results):
    """Extract detection info from YOLO results."""
    boxes = results[0].boxes
    fire_count = smoke_count = 0
    max_conf = 0
    detections = []
    
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        max_conf = max(max_conf, conf)
        
        if label.lower() == "fire":
            fire_count += 1
        elif label.lower() == "smoke":
            smoke_count += 1
        
        detections.append((label, conf))
    
    return fire_count, smoke_count, max_conf, detections

def get_risk_level(max_conf):
    """Determine risk level from confidence."""
    if max_conf >= 0.80:
        return "🚨 HIGH RISK", "high"
    elif max_conf >= 0.50:
        return "⚠️ MEDIUM RISK", "medium"
    elif max_conf > 0:
        return "👁️ LOW RISK", "low"
    else:
        return "✅ ALL CLEAR", "safe"

def render_result_card(fire_count, smoke_count, max_conf):
    """Render top result card."""
    risk_text, risk_class = get_risk_level(max_conf)
    
    html = f"""
    <div class="result-card">
        <div class="result-title">Detection Result</div>
        <div class="risk-status risk-{risk_class}">{risk_text}</div>
        <div class="confidence-text">{max_conf*100:.1f}% confidence</div>
        <div class="stat-row">
            <div class="stat-box">
                <div class="stat-value">{fire_count}</div>
                <div class="stat-label">🔥 Fire</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{smoke_count}</div>
                <div class="stat-label">🌫 Smoke</div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs(["📷 Image", "🎥 Video", "📹 Webcam", "🎬 Demo"])

# ═══════════════════════════════════════════
# TAB 1: IMAGE UPLOAD
# ═══════════════════════════════════════════

with tab1:
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"], label_visibility="collapsed")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        
        # Run detection
        t0 = time.time()
        results = model.predict(image, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        inference_ms = round((time.time() - t0) * 1000, 1)
        
        # Parse results
        fire_count, smoke_count, max_conf, detections = analyze_detections(results)
        
        # RESULT CARD (TOP)
        render_result_card(fire_count, smoke_count, max_conf)
        
        # OUTPUT IMAGE
        result_image = results[0].plot()
        st.markdown('<div class="output-label">Annotated Detection</div>', unsafe_allow_html=True)
        st.image(result_image, use_container_width=True)
        
        # DETECTIONS LIST
        if detections:
            st.markdown('<div class="expandable-title">Detected Objects</div>', unsafe_allow_html=True)
            for label, conf in detections:
                icon = "🌫" if label.lower() == "smoke" else "🔥"
                st.markdown(f'<div class="det-item {"smoke" if label.lower() == "smoke" else ""}"><span>{icon} {label.upper()}</span><span class="det-conf">{conf*100:.1f}%</span></div>', unsafe_allow_html=True)
        
        # TECHNICAL METRICS (COLLAPSED)
        with st.expander("📊 Technical Details"):
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Inference Time", f"{inference_ms} ms")
            with c2:
                st.metric("Resolution", f"{image.size[0]}×{image.size[1]}")
        
        # DOWNLOAD
        result_pil = Image.fromarray(result_image)
        result_pil.save("result.jpg")
        with open("result.jpg", "rb") as f:
            st.download_button("⬇ Download Result", f, "detection_result.jpg", use_container_width=True)
    else:
        st.markdown('<div class="input-section">📷 <strong>Upload an image</strong> to detect fire and smoke</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TAB 2: VIDEO UPLOAD
# ═══════════════════════════════════════════

with tab2:
    video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"], label_visibility="collapsed")
    
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.flush()
        
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        
        # Process every Nth frame for speed
        frame_skip = max(1, int(fps // 5))
        
        st.markdown('<div class="expandable-title">Processing Video...</div>', unsafe_allow_html=True)
        progress = st.progress(0)
        
        frame_idx = 0
        fire_total = smoke_total = 0
        result_frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_skip == 0:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                results = model.predict(img, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
                
                for box in results[0].boxes:
                    label = model.names[int(box.cls[0])]
                    if label.lower() == "fire": fire_total += 1
                    elif label.lower() == "smoke": smoke_total += 1
                
                result_frames.append(results[0].plot())
            
            progress.progress(min(frame_idx / total_frames, 1.0))
            frame_idx += 1
        
        cap.release()
        os.unlink(tfile.name)
        
        # RESULT CARD
        max_conf = 1.0 if (fire_total + smoke_total) > 0 else 0
        render_result_card(fire_total, smoke_total, max_conf)
        
        # SHOW FIRST FRAME WITH DETECTION
        if result_frames:
            st.markdown('<div class="output-label">Sample Detection Frame</div>', unsafe_allow_html=True)
            st.image(result_frames[0], use_container_width=True)
        
        # SUMMARY
        st.markdown(f"""
        <div class="result-card">
            <div class="stat-row">
                <div class="stat-box"><div class="stat-value">{fire_total}</div><div class="stat-label">🔥 Fire</div></div>
                <div class="stat-box"><div class="stat-value">{smoke_total}</div><div class="stat-label">🌫 Smoke</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📊 Video Details"):
            st.write(f"**Total Frames:** {total_frames}")
            st.write(f"**FPS:** {int(fps)}")
            st.write(f"**Duration:** {round(total_frames/fps, 1)}s")
    else:
        st.markdown('<div class="input-section">🎥 <strong>Upload a video</strong> for frame-by-frame detection</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TAB 3: WEBCAM (REAL-TIME)
# ═══════════════════════════════════════════

with tab3:
    st.write("📹 **Live Webcam Detection**")
    
    # Two columns: webcam on left, controls on right
    col1, col2 = st.columns([2, 1])
    
    with col1:
        webcam_img = st.camera_input("Capture photo", label_visibility="collapsed")
    
    with col2:
        st.write("**Take a photo** for instant detection")
    
    if webcam_img:
        image = Image.open(webcam_img)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        
        # Run detection
        t0 = time.time()
        results = model.predict(image, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        inference_ms = round((time.time() - t0) * 1000, 1)
        
        # Parse results
        fire_count, smoke_count, max_conf, detections = analyze_detections(results)
        
        # RESULT CARD
        render_result_card(fire_count, smoke_count, max_conf)
        
        # OUTPUT
        result_image = results[0].plot()
        st.markdown('<div class="output-label">Detection Overlay</div>', unsafe_allow_html=True)
        st.image(result_image, use_container_width=True)
        
        # DETECTIONS
        if detections:
            st.markdown('<div class="expandable-title">Detected Objects</div>', unsafe_allow_html=True)
            for label, conf in detections:
                icon = "🌫" if label.lower() == "smoke" else "🔥"
                st.markdown(f'<div class="det-item {"smoke" if label.lower() == "smoke" else ""}"><span>{icon} {label.upper()}</span><span class="det-conf">{conf*100:.1f}%</span></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TAB 4: DEMO MODE
# ═══════════════════════════════════════════

with tab4:
    demo_dir = DEMO_IMAGES_DIR
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)
    
    demo_images = sorted([f for f in os.listdir(demo_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))])
    
    if not demo_images:
        st.markdown('<div class="input-section">🎬 <strong>No demo images found</strong> — add images to demo_images/ folder</div>', unsafe_allow_html=True)
    else:
        total = len(demo_images)
        
        # Session state for demo
        if "demo_index" not in st.session_state:
            st.session_state.demo_index = 0
        if "demo_running" not in st.session_state:
            st.session_state.demo_running = False
        
        idx = st.session_state.demo_index % total
        
        # Controls
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("▶ Start", use_container_width=True, disabled=st.session_state.demo_running):
                st.session_state.demo_running = True
                st.rerun()
        with c2:
            if st.button("⏹ Stop", use_container_width=True, disabled=not st.session_state.demo_running):
                st.session_state.demo_running = False
                st.rerun()
        with c3:
            st.write(f"**{idx+1} / {total}**")
        
        # Load image
        img_path = os.path.join(demo_dir, demo_images[idx])
        image = Image.open(img_path).convert("RGB")
        
        # Run detection
        t0 = time.time()
        results = model.predict(image, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        inference_ms = round((time.time() - t0) * 1000, 1)
        
        fire_count, smoke_count, max_conf, detections = analyze_detections(results)
        
        # RESULT
        render_result_card(fire_count, smoke_count, max_conf)
        
        # IMAGE
        result_image = results[0].plot()
        st.image(result_image, use_container_width=True)
        
        # Auto-advance
        if st.session_state.demo_running:
            time.sleep(2)
            st.session_state.demo_index = (idx + 1) % total
            st.rerun()

# ═══════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════

st.markdown("""
<div class="footer">
    Built by <strong>Shreya Singh</strong> · 
    <a href="https://github.com/shreyaa4567/fire-smoke-detection-app" target="_blank">GitHub</a> · 
    <a href="https://www.linkedin.com/in/shreya-singh-35bab7337/" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)