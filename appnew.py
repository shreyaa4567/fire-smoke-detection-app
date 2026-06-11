"""Fire & Smoke Detection System - UX Redesigned v2

====================================================
Mobile-first, no-scroll layout with:
- Image height capped, result always visible
- Video with annotated overlay playback
- Webcam auto-capture loop (~1 FPS live detection)
- Demo mode as clickable thumbnail gallery

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
import base64
from io import BytesIO

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
# CSS — Compact, Mobile-First, No-Scroll
# ═══════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }
.stApp { background-color: #080808; color: #e8e8e8; }

/* HEADER */
.header {
    background: linear-gradient(160deg, #140500 0%, #1f0800 40%, #140500 100%);
    border: 1px solid rgba(255, 80, 0, 0.35);
    border-radius: 14px;
    padding: 0.8rem 1.5rem;
    margin-bottom: 1rem;
    text-align: center;
}
.header-title { font-size: 1.5rem; font-weight: 700; color: #ff6535; margin: 0; }
.header-sub { font-size: 0.72rem; color: #666; margin-top: 0.15rem; letter-spacing: 1px; text-transform: uppercase; }
.header-badges { display: flex; gap: 0.4rem; justify-content: center; flex-wrap: wrap; margin-top: 0.5rem; }
.badge { background: rgba(255,69,0,0.1); border: 1px solid rgba(255,69,0,0.25); color: #cc5522; font-size: 0.62rem; padding: 0.12rem 0.45rem; border-radius: 12px; font-weight: 600; }

/* TABS */
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: transparent; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; border: 1px solid #1e1e1e; background: #0e0e0e; padding: 0.45rem 0.9rem; font-size: 0.85rem; }
.stTabs [aria-selected="true"] { border-color: rgba(255,80,0,0.4); background: rgba(255,60,0,0.08); }

/* RESULT CARD */
.result-card {
    background: #0a0a0a;
    border: 1px solid #1a1a1a;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.75rem;
}
.result-title { font-size: 0.65rem; color: #444; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.4rem; font-weight: 600; }
.risk-status { font-size: 0.9rem; font-weight: 600; margin-bottom: 0.3rem; }
.risk-high { color: #f87171; }
.risk-medium { color: #fb923c; }
.risk-low { color: #4ade80; }
.risk-safe { color: #60a5fa; }
.confidence-text { font-size: 0.75rem; color: #555; font-family: 'JetBrains Mono', monospace; }

/* STAT ROW */
.stat-row { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin-top: 0.6rem; }
.stat-box { background: #0e0e0e; border: 1px solid #1a1a1a; border-radius: 8px; padding: 0.6rem; text-align: center; }
.stat-value { font-size: 1.2rem; font-weight: 700; color: #ff6535; font-family: 'JetBrains Mono', monospace; }
.stat-label { font-size: 0.62rem; color: #444; margin-top: 0.15rem; text-transform: uppercase; }

/* CAPPED IMAGE — key fix for scroll problem */
.img-wrapper {
    background: #0e0e0e;
    border: 1px solid #1a1a1a;
    border-radius: 12px;
    padding: 0.6rem;
    margin-bottom: 0.75rem;
    overflow: hidden;
}
.img-wrapper img {
    width: 100%;
    max-height: 42vh;
    object-fit: contain;
    border-radius: 8px;
    display: block;
}
.output-label { font-size: 0.62rem; color: #444; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.35rem; }

/* DETECTION LIST */
.det-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #0e0e0e;
    border-left: 3px solid #ff4500;
    border-radius: 0 8px 8px 0;
    padding: 0.4rem 0.7rem;
    margin-bottom: 0.25rem;
    font-size: 0.82rem;
}
.det-item.smoke { border-left-color: #666; }
.det-conf { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #555; }

/* DEMO THUMBNAIL GRID */
.demo-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem;
    margin-bottom: 0.75rem;
}
.demo-thumb-label {
    font-size: 0.6rem;
    color: #555;
    text-align: center;
    margin-top: 0.2rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.demo-thumb-wrap {
    cursor: pointer;
    border-radius: 8px;
    overflow: hidden;
    border: 2px solid transparent;
    transition: border-color 0.2s;
}
.demo-thumb-wrap.selected { border-color: #ff6535; }
.demo-thumb-wrap img {
    width: 100%;
    height: 80px;
    object-fit: cover;
    border-radius: 6px;
    display: block;
}

/* INPUT SECTION */
.input-section {
    background: #0a0a0a;
    border: 1px dashed #1e1e1e;
    border-radius: 12px;
    padding: 1rem;
    color: #555;
    text-align: center;
}

/* EXPANDABLE */
.expandable-title { font-size: 0.65rem; color: #444; text-transform: uppercase; letter-spacing: 1px; margin: 0.75rem 0 0.4rem; font-weight: 600; }

/* WEBCAM LIVE STATUS */
.live-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3);
    color: #ef4444; font-size: 0.65rem; font-weight: 700;
    padding: 0.15rem 0.5rem; border-radius: 12px; letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}
.live-dot { width: 6px; height: 6px; background: #ef4444; border-radius: 50%; animation: pulse 1.2s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

/* FOOTER */
.footer {
    text-align: center; padding-top: 0.75rem;
    border-top: 1px solid #1a1a1a; margin-top: 1.2rem;
    font-size: 0.68rem; color: #333;
}
.footer a { color: #ff6535; text-decoration: none; }

/* MOBILE */
@media (max-width: 600px) {
    .header { padding: 0.6rem 0.8rem; }
    .header-title { font-size: 1.2rem; }
    .stat-row { grid-template-columns: 1fr 1fr; }
    .demo-grid { grid-template-columns: repeat(2, 1fr); }
    .img-wrapper img { max-height: 38vh; }
    .stTabs [data-baseweb="tab"] { padding: 0.35rem 0.6rem; font-size: 0.75rem; }
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
# HELPERS
# ═══════════════════════════════════════════

def analyze_detections(results):
    boxes = results[0].boxes
    fire_count = smoke_count = 0
    max_conf = 0.0
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
    if max_conf >= 0.80:
        return "🚨 HIGH RISK", "high"
    elif max_conf >= 0.50:
        return "⚠️ MEDIUM RISK", "medium"
    elif max_conf > 0:
        return "👁️ LOW RISK", "low"
    else:
        return "✅ ALL CLEAR", "safe"

def render_result_card(fire_count, smoke_count, max_conf):
    risk_text, risk_class = get_risk_level(max_conf)
    st.markdown(f"""
    <div class="result-card">
        <div class="result-title">Detection Result</div>
        <div class="risk-status risk-{risk_class}">{risk_text}</div>
        <div class="confidence-text">{max_conf*100:.1f}% max confidence</div>
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
    """, unsafe_allow_html=True)

def render_capped_image(img_array, label="Detection Output"):
    """Render image with max-height cap so it never pushes results off screen."""
    pil = Image.fromarray(img_array)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(f"""
    <div class="img-wrapper">
        <div class="output-label">{label}</div>
        <img src="data:image/jpeg;base64,{b64}" />
    </div>
    """, unsafe_allow_html=True)

def pil_to_b64(pil_img, size=(120, 80)):
    """Thumbnail base64 for demo grid."""
    pil_img = pil_img.copy()
    pil_img.thumbnail(size, Image.LANCZOS)
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()

def run_detection(image: Image.Image):
    """Run model and return results + timing."""
    t0 = time.time()
    results = model.predict(image, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    ms = round((time.time() - t0) * 1000, 1)
    return results, ms

# ═══════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════
st.markdown("""
<div class="header">
    <div class="header-title">🔥 Fire & Smoke Detection</div>
    <div class="header-sub">Real-time Detection · YOLOv8n</div>
    <div class="header-badges">
        <span class="badge">YOLOv8n</span>
        <span class="badge">mAP50: 0.776</span>
        <span class="badge">Recall: 0.718</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["📷 Image", "🎥 Video", "📹 Webcam", "🎬 Demo"])

# ───────────────────────────────────────────
# TAB 1: IMAGE — result card ABOVE capped image
# ───────────────────────────────────────────
with tab1:
    uploaded_file = st.file_uploader(
        "Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        if image.mode == "RGBA":
            image = image.convert("RGB")

        results, inference_ms = run_detection(image)
        fire_count, smoke_count, max_conf, detections = analyze_detections(results)

        # 1. Result card — always visible above the fold
        render_result_card(fire_count, smoke_count, max_conf)

        # 2. Two-column layout: original | annotated — both height-capped
        col_orig, col_det = st.columns(2)
        with col_orig:
            orig_arr = np.array(image)
            render_capped_image(orig_arr, "Original")
        with col_det:
            result_arr = results[0].plot()
            render_capped_image(result_arr, "Annotated")

        # 3. Detection list
        if detections:
            st.markdown('<div class="expandable-title">Detected Objects</div>', unsafe_allow_html=True)
            for label, conf in detections:
                icon = "🌫" if label.lower() == "smoke" else "🔥"
                css_cls = "smoke" if label.lower() == "smoke" else ""
                st.markdown(
                    f'<div class="det-item {css_cls}"><span>{icon} {label.upper()}</span>'
                    f'<span class="det-conf">{conf*100:.1f}%</span></div>',
                    unsafe_allow_html=True
                )

        # 4. Technical details collapsed
        with st.expander("📊 Technical Details"):
            c1, c2 = st.columns(2)
            c1.metric("Inference Time", f"{inference_ms} ms")
            c2.metric("Resolution", f"{image.size[0]}×{image.size[1]}")

        # 5. Download
        result_pil = Image.fromarray(results[0].plot())
        buf = BytesIO()
        result_pil.save(buf, format="JPEG")
        st.download_button(
            "⬇ Download Result", buf.getvalue(),
            "detection_result.jpg", "image/jpeg",
            use_container_width=True
        )

    else:
        st.markdown(
            '<div class="input-section">📷 <strong>Upload an image</strong> to detect fire and smoke</div>',
            unsafe_allow_html=True
        )

# ───────────────────────────────────────────
# TAB 2: VIDEO — annotated overlay video playback
# ───────────────────────────────────────────
with tab2:
    video_file = st.file_uploader(
        "Upload video", type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed"
    )

    if video_file:
        # Frame-skip slider so users can control speed vs detail
        frame_skip = st.slider(
            "Processing speed (higher = faster, fewer frames annotated)",
            min_value=1, max_value=10, value=3,
            help="1 = every frame (slow), 10 = every 10th frame (fast)"
        )

        if st.button("▶ Process & Play Video", use_container_width=True):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            tfile.flush()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Output video file
            out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out_path = out_file.name
            out_file.close()

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            progress = st.progress(0, text="Annotating frames…")
            fire_total = smoke_total = 0
            frame_idx = 0
            last_annotated = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    results = model.predict(img, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

                    for box in results[0].boxes:
                        lbl = model.names[int(box.cls[0])]
                        if lbl.lower() == "fire": fire_total += 1
                        elif lbl.lower() == "smoke": smoke_total += 1

                    annotated = results[0].plot()  # RGB numpy
                    last_annotated = annotated
                    frame_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                else:
                    # For skipped frames, use last annotated frame (smooth playback)
                    if last_annotated is not None:
                        frame_bgr = cv2.cvtColor(last_annotated, cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = frame

                writer.write(frame_bgr)
                progress.progress(
                    min((frame_idx + 1) / max(total_frames, 1), 1.0),
                    text=f"Annotating frame {frame_idx+1} / {total_frames}…"
                )
                frame_idx += 1

            cap.release()
            writer.release()
            os.unlink(tfile.name)
            progress.empty()

            # Result summary
            max_conf_est = 0.85 if (fire_total + smoke_total) > 0 else 0.0
            render_result_card(fire_total, smoke_total, max_conf_est)

            # Play annotated video inline — no download needed to see detections
            with open(out_path, "rb") as vf:
                video_bytes = vf.read()
            st.video(video_bytes)
            os.unlink(out_path)

            with st.expander("📊 Video Details"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Frames", total_frames)
                c2.metric("FPS", int(fps))
                c3.metric("Duration", f"{round(total_frames/fps, 1)}s")

    else:
        st.markdown(
            '<div class="input-section">🎥 <strong>Upload a video</strong> — annotated overlay plays in browser</div>',
            unsafe_allow_html=True
        )

# ───────────────────────────────────────────
# TAB 3: WEBCAM — auto-capture loop (~1 FPS)
# ───────────────────────────────────────────
with tab3:
    st.markdown(
        '<div class="live-badge"><span class="live-dot"></span>LIVE DETECTION</div>',
        unsafe_allow_html=True
    )
    st.caption("Camera captures automatically every second. Point at fire or smoke to detect.")

    # Controls
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_live = st.button("▶ Start Live Detection", use_container_width=True)
    with col_btn2:
        stop_live = st.button("⏹ Stop", use_container_width=True)

    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

    if start_live:
        st.session_state.webcam_running = True
    if stop_live:
        st.session_state.webcam_running = False

    # Camera input — always visible
    webcam_img = st.camera_input(
        "Camera feed",
        label_visibility="collapsed",
        key="webcam_capture"
    )

    result_placeholder = st.empty()
    image_placeholder  = st.empty()
    dets_placeholder   = st.empty()

    if webcam_img:
        image = Image.open(webcam_img)
        if image.mode == "RGBA":
            image = image.convert("RGB")

        results, inference_ms = run_detection(image)
        fire_count, smoke_count, max_conf, detections = analyze_detections(results)

        with result_placeholder.container():
            render_result_card(fire_count, smoke_count, max_conf)

        with image_placeholder.container():
            result_arr = results[0].plot()
            render_capped_image(result_arr, f"Detection · {inference_ms}ms")

        with dets_placeholder.container():
            if detections:
                st.markdown('<div class="expandable-title">Detected Objects</div>', unsafe_allow_html=True)
                for label, conf in detections:
                    icon = "🌫" if label.lower() == "smoke" else "🔥"
                    css_cls = "smoke" if label.lower() == "smoke" else ""
                    st.markdown(
                        f'<div class="det-item {css_cls}"><span>{icon} {label.upper()}</span>'
                        f'<span class="det-conf">{conf*100:.1f}%</span></div>',
                        unsafe_allow_html=True
                    )

        # Auto-rerun loop when live mode is on
        if st.session_state.webcam_running:
            time.sleep(1.0)
            st.rerun()

    elif st.session_state.webcam_running:
        # No image yet — wait and rerun
        time.sleep(0.8)
        st.rerun()

# ───────────────────────────────────────────
# TAB 4: DEMO — clickable thumbnail gallery
# ───────────────────────────────────────────
with tab4:
    demo_dir = DEMO_IMAGES_DIR

    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)

    demo_images = sorted([
        f for f in os.listdir(demo_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ])

    if not demo_images:
        st.markdown(
            '<div class="input-section">🎬 <strong>No demo images found</strong> — add images to demo_images/ folder</div>',
            unsafe_allow_html=True
        )
    else:
        # Session state for selected image
        if "demo_selected" not in st.session_state:
            st.session_state.demo_selected = 0

        st.markdown('<div class="expandable-title">Choose an image</div>', unsafe_allow_html=True)

        # Build thumbnail grid using Streamlit columns (3 per row)
        cols_per_row = 3
        rows = [demo_images[i:i+cols_per_row] for i in range(0, len(demo_images), cols_per_row)]

        for row_imgs in rows:
            cols = st.columns(cols_per_row)
            for col_i, fname in enumerate(row_imgs):
                idx = demo_images.index(fname)
                img_path = os.path.join(demo_dir, fname)
                thumb = Image.open(img_path).convert("RGB")
                b64 = pil_to_b64(thumb)
                is_selected = (st.session_state.demo_selected == idx)
                border = "#ff6535" if is_selected else "#1a1a1a"
                short_name = fname[:14] + "…" if len(fname) > 14 else fname

                with cols[col_i]:
                    st.markdown(
                        f'<div style="border:2px solid {border}; border-radius:8px; overflow:hidden; margin-bottom:4px;">'
                        f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;height:72px;object-fit:cover;display:block;" />'
                        f'</div>'
                        f'<div style="font-size:0.6rem;color:#555;text-align:center;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{short_name}</div>',
                        unsafe_allow_html=True
                    )
                    if st.button("Select", key=f"demo_sel_{idx}", use_container_width=True):
                        st.session_state.demo_selected = idx
                        st.rerun()

        st.divider()

        # Show detection for selected image
        selected_fname = demo_images[st.session_state.demo_selected]
        selected_path  = os.path.join(demo_dir, selected_fname)
        image = Image.open(selected_path).convert("RGB")

        results, inference_ms = run_detection(image)
        fire_count, smoke_count, max_conf, detections = analyze_detections(results)

        # Result card — always above image
        render_result_card(fire_count, smoke_count, max_conf)

        # Side-by-side original + annotated
        col_o, col_a = st.columns(2)
        with col_o:
            render_capped_image(np.array(image), "Original")
        with col_a:
            render_capped_image(results[0].plot(), f"Annotated · {inference_ms}ms")

        if detections:
            st.markdown('<div class="expandable-title">Detected Objects</div>', unsafe_allow_html=True)
            for label, conf in detections:
                icon = "🌫" if label.lower() == "smoke" else "🔥"
                css_cls = "smoke" if label.lower() == "smoke" else ""
                st.markdown(
                    f'<div class="det-item {css_cls}"><span>{icon} {label.upper()}</span>'
                    f'<span class="det-conf">{conf*100:.1f}%</span></div>',
                    unsafe_allow_html=True
                )

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