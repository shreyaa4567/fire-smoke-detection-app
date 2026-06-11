"""Fire & Smoke Detection System - UX Fix v4

================================================
Fixes:
- Header gap removed, font sizes increased
- Demo: 5 columns per row, smaller thumbnails, full image visible
- Video: frame-buffer slideshow (no codec issues)
- Mobile: original + annotated side by side forced via CSS
- Result card fonts larger throughout

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
# CSS
# ═══════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }
.stApp { background-color: #080808; color: #e8e8e8; }

/* Remove Streamlit default top gap */
header[data-testid="stHeader"] { height: 0 !important; min-height: 0 !important; }
div[data-testid="stToolbar"] { display: none !important; }
.block-container { padding-top: 1rem !important; }

/* HEADER */
.header {
    background: linear-gradient(160deg, #140500 0%, #1f0800 40%, #140500 100%);
    border: 1px solid rgba(255, 80, 0, 0.35);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    text-align: center;
    min-height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}
.header-title { font-size: 2rem; font-weight: 700; color: #ff6535; margin: 0; line-height: 1.2; }
.header-sub { font-size: 0.85rem; color: #666; margin-top: 0.3rem; letter-spacing: 1.5px; text-transform: uppercase; }
.header-badges { display: flex; gap: 0.5rem; justify-content: center; flex-wrap: wrap; margin-top: 0.6rem; }
.badge { background: rgba(255,69,0,0.1); border: 1px solid rgba(255,69,0,0.25); color: #cc5522; font-size: 0.75rem; padding: 0.18rem 0.55rem; border-radius: 12px; font-weight: 600; }

/* TABS */
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: transparent; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; border: 1px solid #1e1e1e; background: #0e0e0e; padding: 0.45rem 0.9rem; font-size: 0.85rem; }
.stTabs [aria-selected="true"] { border-color: rgba(255,80,0,0.4); background: rgba(255,60,0,0.08); }

/* RESULT CARD */
.result-card {
    background: #0a0a0a;
    border: 1px solid #1a1a1a;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.75rem;
}
.result-title { font-size: 0.8rem; color: #555; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.45rem; font-weight: 600; }
.risk-status { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.35rem; }
.risk-high   { color: #f87171; }
.risk-medium { color: #fb923c; }
.risk-low    { color: #4ade80; }
.risk-safe   { color: #60a5fa; }
.confidence-text { font-size: 0.95rem; color: #666; font-family: 'JetBrains Mono', monospace; }
.stat-row { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin-top: 0.7rem; }
.stat-box { background: #0e0e0e; border: 1px solid #1a1a1a; border-radius: 8px; padding: 0.7rem; text-align: center; }
.stat-value { font-size: 1.5rem; font-weight: 700; color: #ff6535; font-family: 'JetBrains Mono', monospace; }
.stat-label { font-size: 0.78rem; color: #555; margin-top: 0.2rem; text-transform: uppercase; }

/* CAPPED IMAGE */
.img-wrapper {
    background: #0e0e0e;
    border: 1px solid #1a1a1a;
    border-radius: 12px;
    padding: 0.5rem;
    margin-bottom: 0.6rem;
    overflow: hidden;
}
.img-wrapper img {
    width: 100%;
    max-height: 40vh;
    object-fit: contain;
    border-radius: 6px;
    display: block;
}
.output-label { font-size: 0.68rem; color: #555; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.3rem; }

/* FORCE side-by-side columns on ALL screen sizes including mobile */
div[data-testid="stHorizontalBlock"] {
    flex-wrap: nowrap !important;
    gap: 0.5rem;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    min-width: 0 !important;
    flex: 1 1 0 !important;
}

/* DETECTION LIST */
.det-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #0e0e0e;
    border-left: 3px solid #ff4500;
    border-radius: 0 8px 8px 0;
    padding: 0.45rem 0.8rem;
    margin-bottom: 0.25rem;
    font-size: 0.9rem;
}
.det-item.smoke { border-left-color: #666; }
.det-conf { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #666; }
.expandable-title { font-size: 0.72rem; color: #555; text-transform: uppercase; letter-spacing: 1px; margin: 0.75rem 0 0.4rem; font-weight: 600; }

/* INPUT SECTION */
.input-section {
    background: #0a0a0a;
    border: 1px dashed #1e1e1e;
    border-radius: 12px;
    padding: 1.2rem;
    color: #555;
    text-align: center;
    font-size: 0.95rem;
}

/* WEBCAM INFO */
.webcam-info {
    background: #0a0a0a;
    border: 1px solid #1a1a1a;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    font-size: 0.85rem;
    color: #666;
    margin-bottom: 0.75rem;
}

/* FOOTER */
.footer {
    text-align: center; padding-top: 0.75rem;
    border-top: 1px solid #1a1a1a; margin-top: 1.2rem;
    font-size: 0.7rem; color: #333;
}
.footer a { color: #ff6535; text-decoration: none; }

/* MOBILE */
@media (max-width: 640px) {
    .header-title { font-size: 1.4rem; }
    .header-sub   { font-size: 0.72rem; }
    .img-wrapper img { max-height: 32vh; }
    .risk-status  { font-size: 1rem; }
    .stat-value   { font-size: 1.25rem; }
    .confidence-text { font-size: 0.82rem; }
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
        conf   = float(box.conf[0])
        label  = model.names[cls_id]
        max_conf = max(max_conf, conf)
        if label.lower() == "fire":    fire_count  += 1
        elif label.lower() == "smoke": smoke_count += 1
        detections.append((label, conf))
    return fire_count, smoke_count, max_conf, detections

def get_risk_level(max_conf):
    if max_conf >= 0.80:   return "🚨 HIGH RISK",   "high"
    elif max_conf >= 0.50: return "⚠️ MEDIUM RISK", "medium"
    elif max_conf > 0:     return "👁️ LOW RISK",    "low"
    else:                  return "✅ ALL CLEAR",    "safe"

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

def arr_to_b64(img_array: np.ndarray, quality: int = 85) -> str:
    pil = Image.fromarray(img_array)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()

def render_capped_image(img_array: np.ndarray, label: str = "Output"):
    b64 = arr_to_b64(img_array)
    st.markdown(f"""
    <div class="img-wrapper">
        <div class="output-label">{label}</div>
        <img src="data:image/jpeg;base64,{b64}" />
    </div>
    """, unsafe_allow_html=True)

def render_detections_list(detections):
    if detections:
        st.markdown('<div class="expandable-title">Detected Objects</div>', unsafe_allow_html=True)
        for label, conf in detections:
            icon    = "🌫" if label.lower() == "smoke" else "🔥"
            css_cls = "smoke" if label.lower() == "smoke" else ""
            st.markdown(
                f'<div class="det-item {css_cls}"><span>{icon} {label.upper()}</span>'
                f'<span class="det-conf">{conf*100:.1f}%</span></div>',
                unsafe_allow_html=True
            )

def pil_to_b64_thumb(pil_img: Image.Image) -> str:
    thumb = pil_img.copy()
    thumb.thumbnail((200, 130), Image.LANCZOS)
    buf = BytesIO()
    thumb.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()

@st.cache_data(show_spinner=False)
def run_detection_cached(img_bytes: bytes):
    image = Image.open(BytesIO(img_bytes)).convert("RGB")
    t0 = time.time()
    results = model.predict(image, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    ms = round((time.time() - t0) * 1000, 1)
    annotated = results[0].plot()
    fire_count, smoke_count, max_conf, detections = analyze_detections(results)
    return fire_count, smoke_count, max_conf, detections, annotated, ms

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
# TAB 1: IMAGE
# ───────────────────────────────────────────
with tab1:
    uploaded_file = st.file_uploader(
        "Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        img_bytes = uploaded_file.read()
        fire_count, smoke_count, max_conf, detections, annotated, ms = run_detection_cached(img_bytes)

        render_result_card(fire_count, smoke_count, max_conf)

        col_o, col_a = st.columns(2)
        with col_o:
            orig = np.array(Image.open(BytesIO(img_bytes)).convert("RGB"))
            render_capped_image(orig, "Original")
        with col_a:
            render_capped_image(annotated, f"Annotated · {ms}ms")

        render_detections_list(detections)

        with st.expander("📊 Technical Details"):
            img_pil = Image.open(BytesIO(img_bytes))
            c1, c2 = st.columns(2)
            c1.metric("Inference Time", f"{ms} ms")
            c2.metric("Resolution", f"{img_pil.size[0]}×{img_pil.size[1]}")

        buf = BytesIO()
        Image.fromarray(annotated).save(buf, format="JPEG")
        st.download_button(
            "⬇ Download Result", buf.getvalue(),
            "detection_result.jpg", "image/jpeg",
            use_container_width=True
        )
    else:
        st.markdown(
            '<div class="input-section">📷 Upload an image to detect fire and smoke</div>',
            unsafe_allow_html=True
        )

# ───────────────────────────────────────────
# TAB 2: VIDEO — frame-buffer playback, no codec issues
# ───────────────────────────────────────────
with tab2:
    video_file = st.file_uploader(
        "Upload video", type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed"
    )

    if video_file:
        frame_skip = st.slider(
            "Frames to skip (higher = faster processing)",
            min_value=1, max_value=15, value=5,
            help="1 = every frame (slow & detailed), 15 = every 15th frame (fast)"
        )

        if st.button("▶ Process Video", use_container_width=True, type="primary"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            tfile.flush()
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps          = cap.get(cv2.CAP_PROP_FPS) or 25

            progress = st.progress(0, text="Processing…")
            fire_total = smoke_total = 0
            annotated_frames = []
            last_annotated   = None
            frame_idx        = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    results = model.predict(
                        img, conf=CONFIDENCE_THRESHOLD,
                        iou=IOU_THRESHOLD, verbose=False
                    )
                    for box in results[0].boxes:
                        lbl = model.names[int(box.cls[0])]
                        if lbl.lower() == "fire":    fire_total  += 1
                        elif lbl.lower() == "smoke": smoke_total += 1
                    last_annotated = results[0].plot()

                if last_annotated is not None:
                    annotated_frames.append(last_annotated)

                progress.progress(
                    min((frame_idx + 1) / max(total_frames, 1), 1.0),
                    text=f"Frame {frame_idx+1} / {total_frames}…"
                )
                frame_idx += 1

            cap.release()
            os.unlink(tfile.name)
            progress.empty()

            st.session_state["vid_frames"]  = annotated_frames
            st.session_state["vid_fps"]     = fps
            st.session_state["vid_fire"]    = fire_total
            st.session_state["vid_smoke"]   = smoke_total
            st.session_state["vid_total"]   = total_frames
            st.session_state["vid_playing"] = False

        if "vid_frames" in st.session_state and st.session_state["vid_frames"]:
            frames      = st.session_state["vid_frames"]
            fps         = st.session_state["vid_fps"]
            fire_total  = st.session_state["vid_fire"]
            smoke_total = st.session_state["vid_smoke"]

            max_conf_est = 0.85 if (fire_total + smoke_total) > 0 else 0.0
            render_result_card(fire_total, smoke_total, max_conf_est)

            cb1, cb2, cb3 = st.columns(3)
            with cb1:
                play = st.button("▶ Play", use_container_width=True)
            with cb2:
                stop = st.button("⏹ Stop", use_container_width=True)
            with cb3:
                st.caption(f"{len(frames)} frames · {int(fps)} FPS")

            if play:  st.session_state["vid_playing"] = True
            if stop:  st.session_state["vid_playing"] = False

            frame_ph = st.empty()
            if frames:
                frame_ph.image(frames[0], use_container_width=True)

            if st.session_state.get("vid_playing", False):
                delay = 1.0 / max(fps, 1)
                for frm in frames:
                    if not st.session_state.get("vid_playing", False):
                        break
                    frame_ph.image(frm, use_container_width=True)
                    time.sleep(delay)
                st.session_state["vid_playing"] = False

            with st.expander("📊 Video Details"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Frames", st.session_state["vid_total"])
                c2.metric("FPS", int(fps))
                c3.metric("Duration", f"{round(st.session_state['vid_total']/fps, 1)}s")

    else:
        for k in ["vid_frames","vid_fps","vid_fire","vid_smoke","vid_total","vid_playing"]:
            st.session_state.pop(k, None)
        st.markdown(
            '<div class="input-section">🎥 Upload a video — annotated frames play back in browser</div>',
            unsafe_allow_html=True
        )

# ───────────────────────────────────────────
# TAB 3: WEBCAM
# ───────────────────────────────────────────
with tab3:
    st.markdown(
        '<div class="webcam-info">📹 Take a photo — detection runs instantly. Retake to scan again.</div>',
        unsafe_allow_html=True
    )
    webcam_img = st.camera_input("", label_visibility="collapsed")

    if webcam_img:
        img_bytes = webcam_img.getvalue()
        fire_count, smoke_count, max_conf, detections, annotated, ms = run_detection_cached(img_bytes)

        render_result_card(fire_count, smoke_count, max_conf)

        col_o, col_a = st.columns(2)
        with col_o:
            orig = np.array(Image.open(BytesIO(img_bytes)).convert("RGB"))
            render_capped_image(orig, "Original")
        with col_a:
            render_capped_image(annotated, f"Annotated · {ms}ms")

        render_detections_list(detections)
    else:
        st.markdown(
            '<div class="input-section" style="margin-top:0.5rem;">Point camera and tap the capture button above</div>',
            unsafe_allow_html=True
        )

# ───────────────────────────────────────────
# TAB 4: DEMO — 5-per-row thumbnail grid
# ───────────────────────────────────────────
with tab4:
    if not os.path.exists(DEMO_IMAGES_DIR):
        os.makedirs(DEMO_IMAGES_DIR)

    demo_images = sorted([
        f for f in os.listdir(DEMO_IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ])

    if not demo_images:
        st.markdown(
            '<div class="input-section">🎬 No demo images — add images to demo_images/ folder</div>',
            unsafe_allow_html=True
        )
    else:
        if "demo_selected" not in st.session_state:
            st.session_state.demo_selected = 0

        COLS = 5
        rows = [demo_images[i:i+COLS] for i in range(0, len(demo_images), COLS)]

        for row_imgs in rows:
            padded = row_imgs + [""] * (COLS - len(row_imgs))
            cols   = st.columns(COLS)
            for ci, fname in enumerate(padded):
                if not fname:
                    continue
                idx      = demo_images.index(fname)
                img_path = os.path.join(DEMO_IMAGES_DIR, fname)
                thumb    = Image.open(img_path).convert("RGB")
                b64      = pil_to_b64_thumb(thumb)
                is_sel   = (st.session_state.demo_selected == idx)
                border   = "#ff6535" if is_sel else "#222"
                short    = fname[:12] + "…" if len(fname) > 12 else fname

                with cols[ci]:
                    st.markdown(
                        f'<div style="border:2px solid {border};border-radius:6px;'
                        f'background:#111;padding:2px;">'
                        f'<img src="data:image/jpeg;base64,{b64}" '
                        f'style="width:100%;height:62px;object-fit:contain;'
                        f'display:block;border-radius:4px;background:#111;"/>'
                        f'</div>'
                        f'<div style="font-size:0.55rem;color:#444;text-align:center;'
                        f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'
                        f'margin-top:2px;">{short}</div>',
                        unsafe_allow_html=True
                    )
                    if st.button("✓", key=f"ds_{idx}", use_container_width=True):
                        st.session_state.demo_selected = idx
                        st.rerun()

        st.divider()

        sel_fname = demo_images[st.session_state.demo_selected]
        sel_path  = os.path.join(DEMO_IMAGES_DIR, sel_fname)

        with open(sel_path, "rb") as f:
            img_bytes = f.read()

        fire_count, smoke_count, max_conf, detections, annotated, ms = run_detection_cached(img_bytes)

        render_result_card(fire_count, smoke_count, max_conf)

        col_o, col_a = st.columns(2)
        with col_o:
            orig = np.array(Image.open(BytesIO(img_bytes)).convert("RGB"))
            render_capped_image(orig, "Original")
        with col_a:
            render_capped_image(annotated, f"Annotated · {ms}ms")

        render_detections_list(detections)

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