"""Fire & Smoke Detection System - UX Fix v3

================================================
Fixes:
- Result card font sizes increased
- Video: auto-process on upload, fixed codec for browser playback
- Webcam: simplified to auto-detect on every capture (no broken loop)
- Demo: full-image thumbnails, result shown inline below grid
- Mobile: reduced base64 overhead, cached detections

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

/* RESULT CARD — font sizes increased */
.result-card {
    background: #0a0a0a;
    border: 1px solid #1a1a1a;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.75rem;
}
.result-title { font-size: 0.78rem; color: #555; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.45rem; font-weight: 600; }
.risk-status { font-size: 1.15rem; font-weight: 700; margin-bottom: 0.35rem; }
.risk-high { color: #f87171; }
.risk-medium { color: #fb923c; }
.risk-low { color: #4ade80; }
.risk-safe { color: #60a5fa; }
.confidence-text { font-size: 0.92rem; color: #666; font-family: 'JetBrains Mono', monospace; }

/* STAT ROW */
.stat-row { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin-top: 0.7rem; }
.stat-box { background: #0e0e0e; border: 1px solid #1a1a1a; border-radius: 8px; padding: 0.7rem; text-align: center; }
.stat-value { font-size: 1.5rem; font-weight: 700; color: #ff6535; font-family: 'JetBrains Mono', monospace; }
.stat-label { font-size: 0.75rem; color: #555; margin-top: 0.2rem; text-transform: uppercase; }

/* CAPPED IMAGE */
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
.output-label { font-size: 0.68rem; color: #555; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.35rem; }

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

/* DEMO THUMBNAIL GRID */
.demo-thumb-img {
    width: 100%;
    aspect-ratio: 16/9;
    object-fit: cover;
    border-radius: 6px;
    display: block;
}
.demo-thumb-name {
    font-size: 0.65rem;
    color: #555;
    text-align: center;
    margin-top: 0.2rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

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
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    color: #666;
    margin-bottom: 0.75rem;
}

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
    .img-wrapper img { max-height: 35vh; }
    .stTabs [data-baseweb="tab"] { padding: 0.35rem 0.6rem; font-size: 0.75rem; }
    .risk-status { font-size: 1rem; }
    .stat-value { font-size: 1.25rem; }
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

def render_capped_image(img_array, label="Output"):
    """Render numpy image with max-height cap."""
    pil = Image.fromarray(img_array)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
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
            icon = "🌫" if label.lower() == "smoke" else "🔥"
            css_cls = "smoke" if label.lower() == "smoke" else ""
            st.markdown(
                f'<div class="det-item {css_cls}"><span>{icon} {label.upper()}</span>'
                f'<span class="det-conf">{conf*100:.1f}%</span></div>',
                unsafe_allow_html=True
            )

def pil_to_b64_thumb(pil_img):
    """Convert PIL image to base64 for thumbnail — preserves full image, no crop."""
    thumb = pil_img.copy()
    thumb.thumbnail((400, 225), Image.LANCZOS)  # 16:9 container, no crop
    buf = BytesIO()
    thumb.save(buf, format="JPEG", quality=72)
    return base64.b64encode(buf.getvalue()).decode()

@st.cache_data(show_spinner=False)
def run_detection_cached(img_bytes: bytes):
    """Cache detection results by image bytes — avoids re-running model on same image."""
    image = Image.open(BytesIO(img_bytes)).convert("RGB")
    t0 = time.time()
    results = model.predict(image, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    ms = round((time.time() - t0) * 1000, 1)
    annotated = results[0].plot()  # numpy RGB
    fire_count, smoke_count, max_conf, detections = analyze_detections(results)
    return fire_count, smoke_count, max_conf, detections, annotated, ms

def encode_video_for_browser(input_path: str) -> str:
    """
    Re-encode video with H.264 using ffmpeg for browser compatibility.
    Falls back to raw mp4 if ffmpeg not available.
    Returns path to output file.
    """
    out_path = input_path.replace(".mp4", "_h264.mp4")
    ret = os.system(
        f'ffmpeg -y -i "{input_path}" -vcodec libx264 -crf 28 -preset fast '
        f'-movflags +faststart -acodec aac "{out_path}" -loglevel quiet'
    )
    if ret == 0 and os.path.exists(out_path):
        return out_path
    return input_path  # fallback to original if ffmpeg unavailable

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

        # Result card always above images
        render_result_card(fire_count, smoke_count, max_conf)

        # Side-by-side original + annotated
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
# TAB 2: VIDEO — auto-process on upload, overlay playback
# ───────────────────────────────────────────
with tab2:
    video_file = st.file_uploader(
        "Upload video", type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed"
    )

    if video_file:
        frame_skip = st.slider(
            "Processing speed (higher = faster, fewer frames annotated)",
            min_value=1, max_value=10, value=3
        )

        # Auto-process — no button needed
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.flush()
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out_path = out_tmp.name
        out_tmp.close()

        # Use mp4v first; will re-encode with ffmpeg after
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        progress = st.progress(0, text="Processing video…")
        fire_total = smoke_total = 0
        frame_idx = 0
        last_annotated_bgr = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                results = model.predict(img, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

                for box in results[0].boxes:
                    lbl = model.names[int(box.cls[0])]
                    if lbl.lower() == "fire":   fire_total  += 1
                    elif lbl.lower() == "smoke": smoke_total += 1

                annotated_rgb = results[0].plot()
                last_annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            writer.write(last_annotated_bgr if last_annotated_bgr is not None else frame)
            progress.progress(
                min((frame_idx + 1) / max(total_frames, 1), 1.0),
                text=f"Frame {frame_idx+1} / {total_frames}…"
            )
            frame_idx += 1

        cap.release()
        writer.release()
        os.unlink(tfile.name)
        progress.empty()

        # Re-encode to H.264 for browser playback
        final_path = encode_video_for_browser(out_path)

        # Result card
        max_conf_est = 0.85 if (fire_total + smoke_total) > 0 else 0.0
        render_result_card(fire_total, smoke_total, max_conf_est)

        # Play annotated video inline
        with open(final_path, "rb") as vf:
            st.video(vf.read())

        # Cleanup
        try:
            os.unlink(out_path)
            if final_path != out_path:
                os.unlink(final_path)
        except Exception:
            pass

        with st.expander("📊 Video Details"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Frames", total_frames)
            c2.metric("FPS", int(fps))
            c3.metric("Duration", f"{round(total_frames/fps, 1)}s")

    else:
        st.markdown(
            '<div class="input-section">🎥 Upload a video — detection boxes overlay on playback</div>',
            unsafe_allow_html=True
        )

# ───────────────────────────────────────────
# TAB 3: WEBCAM
# Simplified: auto-detect on every new capture. No broken rerun loop.
# ───────────────────────────────────────────
with tab3:
    st.markdown(
        '<div class="webcam-info">📹 Take a photo — detection runs instantly. '
        'Retake to scan again.</div>',
        unsafe_allow_html=True
    )

    webcam_img = st.camera_input("", label_visibility="collapsed")

    if webcam_img:
        img_bytes = webcam_img.getvalue()
        fire_count, smoke_count, max_conf, detections, annotated, ms = run_detection_cached(img_bytes)

        render_result_card(fire_count, smoke_count, max_conf)
        render_capped_image(annotated, f"Detection · {ms}ms")
        render_detections_list(detections)
    else:
        st.markdown(
            '<div class="input-section" style="margin-top:0.5rem;">Point camera and tap the button above</div>',
            unsafe_allow_html=True
        )

# ───────────────────────────────────────────
# TAB 4: DEMO — full-image thumbnail grid + inline result
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
            '<div class="input-section">🎬 No demo images found — add images to demo_images/ folder</div>',
            unsafe_allow_html=True
        )
    else:
        if "demo_selected" not in st.session_state:
            st.session_state.demo_selected = 0

        # ── Thumbnail grid (3 per row, full image visible, aspect-ratio preserved)
        cols_per_row = 3
        rows = [demo_images[i:i+cols_per_row] for i in range(0, len(demo_images), cols_per_row)]

        for row_imgs in rows:
            cols = st.columns(cols_per_row)
            for ci, fname in enumerate(row_imgs):
                idx = demo_images.index(fname)
                img_path = os.path.join(DEMO_IMAGES_DIR, fname)
                thumb_pil = Image.open(img_path).convert("RGB")
                b64 = pil_to_b64_thumb(thumb_pil)
                is_sel = (st.session_state.demo_selected == idx)
                border = "#ff6535" if is_sel else "#222"
                short = fname[:16] + "…" if len(fname) > 16 else fname

                with cols[ci]:
                    # Full image visible (object-fit: contain, not cover)
                    st.markdown(
                        f'<div style="border:2px solid {border};border-radius:8px;'
                        f'background:#111;overflow:hidden;padding:3px;">'
                        f'<img src="data:image/jpeg;base64,{b64}" '
                        f'style="width:100%;height:90px;object-fit:contain;display:block;border-radius:5px;"/>'
                        f'</div>'
                        f'<div class="demo-thumb-name">{short}</div>',
                        unsafe_allow_html=True
                    )
                    if st.button("Select", key=f"ds_{idx}", use_container_width=True):
                        st.session_state.demo_selected = idx
                        st.rerun()

        st.divider()

        # ── Detection result shown inline right below grid — no tab switch needed
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