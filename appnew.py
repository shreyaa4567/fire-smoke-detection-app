"""Fire & Smoke Detection System - v5

================================================
Changes from v4:
- Confidence threshold slider in sidebar (default 0.25, was hardcoded 0.15)
- Video: fix risk level to use actual max_conf across all frames
- Video: detection timeline showing timestamps of fire/smoke events
- Video: download button for annotated frames as ZIP
- Performance: reduced redundant re-renders in video/webcam tabs
- Webcam label updated to reflect snapshot (not live)

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
import zipfile
from io import BytesIO

# ═══════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════
MODEL_PATH    = "best_shreya_v2.pt"
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

header[data-testid="stHeader"] { height: 0 !important; min-height: 0 !important; }
div[data-testid="stToolbar"] { display: none !important; }
.block-container { padding-top: 1rem !important; }

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

.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: transparent; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; border: 1px solid #1e1e1e; background: #0e0e0e; padding: 0.45rem 0.9rem; font-size: 0.85rem; }
.stTabs [aria-selected="true"] { border-color: rgba(255,80,0,0.4); background: rgba(255,60,0,0.08); }

.result-card {
    background: #0a0a0a;
    border: 1px solid #1a1a1a;
    border-radius: 12px;
    padding: 0.45rem 1rem;
    margin-bottom: 0.4rem;
}
.result-title { font-size: 0.7rem; color: #555; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.2rem; font-weight: 600; }
.risk-status { font-size: 1rem; font-weight: 700; margin-bottom: 0.15rem; }
.risk-high   { color: #f87171; }
.risk-medium { color: #fb923c; }
.risk-low    { color: #4ade80; }
.risk-safe   { color: #60a5fa; }
.confidence-text { font-size: 0.8rem; color: #666; font-family: 'JetBrains Mono', monospace; }
.stat-row { display: grid; grid-template-columns: 1fr 1fr; gap: 0.35rem; margin-top: 0.35rem; }
.stat-box { background: #0e0e0e; border: 1px solid #1a1a1a; border-radius: 8px; padding: 0.3rem; text-align: center; }
.stat-value { font-size: 1.1rem; font-weight: 700; color: #ff6535; font-family: 'JetBrains Mono', monospace; }
.stat-label { font-size: 0.65rem; color: #555; margin-top: 0.1rem; text-transform: uppercase; }

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

div[data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: 0.5rem; }
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] { min-width: 0 !important; flex: 1 1 0 !important; }

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

.input-section {
    background: #0a0a0a;
    border: 1px dashed #1e1e1e;
    border-radius: 12px;
    padding: 1.2rem;
    color: #555;
    text-align: center;
    font-size: 0.95rem;
}

.webcam-info {
    background: #0a0a0a;
    border: 1px solid #1a1a1a;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    font-size: 0.85rem;
    color: #666;
    margin-bottom: 0.75rem;
}

/* Timeline */
.timeline-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.3rem 0.6rem;
    border-left: 2px solid #1a1a1a;
    margin-bottom: 0.2rem;
    font-size: 0.8rem;
}
.timeline-item.fire  { border-left-color: #ff4500; }
.timeline-item.smoke { border-left-color: #888; }
.timeline-ts { font-family: 'JetBrains Mono', monospace; color: #555; min-width: 42px; }
.timeline-label { color: #aaa; }
.timeline-conf  { color: #555; margin-left: auto; font-family: 'JetBrains Mono', monospace; }

.footer {
    text-align: center; padding-top: 0.75rem;
    border-top: 1px solid #1a1a1a; margin-top: 1.2rem;
    font-size: 0.7rem; color: #333;
}
.footer a { color: #ff6535; text-decoration: none; }

.demo-card { cursor: pointer; transition: transform 0.15s; }
.demo-card:hover { transform: scale(1.04); filter: brightness(1.15); }
/* Cap video frame size */
[data-testid="stImage"] img {
    max-height: 50vh !important; object-fit: contain !important;
}
/* Limit camera live preview height */
[data-testid="stCameraInput"] video {
    max-height: 280px !important; object-fit: contain;
}

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
# SIDEBAR — Confidence slider
# ═══════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")
    CONFIDENCE_THRESHOLD = st.slider(
        "Confidence Threshold",
        min_value=0.10, max_value=0.90,
        value=0.25, step=0.05,
        help="Lower = more detections (more false positives). Higher = stricter (may miss some)."
    )
    st.caption(f"Current: **{CONFIDENCE_THRESHOLD:.2f}** · Recommended: 0.25–0.40")
    st.markdown("---")
    st.caption("YOLOv8n · mAP50: 0.776 · Recall: 0.718")

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

def arr_to_b64(img_array: np.ndarray, quality: int = 70, max_dim: int = 800) -> str:
    pil = Image.fromarray(img_array)
    if max(pil.size) > max_dim:
        pil.thumbnail((max_dim, max_dim), Image.LANCZOS)
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

def render_detection_timeline(timeline: list):
    """Render a timeline of detection events from video processing.
    timeline: list of (timestamp_sec, label, conf)
    """
    if not timeline:
        return
    st.markdown('<div class="expandable-title">Detection Timeline</div>', unsafe_allow_html=True)
    for ts, label, conf in timeline:
        mins = int(ts // 60)
        secs = int(ts % 60)
        ts_str = f"{mins}:{secs:02d}"
        icon = "🔥" if label.lower() == "fire" else "🌫"
        css_cls = "fire" if label.lower() == "fire" else "smoke"
        st.markdown(
            f'<div class="timeline-item {css_cls}">'
            f'<span class="timeline-ts">{ts_str}</span>'
            f'<span class="timeline-label">{icon} {label.upper()}</span>'
            f'<span class="timeline-conf">{conf*100:.0f}%</span>'
            f'</div>',
            unsafe_allow_html=True
        )

def frames_to_mp4(frames: list, fps: float = 10.0) -> bytes:
    """Encode annotated frames into an MP4 video."""
    if not frames:
        return b""
    h, w = frames[0].shape[:2]
    buf = BytesIO()
    # Write to a temp file since cv2.VideoWriter needs a file path
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, fps, (w, h))
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()
    with open(tmp.name, "rb") as f:
        data = f.read()
    os.unlink(tmp.name)
    return data

@st.cache_data(show_spinner=False)
def get_demo_thumb_b64(img_path: str) -> str:
    thumb = Image.open(img_path).convert("RGB")
    thumb.thumbnail((200, 130), Image.LANCZOS)
    buf = BytesIO()
    thumb.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()

@st.cache_data(show_spinner=False)
def run_detection_cached(img_bytes: bytes, conf: float):
    image = Image.open(BytesIO(img_bytes)).convert("RGB")
    orig_array = np.array(image)
    t0 = time.time()
    results = model.predict(image, conf=conf, iou=IOU_THRESHOLD, verbose=False)
    ms = round((time.time() - t0) * 1000, 1)
    annotated = results[0].plot()
    fire_count, smoke_count, max_conf, detections = analyze_detections(results)
    return fire_count, smoke_count, max_conf, detections, annotated, ms, orig_array

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
tab1, tab2, tab3, tab4 = st.tabs(["📷 Image", "🎥 Video", "📸 Webcam", "🎬 Demo"])

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
        fire_count, smoke_count, max_conf, detections, annotated, ms, orig = run_detection_cached(img_bytes, CONFIDENCE_THRESHOLD)

        col_o, col_a = st.columns(2)
        with col_o:
            render_capped_image(orig, "Original")
        with col_a:
            render_capped_image(annotated, f"Annotated · {ms}ms")

        render_result_card(fire_count, smoke_count, max_conf)
        render_detections_list(detections)

        with st.expander("📊 Technical Details"):
            c1, c2 = st.columns(2)
            c1.metric("Inference Time", f"{ms} ms")
            c2.metric("Resolution", f"{orig.shape[1]}×{orig.shape[0]}")

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
# TAB 2: VIDEO
# ───────────────────────────────────────────
with tab2:
    frame_skip = st.slider(
        "Frames to skip (higher = faster processing)",
        min_value=1, max_value=15, value=5,
        help="1 = every frame (slow but thorough), 15 = every 15th frame (fast)"
    )

    video_file = st.file_uploader(
        "Upload video", type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed"
    )

    if video_file:
        # Include confidence in the key so changing the slider re-processes
        vid_key = f"{video_file.name}_{video_file.size}_{CONFIDENCE_THRESHOLD}_{frame_skip}"
        needs_processing = st.session_state.get("vid_processed_key") != vid_key

        if needs_processing:
            st.session_state["vid_processed_key"] = vid_key
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            tfile.flush()
            tfile.close()

            cap          = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps          = cap.get(cv2.CAP_PROP_FPS) or 25

            progress = st.progress(0, text="Processing…")
            fire_total = smoke_total = 0
            max_conf_video = 0.0
            annotated_frames = []
            timeline_events  = []   # (timestamp_sec, label, conf)
            last_annotated   = None
            frame_idx        = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    img     = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    results = model.predict(
                        img, conf=CONFIDENCE_THRESHOLD,
                        iou=IOU_THRESHOLD, verbose=False
                    )
                    timestamp_sec = frame_idx / fps

                    for box in results[0].boxes:
                        lbl    = model.names[int(box.cls[0])]
                        conf_v = float(box.conf[0])
                        max_conf_video = max(max_conf_video, conf_v)
                        if lbl.lower() == "fire":
                            fire_total += 1
                            timeline_events.append((timestamp_sec, "fire", conf_v))
                        elif lbl.lower() == "smoke":
                            smoke_total += 1
                            timeline_events.append((timestamp_sec, "smoke", conf_v))

                    last_annotated = results[0].plot()

                    if len(annotated_frames) < 50 and last_annotated is not None:
                        annotated_frames.append(last_annotated)

                progress.progress(
                    min((frame_idx + 1) / max(total_frames, 1), 1.0),
                    text=f"Frame {frame_idx+1} / {total_frames}…"
                )
                frame_idx += 1

            cap.release()
            os.unlink(tfile.name)
            progress.empty()

            st.session_state["vid_frames"]   = annotated_frames
            st.session_state["vid_fps"]      = fps
            st.session_state["vid_fire"]     = fire_total
            st.session_state["vid_smoke"]    = smoke_total
            st.session_state["vid_total"]    = total_frames
            st.session_state["vid_max_conf"] = max_conf_video
            st.session_state["vid_timeline"] = timeline_events

        if "vid_frames" in st.session_state and st.session_state["vid_frames"]:
            frames      = st.session_state["vid_frames"]
            fps         = st.session_state["vid_fps"]
            fire_total  = st.session_state["vid_fire"]
            smoke_total = st.session_state["vid_smoke"]
            max_conf_vid = st.session_state.get("vid_max_conf", 0.0)
            timeline     = st.session_state.get("vid_timeline", [])

            render_result_card(fire_total, smoke_total, max_conf_vid)

            frame_idx = st.slider(
                "Browse detected frames", 0, len(frames) - 1, 0,
                key="vid_slider"
            )
            st.image(frames[frame_idx], use_container_width=True)
            st.caption(f"Frame {frame_idx + 1} / {len(frames)} · {int(fps)} FPS")

            # Download annotated video
            vid_bytes = frames_to_mp4(frames, fps)
            st.download_button(
                "⬇ Download Annotated Video (MP4)",
                vid_bytes,
                "annotated_video.mp4",
                "video/mp4",
                use_container_width=True
            )

            # Video Details only
            with st.expander("📊 Video Details"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Frames", st.session_state["vid_total"])
                c2.metric("FPS", int(fps))
                c3.metric("Duration", f"{round(st.session_state['vid_total']/fps, 1)}s")

    else:
        for k in ["vid_frames","vid_fps","vid_fire","vid_smoke","vid_total",
                  "vid_processed_key","vid_max_conf","vid_timeline"]:
            st.session_state.pop(k, None)
        st.markdown(
            '<div class="input-section">🎥 Upload a video — detection runs automatically</div>',
            unsafe_allow_html=True
        )

# ───────────────────────────────────────────
# TAB 3: WEBCAM
# ───────────────────────────────────────────
with tab3:
    st.markdown(
        '<div class="webcam-info">📸 Capture a snapshot — detection runs instantly on the photo.</div>',
        unsafe_allow_html=True
    )
    cam_col, ann_col = st.columns(2)
    with cam_col:
        webcam_img = st.camera_input("Capture", label_visibility="collapsed")

    if webcam_img:
        img_bytes = webcam_img.getvalue()
        fire_count, smoke_count, max_conf, detections, annotated, ms, orig = run_detection_cached(img_bytes, CONFIDENCE_THRESHOLD)

        with ann_col:
            st.image(annotated, caption=f"Annotated · {ms}ms", use_container_width=True)

        render_result_card(fire_count, smoke_count, max_conf)
        render_detections_list(detections)
    else:
        with ann_col:
            st.markdown(
                '<div class="input-section" style="margin-top:2rem;">Point camera and tap the capture button</div>',
                unsafe_allow_html=True
            )

# ───────────────────────────────────────────
# TAB 4: DEMO
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
                b64      = get_demo_thumb_b64(img_path)
                is_sel   = (st.session_state.demo_selected == idx)
                border   = "#ff6535" if is_sel else "#222"
                short    = fname[:12] + "…" if len(fname) > 12 else fname

                with cols[ci]:
                    st.markdown(
                        f'<div class="demo-card" style="border:2px solid {border};border-radius:6px;'
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
                    if st.button("Select", key=f"ds_{idx}", use_container_width=True):
                        st.session_state.demo_selected = idx
                        st.rerun()

        st.divider()

        sel_fname = demo_images[st.session_state.demo_selected]
        sel_path  = os.path.join(DEMO_IMAGES_DIR, sel_fname)

        with open(sel_path, "rb") as f:
            img_bytes = f.read()

        fire_count, smoke_count, max_conf, detections, annotated, ms, orig = run_detection_cached(img_bytes, CONFIDENCE_THRESHOLD)

        col_o, col_a = st.columns(2)
        with col_o:
            render_capped_image(orig, "Original")
        with col_a:
            render_capped_image(annotated, f"Annotated · {ms}ms")

        render_result_card(fire_count, smoke_count, max_conf)
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