# 📂 Project Structure

Detailed explanation of every file and folder in the Fire & Smoke Detection System.

---

## Directory Tree

```
fire-smoke-detection-app/
│
├── appnew.py                 # Main application
├── best_shreya_v2.pt         # YOLO model weights
├── requirements.txt          # Dependencies
├── .gitignore                # Git ignore rules
├── LICENSE                   # MIT License
├── README.md                 # Main documentation
├── CONTRIBUTING.md           # Contribution guide
├── PROJECT_STRUCTURE.md      # This file
├── RELEASE_CHECKLIST.md      # Release preparation
│
├── assets/                   # README screenshots
│   └── README.md
│
├── demo_images/              # Demo Mode images
│   └── README.md
│
├── samples/                  # Quick Test images
│   └── README.md
│
├── cctv_detect.py            # CCTV detection (standalone)
└── autolabel.py              # Auto-labeling utility
```

---

## Core Application

### `appnew.py` — Main Streamlit Application

The heart of the project. A single-file Streamlit app (~900 lines) containing:

| Section | Lines | Purpose |
|---------|-------|---------|
| Imports & Config | 1–30 | Dependencies and configuration constants |
| Custom CSS | 31–475 | Complete dark industrial design system |
| Header | 476–495 | Branded header with model metric badges |
| Model Loading | 496–505 | Cached YOLO model initialization |
| Session State | 506–530 | State management for demo mode, video, samples |
| Helper Functions | 531–570 | Utility functions (icon mapping, file listing) |
| `render_detection_results()` | 571–720 | **Core reusable function** — renders all detection UI |
| `render_timeline()` | 721–755 | Detection timeline visualization |
| `render_statistics()` | 756–800 | Aggregate statistics panel |
| Mode Selector | 801–820 | Horizontal radio navigation |
| Image Upload Tab | 821–870 | Standard image upload and detection |
| Quick Test Tab | 871–930 | One-click sample image testing |
| Video Upload Tab | 931–1050 | Video processing with re-encoded output |
| Webcam Tab | 1051–1090 | Webcam snapshot capture |
| Demo Mode Tab | 1091–1170 | Auto-cycling slideshow with controls |
| Auto-Rerun Logic | 1171–1180 | Session-state-driven demo cycling |
| Footer | 1181–1210 | Professional project information |

**Key design decisions:**
- Single file for Streamlit simplicity — no multi-page complexity
- `render_detection_results()` is reused across all 5 modes
- `st.radio(horizontal=True)` with `on_change` callback for mode switching (persists across `st.rerun()`)
- `st.session_state` for all stateful behavior (demo cycling, video results, sample selection)
- Custom CSS injected via `st.markdown(unsafe_allow_html=True)` for premium dark UI

---

### `best_shreya_v2.pt` — Model Weights

Custom-trained YOLOv8n model file (6.2 MB).

| Property | Value |
|----------|-------|
| Architecture | YOLOv8 Nano |
| Dataset | Smoke-Fire-Detection-YOLO |
| Classes | `fire`, `smoke` |
| Precision | 0.760 |
| Recall | 0.718 |
| mAP50 | 0.776 |

Loaded once via `@st.cache_resource` to avoid reloading on every Streamlit rerun.

---

## Configuration Files

### `requirements.txt` — Python Dependencies

Minimal dependency list derived from `appnew.py` imports:

```
streamlit>=1.28.0      # Web framework
ultralytics>=8.0.0     # YOLOv8 model inference
Pillow>=9.0.0          # Image processing
numpy>=1.23.0          # Array operations
opencv-python>=4.7.0   # Video processing & frame extraction
```

### `.gitignore` — Git Ignore Rules

Excludes: Python cache, virtual environments, IDE files, YOLO training runs, temporary output files, OS artifacts.

### `LICENSE` — MIT License

Permissive open-source license allowing free use, modification, and distribution.

---

## Documentation Files

### `README.md` — Main Project Documentation

The primary repository documentation with:
- Shields.io badges, feature overview, architecture diagram
- Installation steps, usage guide, model performance table
- Project structure, technologies, future improvements

### `CONTRIBUTING.md` — Contribution Guidelines

Instructions for contributors: code style (PEP 8, type hints), commit conventions, PR process, issue templates.

### `PROJECT_STRUCTURE.md` — This File

Detailed breakdown of every file and folder in the project.

### `RELEASE_CHECKLIST.md` — Release Preparation Guide

Step-by-step checklist for preparing a GitHub release: files to include, screenshots to capture, testing steps, deployment instructions.

---

## Asset Directories

### `assets/` — README Screenshots

Stores screenshots and diagrams referenced by `README.md`.

**Required files:**
- `dashboard.png` — Full app overview
- `detection.png` — Image detection results
- `demo-mode.png` — Demo Mode in action
- `video-processing.png` — Video upload results

### `demo_images/` — Demo Mode Images

Contains fire and smoke images for the Demo Mode auto-slideshow.

- Add 5–10 images with clear fire/smoke content
- Supported formats: JPG, PNG, BMP, WebP
- Images are sorted alphabetically and cycled automatically

### `samples/` — Quick Test Sample Images

Contains named images for one-click Quick Test buttons.

- Filename becomes button label: `wildfire.jpg` → `🔥 Wildfire`
- Icons auto-assigned based on keywords in filename
- Supported formats: JPG, PNG, BMP, WebP

---

## Utility Scripts

### `cctv_detect.py` — CCTV Detection Script

Standalone script for running YOLO detection on CCTV/webcam feeds outside the Streamlit app. Useful for direct OpenCV video processing.

### `autolabel.py` — Auto-Labeling Utility

Helper script for auto-labeling images using the trained model. Useful for expanding the training dataset with model-assisted annotations.

---

## Files NOT in Repository

These files are generated at runtime and excluded by `.gitignore`:

| File | Purpose |
|------|---------|
| `converted.jpg` | Temporary RGBA→RGB conversion output |
| `result.jpg` | Temporary detection result (legacy) |
| `runs/` | YOLO training/inference run artifacts |
| `.venv/` | Python virtual environment |
| `__pycache__/` | Python bytecode cache |
