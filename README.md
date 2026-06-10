<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black" />
  <img src="https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<h1 align="center">🔥 Fire & Smoke Detection System</h1>

<p align="center">
  <strong>AI-powered fire and smoke detection — custom-trained YOLOv8n model deployed with a Streamlit dashboard</strong>
</p>

<p align="center">
  <a href="https://fire-smoke-detection-app.streamlit.app/">🚀 Live Demo</a> &nbsp;·&nbsp;
  <a href="#-model">Model</a> &nbsp;·&nbsp;
  <a href="#-dataset--training">Dataset & Training</a> &nbsp;·&nbsp;
  <a href="#-installation">Installation</a> &nbsp;·&nbsp;
  <a href="#-features">Features</a>
</p>

---

## 📖 Overview

A real-time fire and smoke detection application built end-to-end — from custom model training to a deployed web interface. The model was trained on 14,000+ images using YOLOv8n and achieves **0.776 mAP50** with **0.718 recall** across fire and smoke classes.

---

## 📊 Model

| Metric | Overall | Smoke | Fire |
|--------|---------|-------|------|
| **Precision** | 0.760 | 0.799 | 0.721 |
| **Recall** | 0.718 | 0.772 | 0.665 |
| **mAP50** | 0.776 | 0.830 | 0.722 |
| **mAP50-95** | 0.440 | 0.505 | 0.374 |

**Architecture:** YOLOv8n (Nano) · **Model size:** 6.2 MB · **Input:** 640×640 · **Classes:** Fire, Smoke

---

## 🗃 Dataset & Training

| | Details |
|---|---|
| **Dataset** | [Smoke-Fire-Detection-YOLO](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo) by sayedgamal99 |
| **Train / Val / Test** | 14,122 / 3,099 / 4,306 images |
| **Baseline Training** | Google Colab (Tesla T4 GPU) · 20 epochs |
| **Fine-tuning** | Kaggle Notebooks (Tesla T4 GPU) · 50 additional epochs |
| **Base Model** | YOLOv8n pretrained on COCO |
| **Total Epochs** | 70 (20 baseline + 50 fine-tuned) |
| **Key Settings** | `iou=0.5`, `cos_lr=True`, `lr0=0.005`, `mixup=0.15`, `copy_paste=0.3` |

**Training improvement over baseline:**

| | Baseline (20 epochs) | Final (70 epochs) |
|---|---|---|
| Recall | 0.664 | **0.718** (+8.1%) |
| mAP50 | 0.727 | **0.776** (+6.7%) |
| Precision | 0.724 | **0.760** (+5.0%) |

---

## ✨ Features

| Mode | Description |
|------|-------------|
| 📷 **Image Upload** | Upload any image — instant detection with side-by-side comparison |
| 🎥 **Video Upload** | Frame-by-frame processing with live progress and aggregate stats |
| 📹 **Webcam** | Capture a snapshot and run instant detection |
| 🎬 **Demo Mode** | Auto-cycling slideshow through `demo_images/` — built for presentations |

**Dashboard includes:** Risk assessment panel · Detection timeline · Inference time · Confidence stats · Download results

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/shreyaa4567/fire-smoke-detection-app.git
cd fire-smoke-detection-app

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run appnew.py
```

App opens at `http://localhost:8501`

---

## 📁 Project Structure

```
fire-smoke-detection-app/
├── appnew.py                 # Main Streamlit application
├── best_shreya_v2.pt         # Trained YOLOv8n model weights
├── requirements.txt          # Python dependencies
├── demo_images/              # Images for Demo Mode slideshow
└── samples/                  # Sample images for quick testing
```

---

## 🛠 Technologies Used

| Technology | Purpose |
|------------|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) | Core programming language |
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) | Web application framework & cloud deployment |
| ![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=flat-square&logo=yolo&logoColor=black) | Object detection model |
| ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white) | Video processing & frame extraction |
| ![Pillow](https://img.shields.io/badge/Pillow-3776AB?style=flat-square&logo=python&logoColor=white) | Image handling & manipulation |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Numerical computation |
| ![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white) | Model fine-tuning (T4 GPU) |
| ![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=black) | Baseline model training |

---

## 👤 Author

**Shreya Singh**
- GitHub: [@shreyaa4567](https://github.com/shreyaa4567)
- LinkedIn: [Shreya Singh](https://www.linkedin.com/in/shreya-singh-35bab7337/)

---

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) — YOLOv8 framework
- [sayedgamal99](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo) — Smoke-Fire-Detection-YOLO dataset
- [Streamlit](https://streamlit.io/) — Web framework and cloud deployment
- [Google Colab](https://colab.research.google.com/) — Baseline model training
- [Kaggle Notebooks](https://www.kaggle.com/code) — Fine-tuning and extended training