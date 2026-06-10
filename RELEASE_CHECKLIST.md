# 🚀 Release Checklist

Step-by-step guide for preparing the Fire & Smoke Detection System for a public GitHub release.

---

## 1. Files to Include in Repository

Verify all required files are present before pushing:

### Core Files
- [ ] `appnew.py` — Main Streamlit application
- [ ] `best_shreya_v2.pt` — Trained YOLOv8n model weights (6.2 MB)
- [ ] `requirements.txt` — Python dependencies

### Documentation
- [ ] `README.md` — Main project documentation with badges
- [ ] `LICENSE` — MIT License
- [ ] `CONTRIBUTING.md` — Contribution guidelines
- [ ] `PROJECT_STRUCTURE.md` — File/folder explanations
- [ ] `RELEASE_CHECKLIST.md` — This file

### Configuration
- [ ] `.gitignore` — Proper ignore rules active

### Asset Directories
- [ ] `assets/` — Contains README screenshots (capture after setup)
- [ ] `demo_images/` — Contains 5–10 fire/smoke sample images
- [ ] `samples/` — Contains 3–5 named sample images for Quick Test

### Optional / Utility
- [ ] `cctv_detect.py` — CCTV detection script
- [ ] `autolabel.py` — Auto-labeling utility

---

## 2. Screenshots to Capture

Run the app locally (`streamlit run appnew.py`) and capture these screenshots for `assets/`:

| Screenshot | Filename | What to Show |
|------------|----------|--------------|
| Dashboard Overview | `dashboard.png` | Full page with header, mode selector, empty state |
| Image Detection | `detection.png` | Upload result with original vs detection, stat cards, risk panel |
| Demo Mode Running | `demo-mode.png` | Demo cycling with status indicator, timeline, statistics |
| Video Processing | `video-processing.png` | Processed video player, frame stats, timeline |
| Quick Test | `quick-test.png` | Sample buttons + detection result below |
| Webcam Mode | `webcam.png` | Camera capture with detection overlay |

### Screenshot Tips
- Browser: **Chrome** at **1920×1080** for consistent sizing
- Hide browser UI: Press **F11** for fullscreen
- Tool: **Snipping Tool** (Windows) or **ShareX**
- File format: **PNG** for crisp text rendering
- Crop to content — remove address bars and OS chrome

---

## 3. Final Testing Steps

### Core Functionality
- [ ] App starts without errors: `streamlit run appnew.py`
- [ ] Header renders with correct model metrics
- [ ] Mode selector switches between all 5 modes

### Image Upload Mode
- [ ] Upload JPG image → detection runs successfully
- [ ] Upload PNG image → detection runs successfully
- [ ] Info cards show: inference time, detections, confidence, risk
- [ ] Side-by-side images render correctly
- [ ] Stat cards show correct counts
- [ ] Detection list shows each detection with confidence
- [ ] Risk panel shows appropriate level
- [ ] Download button produces valid JPEG file

### Quick Test Mode
- [ ] Sample buttons render with correct icons and names
- [ ] Clicking a sample runs detection immediately
- [ ] Results display correctly below buttons
- [ ] Empty state shows when no samples exist

### Video Upload Mode
- [ ] Upload MP4 → preview displays correctly
- [ ] "Process Video" → progress bar appears and completes
- [ ] Annotated video plays in browser
- [ ] Processing stats (frames, time, FPS, duration) are correct
- [ ] Detection statistics show fire/smoke totals
- [ ] Detection timeline shows timestamped entries
- [ ] Download button produces valid MP4 file

### Webcam Mode
- [ ] Camera permission prompt appears
- [ ] Captured photo runs detection correctly
- [ ] Results display using standard detection UI

### Demo Mode
- [ ] "Run Demo" starts auto-cycling
- [ ] Status indicator shows correct: `Demo Image X / Y`
- [ ] Images cycle every ~2 seconds
- [ ] "Stop Demo" halts the slideshow
- [ ] Detection timeline accumulates across images
- [ ] Statistics update with aggregate data
- [ ] Switching modes stops the demo automatically

### Edge Cases
- [ ] Upload non-image file → handled gracefully
- [ ] Empty `demo_images/` folder → shows helpful message
- [ ] Empty `samples/` folder → shows helpful message
- [ ] Very large image (4000×4000) → processes without crashing
- [ ] Image with no fire/smoke → shows "All Clear" / "No detections"

### Visual Quality
- [ ] Dark theme renders consistently across all modes
- [ ] No white flashes or unstyled elements
- [ ] All CSS animations work (hover effects, pulse border)
- [ ] Footer renders with correct info
- [ ] Responsive layout works on different window sizes

---

## 4. Deployment Checklist

### Streamlit Cloud

1. **Repository**: Push all code to GitHub (public repo)
2. **Go to**: [share.streamlit.io](https://share.streamlit.io/)
3. **Connect**: Link your GitHub account
4. **Deploy**:
   - Repository: `shreyaa4567/fire-smoke-detection-app`
   - Branch: `main`
   - Main file: `appnew.py`
5. **Environment**: Streamlit Cloud auto-installs from `requirements.txt`
6. **Model file**: Ensure `best_shreya_v2.pt` is committed (6.2 MB — within GitHub's 100 MB limit)
7. **Test**: Verify all modes work on the deployed URL

### Common Deployment Issues

| Issue | Solution |
|-------|----------|
| Model file too large | Use Git LFS: `git lfs track "*.pt"` |
| OpenCV import error | Add `opencv-python-headless` to requirements |
| Webcam not working | Expected — webcam requires HTTPS in production |
| Slow video processing | Normal for CPU inference on cloud — add a warning |

> **Tip:** For Streamlit Cloud, replace `opencv-python` with `opencv-python-headless` in `requirements.txt` to avoid display driver issues on Linux servers.

---

## 5. GitHub Repository Setup

### About Section
Copy this into the GitHub "About" field:

> AI-powered fire and smoke detection system with 5 analysis modes — built with YOLOv8 + Streamlit. Features real-time image/video detection, auto-cycling demo mode, webcam capture, risk assessment, and annotated video output.

### Topics / Tags
Add these topics to the repository:

```
fire-detection, smoke-detection, yolov8, streamlit, computer-vision,
deep-learning, object-detection, python, opencv, real-time-detection,
portfolio-project, ai
```

### Social Preview Image
- Use `assets/dashboard.png` as the social preview
- Go to: Settings → Social preview → Upload image
- Recommended size: **1280 × 640 px**

### Repository Settings
- [ ] Description filled (About section)
- [ ] Topics added (12 tags)
- [ ] Social preview image uploaded
- [ ] Website URL set (if deployed to Streamlit Cloud)
- [ ] License detected as MIT

---

## 6. Pre-Push Checks

### Code Quality
- [ ] `python -c "import py_compile; py_compile.compile('appnew.py', doraise=True)"` — No syntax errors
- [ ] No hardcoded absolute paths in code
- [ ] No API keys or secrets in any files
- [ ] No personal data (emails, passwords) in code

### Git Hygiene
- [ ] `.gitignore` is working — run `git status` and verify excluded files don't appear
- [ ] No `__pycache__/` directories staged
- [ ] No `.venv/` directory staged
- [ ] No `runs/` directory staged
- [ ] No `converted.jpg` or `result.jpg` staged
- [ ] Model file `best_shreya_v2.pt` IS included (6.2 MB)

### Final Git Commands

```bash
# Check what will be committed
git status

# Add all files
git add .

# Verify .gitignore is working
git status  # Should NOT show .venv/, __pycache__/, runs/, etc.

# Commit
git commit -m "feat: complete fire & smoke detection app with 5 modes"

# Push
git push origin main
```

---

## 7. Post-Release

- [ ] Verify README renders correctly on GitHub (badges, Mermaid diagram, tables)
- [ ] Check all internal links work (LICENSE, CONTRIBUTING.md, PROJECT_STRUCTURE.md)
- [ ] Star your own repository
- [ ] Share on LinkedIn (see professional descriptions)
- [ ] Pin repository on your GitHub profile
- [ ] Add to resume/portfolio website
