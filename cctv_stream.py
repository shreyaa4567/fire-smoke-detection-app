"""
cctv_stream.py — CCTV / RTSP Stream Inference
==============================================
Run fire and smoke detection on a live CCTV or IP camera stream.
Uses the same YOLOv8n model as the Streamlit dashboard.

Usage:
    python cctv_stream.py                          # webcam (device 0)
    python cctv_stream.py --source 0               # webcam (device 0)
    python cctv_stream.py --source rtsp://ip/stream  # RTSP stream
    python cctv_stream.py --source video.mp4       # video file
    python cctv_stream.py --conf 0.25 --save       # save output video

Author: Shreya Singh
"""

import argparse
import cv2
import time
from ultralytics import YOLO

# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
MODEL_PATH        = "best_shreya_v2.pt"
CONFIDENCE        = 0.15
IOU               = 0.45
WINDOW_TITLE      = "Fire & Smoke Detection — press Q to quit"
ALERT_CONF        = 0.50          # confidence threshold to print alert


def parse_args():
    parser = argparse.ArgumentParser(description="Fire & Smoke CCTV stream inference")
    parser.add_argument("--source", default="0",
                        help="Stream source: 0 (webcam), RTSP URL, or video file path")
    parser.add_argument("--model", default=MODEL_PATH,
                        help="Path to YOLOv8 weights (default: best_shreya_v2.pt)")
    parser.add_argument("--conf", type=float, default=CONFIDENCE,
                        help="Detection confidence threshold (default: 0.15)")
    parser.add_argument("--iou", type=float, default=IOU,
                        help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated output to output.mp4")
    parser.add_argument("--no-display", action="store_true",
                        help="Suppress live window (useful for headless servers)")
    return parser.parse_args()


def get_risk_label(max_conf: float) -> str:
    if max_conf >= 0.80:
        return "HIGH RISK"
    elif max_conf >= 0.50:
        return "MEDIUM RISK"
    elif max_conf > 0:
        return "LOW RISK"
    return "ALL CLEAR"


def draw_overlay(frame, fire_count: int, smoke_count: int,
                 max_conf: float, fps: float):
    """Draw HUD overlay on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    risk = get_risk_label(max_conf)
    color_map = {
        "HIGH RISK":   (60,  60,  220),
        "MEDIUM RISK": (30, 140, 255),
        "LOW RISK":    (60, 200,  60),
        "ALL CLEAR":   (180, 180, 180),
    }
    color = color_map[risk]

    cv2.putText(frame, f"STATUS: {risk}",
                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Fire: {fire_count}  Smoke: {smoke_count}  "
                        f"Conf: {max_conf*100:.0f}%  FPS: {fps:.1f}",
                (w - 420, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 200), 1, cv2.LINE_AA)


def main():
    args = parse_args()

    # Convert "0" string to int for webcam
    source = int(args.source) if args.source.isdigit() else args.source

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] Opening stream: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {source}")
        return

    # Optional video writer
    writer = None
    if args.save:
        fps_src  = cap.get(cv2.CAP_PROP_FPS) or 25
        width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter("output.mp4", fourcc, fps_src, (width, height))
        print(f"[INFO] Saving output to output.mp4")

    print("[INFO] Running — press Q to quit\n")

    fps_timer = time.time()
    fps       = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Stream ended or frame not available.")
            break

        # Run inference
        results = model.predict(frame, conf=args.conf, iou=args.iou, verbose=False)

        # Parse detections
        fire_count = smoke_count = 0
        max_conf = 0.0
        for box in results[0].boxes:
            label = model.names[int(box.cls[0])]
            conf  = float(box.conf[0])
            max_conf = max(max_conf, conf)
            if label.lower() == "fire":
                fire_count += 1
            elif label.lower() == "smoke":
                smoke_count += 1

        # Alert to terminal
        if max_conf >= ALERT_CONF:
            print(f"[ALERT] {get_risk_label(max_conf)} — "
                  f"Fire: {fire_count}, Smoke: {smoke_count}, "
                  f"Conf: {max_conf*100:.1f}%")

        # Annotated frame
        annotated = results[0].plot()

        # FPS calculation
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - fps_timer, 1e-6))
        fps_timer = now

        draw_overlay(annotated, fire_count, smoke_count, max_conf, fps)

        if writer:
            writer.write(annotated)

        if not args.no_display:
            cv2.imshow(WINDOW_TITLE, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Quit.")
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()