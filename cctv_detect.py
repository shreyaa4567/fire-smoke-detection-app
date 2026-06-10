import cv2
from ultralytics import YOLO

model = YOLO("best_shreya_v2.pt")

# Your phone's IP Webcam URL
cap = cv2.VideoCapture("http://192.168.1.31:8080/video")

print("Starting live detection... Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read frame, retrying...")
        continue

    results = model.predict(
        frame,
        conf=0.15,
        iou=0.45,
        verbose=False
    )

    annotated = results[0].plot()

    cv2.imshow("Fire & Smoke Detection - Live", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()