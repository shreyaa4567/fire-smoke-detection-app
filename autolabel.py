from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(
    source="AUTO_LABEL/images",
    save=True,
    save_txt=True,
    save_conf=False,
    project="AUTO_LABEL",
    name="predictions",
    conf=0.40
)

print("Finished")