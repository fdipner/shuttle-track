from ultralytics.models import YOLO


class Test:

    def __init__(self):
        self.x = 0
        pass


model = YOLO("yolov8n.pt")

model.train(
    data=r"./data/processed/iphone_20251015.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="custom_yolo_exp",
    pretrained=True,
)

metrics = model.val(split="val")
# 4️⃣ Print results
print("Evaluation Results:")
print(metrics)
