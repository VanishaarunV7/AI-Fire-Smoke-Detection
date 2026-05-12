from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Test image
results = model("test.jpg", save=True)

# Print detections
for result in results:

    boxes = result.boxes

    for box in boxes:

        confidence = float(box.conf)

        class_id = int(box.cls)

        class_name = model.names[class_id]

        print(f"\nDetected : {class_name}")

        print(f"Confidence : {confidence:.2f}")

        # Uncertainty logic
        if confidence < 0.60:

            print("⚠️ Uncertain Prediction")

        else:

            print("✅ Confident Prediction")