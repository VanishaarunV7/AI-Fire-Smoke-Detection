import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run detection
    results = model(frame, verbose=False)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            confidence = float(box.conf)
            class_id = int(box.cls)

            class_name = model.names[class_id]

            # Box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Uncertainty logic
            if confidence < 0.60:
                label = f"{class_name} {confidence:.2f} ⚠️ Uncertain"
            else:
                label = f"{class_name} {confidence:.2f}"

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put text
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Show webcam
    cv2.imshow("Fire & Smoke Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()