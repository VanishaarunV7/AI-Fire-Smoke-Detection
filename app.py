from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import pygame
import time
import os

app = Flask(__name__)

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Initialize pygame
pygame.mixer.init()

# Open webcam
camera = cv2.VideoCapture(0)

# Create detections folder
if not os.path.exists("detections"):
    os.makedirs("detections")

# Counters
fire_count = 0
smoke_count = 0

# Live values
latest_confidence = 0
latest_detection = "No Detection"

# Alert status
alert_status = "🟢 System Safe"
uncertainty_level = "Low"


def generate_frames():

    global fire_count, smoke_count
    global latest_confidence, latest_detection
    global alert_status
    global uncertainty_level

    while True:

        success, frame = camera.read()

        if not success:
            break

        # Run detection
        results = model(frame, verbose=False)

        for result in results:

            boxes = result.boxes

            for box in boxes:

                confidence = float(box.conf)

                class_id = int(box.cls)

                class_name = model.names[class_id]

                latest_confidence = int(confidence * 100)

                latest_detection = class_name

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                                # Alert + Uncertainty Logic
                if confidence < 0.60:

                    label = f"{class_name} : {latest_confidence}% ⚠️"

                    color = (0, 0, 255)

                    alert_status = "⚠️ Uncertain Detection"

                    uncertainty_level = "🔴 High"

                elif confidence < 0.80:

                    label = f"{class_name} : {latest_confidence}%"

                    color = (0, 255, 255)

                    alert_status = f"🟡 {class_name.upper()} DETECTED"

                    uncertainty_level = "🟡 Medium"

                else:

                    label = f"{class_name} : {latest_confidence}%"

                    color = (0, 255, 0)

                    alert_status = f"🔥 {class_name.upper()} DETECTED"

                    uncertainty_level = "🟢 Low"
                # Count detections
                if class_name == "fire":
                    fire_count += 1

                if class_name == "smoke":
                    smoke_count += 1

                # Save screenshots
                timestamp = int(time.time())

                filename = f"detections/{class_name}_{timestamp}.jpg"

                cv2.imwrite(filename, frame)

                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # Put label
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        # Convert frame
        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )


@app.route('/')
def index():

    return render_template(
        'index.html',
        fire_count=fire_count,
        smoke_count=smoke_count,
        latest_confidence=latest_confidence,
        latest_detection=latest_detection,
        alert_status=alert_status,
        uncertainty_level=uncertainty_level
    )


@app.route('/video')
def video():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':

    app.run(debug=True)