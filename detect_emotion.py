#!/usr/bin/env python3

import cv2
import numpy as np
from ultralytics import YOLO

# Function to draw a fancy border around the bounding box
def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2

    # Draw top left corner
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Draw top right corner
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Draw bottom left corner
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Draw bottom right corner
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


# Path to the trained YOLOv8 model
model_path = '/home/omar_ben_emad/State of The Art/Facial_Emotion/runs/detect/train2/weights/best.pt'

# Load the YOLOv8 model
model = YOLO(model_path)

# Emotion mapping to filter invalid emotions
valid_emotions = {"Happy": "Happy", "Neutral": "Neutral", "Surprised": "Surprised", "Sad": "Neutral"}

# Emotion-specific colors (BGR)
emotion_colors = {
    "Neutral": (255, 0, 0),    # Blue
    "Happy": (0, 255, 0),      # Green
    "Surprised": (0, 165, 255) # Orange
}

# Start webcam capture
cap = cv2.VideoCapture(0)
confidence_threshold = 0.5

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Function to handle the detection and display of emotions
def process_frame(frame):
    results = model.predict(frame, conf=confidence_threshold)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = box.cls[0]
        confidence = box.conf[0]

        # Get the emotion label
        emotion = model.names[int(label)]
        emotion = valid_emotions.get(emotion, None)

        if emotion:
            # Set the color for the bounding box based on emotion
            color = emotion_colors.get(emotion, (255, 255, 255))

            # Draw the fancy bounding box
            draw_border(frame, (x1, y1), (x2, y2), color, 2, r=15, d=20)

            # Display the emotion and confidence on the frame
            cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

# Main loop for webcam capture and processing
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image from webcam.")
        break

    # Process the captured frame
    processed_frame = process_frame(frame)

    # Show the frame with emotion annotations
    cv2.imshow('YOLOv8 Fancy Bounding Boxes', processed_frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
