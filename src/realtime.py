import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("../models/emotion_model.h5")

# Emotion labels (change these if needed)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale (better for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face ROI (Region of Interest)
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))  # Resize to match model input size
        face = face / 255.0  # Normalize pixel values
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        face = np.expand_dims(face, axis=-1)  # Add channel dimension

        # Predict emotion
        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display emotion text
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the webcam feed with predictions
    cv2.imshow("Real-Time Emotion Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
