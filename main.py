from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model("models/emotion_model.h5", compile=False)

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_data = data['image'].split(',')[1]  # remove base64 header
    decoded = base64.b64decode(image_data)

    # Convert image to numpy array
    nparr = np.frombuffer(decoded, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert to grayscale and resize
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face / 255.0
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)

    # Predict
    prediction = model.predict(face)
    emotion = emotion_labels[np.argmax(prediction)]

    return jsonify({"emotion": emotion})

if __name__ == "__main__":
    app.run(debug=True)
