import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load your trained model
model = load_model("models/emotion_model.h5")

# Define labels (change this based on your training labels)
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_emotion(image: Image.Image):
    # Resize to 48x48 (or your model's input size)
    image = image.resize((48, 48))
    image = image.convert('L')  # convert to grayscale

    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # normalize

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    return labels[predicted_class]
