from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import cv2
import os
import base64

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

app = Flask(__name__)

# =========================
# MODEL PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "..", "model", "fruit_model.h5")
)

# =========================
# LOAD MODELS
# =========================

# Shelf life model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Fruit detection model (ImageNet pretrained)
fruit_detector = MobileNetV2(weights='imagenet')

# =========================
# PREPROCESS IMAGE
# =========================
def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# =========================
# LABEL FUNCTION
# =========================
def get_label(days):
    if days >= 5:
        return "Fresh 😀"
    elif days >= 2:
        return "Medium 😥"
    else:
        return "Rotten ❌"

# =========================
# ENCODE IMAGE FOR FRONTEND
# =========================
def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

# =========================
# FRUIT DETECTION (FIXED)
# =========================
def is_fruit(img):

    resized = cv2.resize(img, (224, 224))
    x = np.expand_dims(resized, axis=0)
    x = preprocess_input(x)

    predictions = fruit_detector.predict(x)
    decoded = decode_predictions(predictions, top=5)[0]

    fruit_keywords = [
        'banana', 'apple', 'orange', 'pineapple', 'lemon',
        'mango', 'grape', 'strawberry', 'watermelon',
        'papaya', 'pear', 'peach', 'plum', 'cherry',
        'pomegranate', 'fig', 'jackfruit', 'custard_apple'
    ]

    # Simple reliable check
    for _, label, _ in decoded:
        label = label.lower()

        for fruit in fruit_keywords:
            if fruit in label:
                return True

    return False

# =========================
# MAIN ROUTE
# =========================
@app.route('/', methods=['GET', 'POST'])
def index():

    result = None
    image_data = None

    if request.method == 'POST':

        file = request.files['image']

        # Read image
        img = cv2.imdecode(
            np.frombuffer(file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        # Encode image for UI
        image_data = encode_image(img)

        # =========================
        # CHECK IF FRUIT
        # =========================
        if not is_fruit(img):

            result = {
                "prediction": "N/A",
                "label": "❌ Not a Fruit",
                "confidence": 0
            }

            return render_template(
                "index.html",
                result=result,
                image=image_data
            )

        # =========================
        # PREDICTION
        # =========================
        processed = preprocess(img)

        prediction = model.predict(processed)[0][0]
        prediction = max(0, prediction)

        label = get_label(prediction)

        confidence = max(
            0,
            100 - abs(prediction - round(prediction)) * 20
        )

        result = {
            "prediction": round(prediction, 2),
            "label": label,
            "confidence": round(confidence, 1)
        }

    return render_template(
        "index.html",
        result=result,
        image=image_data
    )

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)