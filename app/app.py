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


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.normpath(
    os.path.join(
        BASE_DIR,
        "..",
        "model",
        "fruit_model.h5"
    )
)

# =========================================================
# LOAD MODEL
# =========================================================

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


fruit_detector = MobileNetV2(weights="imagenet")


# =========================================================
# PREPROCESS IMAGE
# =========================================================

def preprocess_image(img):

    img = cv2.resize(
        img,
        (224, 224)
    )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = img.astype(
        np.float32
    )

    img = preprocess_input(
        img
    )

    return np.expand_dims(
        img,
        axis=0
    )


def get_condition(days):

    if days >= 2:
        return "Fresh 😀"
    if days >= 1:
        return "Medium 😥"
    return "Rotten ❌"


def is_fruit(img):

    processed = preprocess_image(img)
    predictions = fruit_detector.predict(processed, verbose=0)
    decoded = decode_predictions(predictions, top=5)[0]

    fruit_keywords = {
        "apple", "banana", "orange", "pineapple", "lemon", "mango",
        "grape", "strawberry", "watermelon", "papaya", "pear", "peach",
        "plum", "cherry", "pomegranate", "fig", "kiwi", "coconut",
        "avocado"
    }

    return any(
        label.lower() in fruit_keywords
        for _, label, _ in decoded
    )


# =========================================================
# ENCODE IMAGE
# =========================================================

def encode_image(img):

    _, buffer = cv2.imencode(
        ".jpg",
        img
    )

    return base64.b64encode(
        buffer
    ).decode("utf-8")


# =========================================================
# PREDICTION
# =========================================================

def predict_shelf_life(img):

    if not is_fruit(img):
        return None

    processed = preprocess_image(
        img
    )

    days = model.predict(
        processed,
        verbose=0
    )[0][0]

    days = max(0.0, float(days))
    confidence = max(0.0, 100.0 - abs(days - round(days)) * 20.0)

    return days, confidence


# =========================================================
# MAIN ROUTE
# =========================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    result = None

    image_data = None

    if request.method == "POST":

        file = request.files.get(
            "image"
        )

        if not file:

            result = {
                "label": "No image uploaded",
                "confidence": 0
            }

            return render_template(
                "index.html",
                result=result,
                image=None
            )

        # -------------------------------------------------
        # READ IMAGE
        # -------------------------------------------------

        img = cv2.imdecode(
            np.frombuffer(
                file.read(),
                np.uint8
            ),
            cv2.IMREAD_COLOR
        )

        if img is None:

            result = {
                "label": "Invalid image",
                "confidence": 0
            }

            return render_template(
                "index.html",
                result=result,
                image=None
            )

        # -------------------------------------------------
        # IMAGE FOR FRONTEND
        # -------------------------------------------------

        image_data = encode_image(
            img
        )

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        prediction = predict_shelf_life(
            img
        )

        if prediction is None:
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

        days, confidence = prediction
        label = get_condition(days)

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        result = {

            "prediction": round(days, 2),

            "label": label,

            "confidence": round(
                confidence,
                2
            )
        }

    return render_template(
        "index.html",
        result=result,
        image=image_data
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )