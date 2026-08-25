from flask import Flask, request, render_template

import tensorflow as tf
import numpy as np
import cv2
import os
import base64
import json

from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input
)


# =========================================================
# FLASK APP
# =========================================================

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

if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.normpath(
        os.path.join(
            BASE_DIR,
            "..",
            "model",
            "best_fruit_model.h5"
        )
    )

CLASS_NAMES_PATH = os.path.normpath(
    os.path.join(
        BASE_DIR,
        "..",
        "model",
        "class_names.json"
    )
)


# =========================================================
# LOAD MODEL
# =========================================================

print("\n[INFO] Loading freshness model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("[INFO] Model loaded successfully.")


# =========================================================
# LOAD CLASS NAMES
# =========================================================

if os.path.exists(
    CLASS_NAMES_PATH
):

    with open(
        CLASS_NAMES_PATH,
        "r"
    ) as f:

        CLASS_NAMES = json.load(f)

else:

    # Fallback
    CLASS_NAMES = [
        "fresh",
        "medium",
        "rotten"
    ]


print(
    f"[INFO] Classes: {CLASS_NAMES}"
)


# =========================================================
# PREPROCESS IMAGE
# =========================================================

def preprocess_image(img):

    # Resize
    img = cv2.resize(
        img,
        (224, 224)
    )

    # OpenCV BGR -> RGB
    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # Convert to float
    img = img.astype(
        np.float32
    )

    # MobileNetV2 preprocessing
    img = preprocess_input(
        img
    )

    # Add batch dimension
    img = np.expand_dims(
        img,
        axis=0
    )

    return img


# =========================================================
# ENCODE IMAGE
# =========================================================

def encode_image(img):

    success, buffer = cv2.imencode(
        ".jpg",
        img
    )

    if not success:
        return None

    return base64.b64encode(
        buffer
    ).decode("utf-8")


# =========================================================
# PREDICTION
# =========================================================

def predict_freshness(img):

    processed = preprocess_image(
        img
    )

    # Model prediction
    predictions = model.predict(
        processed,
        verbose=0
    )[0]

    # Get highest probability
    predicted_index = int(
        np.argmax(predictions)
    )

    confidence = float(
        predictions[predicted_index]
    ) * 100

    # Get class name
    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    return (
        predicted_class,
        confidence,
        predictions
    )


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

        # -------------------------------------------------
        # GET FILE
        # -------------------------------------------------

        file = request.files.get(
            "image"
        )

        if not file:

            result = {
                "prediction": "N/A",
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
                "prediction": "N/A",
                "label": "Invalid image",
                "confidence": 0
            }

            return render_template(
                "index.html",
                result=result,
                image=None
            )

        # -------------------------------------------------
        # ENCODE IMAGE FOR FRONTEND
        # -------------------------------------------------

        image_data = encode_image(
            img
        )

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        (
            predicted_class,
            confidence,
            probabilities
        ) = predict_freshness(
            img
        )

        # -------------------------------------------------
        # DISPLAY LABEL
        # -------------------------------------------------
        if predicted_class == "fresh":

            label = "Fresh 😀"
            shelf_life = "2–3 days"

        elif predicted_class == "medium":

            label = "Medium 😥"
            shelf_life = "1–2 days"

        elif predicted_class == "rotten":

            label = "Rotten ❌"
            shelf_life = "0 days"

        else:

            label = predicted_class.capitalize()
            shelf_life = "Unknown"

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        result = {

            "prediction": predicted_class.capitalize(),

            "label": label,

            "shelf_life": shelf_life,

            "confidence": round(
                confidence,
                2
            )
        }
        # -------------------------------------------------
        # DEBUG INFORMATION
        # -------------------------------------------------

        print("\n" + "=" * 50)
        print("PREDICTION")
        print("=" * 50)

        print(
            f"Fresh  : {probabilities[0] * 100:.2f}%"
        )

        print(
            f"Medium : {probabilities[1] * 100:.2f}%"
        )

        print(
            f"Rotten : {probabilities[2] * 100:.2f}%"
        )

        print(
            f"\nResult: {label}"
        )

        print(
            f"Confidence: {confidence:.2f}%"
        )

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