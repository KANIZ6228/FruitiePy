from flask import Flask, request, render_template

import tensorflow as tf
import numpy as np
import cv2
import os
import json
import base64

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


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

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


# =========================================================
# LOAD CLASS NAMES
# =========================================================

with open(
    CLASS_NAMES_PATH,
    "r"
) as f:

    class_names = json.load(f)


print(
    "[INFO] Loaded classes:"
)

print(class_names)


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

def predict_fruit(img):

    processed = preprocess_image(
        img
    )

    predictions = model.predict(
        processed,
        verbose=0
    )[0]

    predicted_index = np.argmax(
        predictions
    )

    confidence = (
        predictions[predicted_index]
        * 100
    )

    predicted_class = class_names[
        predicted_index
    ]

    # Example:
    # banana_fresh

    parts = predicted_class.split(
        "_"
    )

    condition = parts[-1]

    fruit = "_".join(
        parts[:-1]
    )

    return (
        fruit,
        condition,
        float(confidence)
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

        fruit, condition, confidence = predict_fruit(
            img
        )

        # -------------------------------------------------
        # FORMAT LABEL
        # -------------------------------------------------

        condition_display = {
            "fresh": "Fresh 😀",
            "medium": "Medium 😥",
            "rotten": "Rotten ❌"
        }

        label = condition_display.get(
            condition,
            condition.capitalize()
        )

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        result = {

            "fruit": fruit.capitalize(),

            "prediction": label,

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