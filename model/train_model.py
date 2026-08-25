import os
import json
import cv2
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# =========================================================
# CONFIG
# =========================================================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 15

TEST_SIZE = 0.2
RANDOM_STATE = 42

DATASET_PATH = "../dataset"
MODEL_SAVE_PATH = "fruit_model.h5"
CLASS_NAMES_PATH = "class_names.json"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "dataset"))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "fruit_model.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(dataset_path):

    data = []
    labels = []

    class_names = []

    print("\n[INFO] Loading dataset...")

    for fruit_name in sorted(os.listdir(dataset_path)):

        fruit_path = os.path.join(dataset_path, fruit_name)

        if not os.path.isdir(fruit_path):
            continue

        for condition in ["fresh", "medium", "rotten"]:

            condition_path = os.path.join(
                fruit_path,
                condition
            )

            if not os.path.isdir(condition_path):
                continue

            class_name = f"{fruit_name}_{condition}"

            if class_name not in class_names:
                class_names.append(class_name)

            class_index = class_names.index(class_name)

            count = 0

            for img_name in os.listdir(condition_path):

                img_path = os.path.join(
                    condition_path,
                    img_name
                )

                img = cv2.imread(img_path)

                if img is None:
                    continue

                img = cv2.resize(
                    img,
                    (IMG_SIZE, IMG_SIZE)
                )

                img = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2RGB
                )

                img = img.astype(np.float32)

                img = preprocess_input(img)

                data.append(img)
                labels.append(class_index)

                count += 1

            print(
                f"[{class_name}] "
                f"{count} images"
            )

    print("\n[INFO] Classes found:")

    for i, class_name in enumerate(class_names):
        print(f"{i}: {class_name}")

    return (
        np.array(data, dtype=np.float32),
        np.array(labels, dtype=np.int32),
        class_names
    )


# =========================================================
# BUILD MODEL
# =========================================================

def build_model(num_classes):

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    inputs = tf.keras.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    x = base_model(
        inputs,
        training=False
    )

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs
    )

    return model, base_model


# =========================================================
# DATA AUGMENTATION
# =========================================================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip(
        "horizontal"
    ),

    tf.keras.layers.RandomRotation(
        0.1
    ),

    tf.keras.layers.RandomZoom(
        0.15
    ),

    tf.keras.layers.RandomContrast(
        0.1
    )
])


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("        FruitiePy - Fruit Freshness Classifier")
    print("=" * 60)

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    X, y, class_names = load_dataset(
        DATASET_PATH
    )

    if len(X) == 0:

        print(
            "\n[ERROR] No images found."
        )

        return

    print(
        f"\n[INFO] Total images: {len(X)}"
    )

    print(
        f"[INFO] Number of classes: "
        f"{len(class_names)}"
    )

    # -----------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y
    )

    print(
        f"\n[INFO] Training images: "
        f"{len(X_train)}"
    )

    print(
        f"[INFO] Validation images: "
        f"{len(X_test)}"
    )

    # -----------------------------------------------------
    # BUILD MODEL
    # -----------------------------------------------------

    model, base_model = build_model(
        len(class_names)
    )

    model.summary()

    # -----------------------------------------------------
    # PHASE 1
    # -----------------------------------------------------

    print(
        "\n[PHASE 1] Training classifier..."
    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)
    )

    train_dataset = (
        train_dataset
        .shuffle(1000)
        .batch(BATCH_SIZE)
        .map(
            lambda x, y: (
                data_augmentation(x, training=True),
                y
            )
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (X_test, y_test)
    )

    validation_dataset = (
        validation_dataset
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    callbacks = [

        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),

        ModelCheckpoint(
            "best_fruit_model.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    model.fit(

        train_dataset,

        validation_data=validation_dataset,

        epochs=EPOCHS_PHASE1,

        callbacks=callbacks
    )

    # -----------------------------------------------------
    # PHASE 2 - FINE TUNING
    # -----------------------------------------------------

    print(
        "\n[PHASE 2] Fine-tuning MobileNetV2..."
    )

    base_model.trainable = True

    for layer in base_model.layers[:-30]:

        layer.trainable = False

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    model.fit(

        train_dataset,

        validation_data=validation_dataset,

        epochs=EPOCHS_PHASE2,

        callbacks=callbacks
    )

    # -----------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------

    print(
        "\n[STEP] Evaluating model..."
    )

    loss, accuracy = model.evaluate(
        validation_dataset
    )

    print(
        f"\nValidation Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------

    model.save(
        MODEL_SAVE_PATH
    )

    print(
        f"\n[INFO] Model saved to: "
        f"{MODEL_SAVE_PATH}"
    )

    # -----------------------------------------------------
    # SAVE CLASS NAMES
    # -----------------------------------------------------

    with open(
        CLASS_NAMES_PATH,
        "w"
    ) as f:

        json.dump(
            class_names,
            f,
            indent=4
        )

    print(
        f"[INFO] Classes saved to: "
        f"{CLASS_NAMES_PATH}"
    )

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()