import os
import json
import cv2
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

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

TEST_SIZE = 0.20
RANDOM_STATE = 42

CLASS_NAMES = [
    "fresh",
    "medium",
    "rotten"
]

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.normpath(
    os.path.join(
        BASE_DIR,
        "..",
        "dataset"
    )
)

MODEL_SAVE_PATH = os.path.join(
    BASE_DIR,
    "fruit_model.h5"
)

BEST_MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_fruit_model.h5"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(dataset_path):

    data = []
    labels = []

    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_path = os.path.join(
            dataset_path,
            class_name
        )

        if not os.path.isdir(class_path):

            print(
                f"\n[WARNING] Folder not found: "
                f"{class_path}"
            )

            continue

        count = 0

        for img_name in os.listdir(class_path):

            img_path = os.path.join(
                class_path,
                img_name
            )

            # Ignore directories
            if not os.path.isfile(img_path):
                continue

            img = cv2.imread(img_path)

            if img is None:

                print(
                    f"[WARNING] Could not read: "
                    f"{img_path}"
                )

                continue

            # Resize
            img = cv2.resize(
                img,
                (IMG_SIZE, IMG_SIZE)
            )

            # BGR -> RGB
            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            # Float32
            img = img.astype(
                np.float32
            )

            # MobileNetV2 preprocessing
            img = preprocess_input(img)

            data.append(img)
            labels.append(class_index)

            count += 1

        print(
            f"{class_name.upper():<10} -> "
            f"{count} images"
        )

    data = np.array(
        data,
        dtype=np.float32
    )

    labels = np.array(
        labels,
        dtype=np.int32
    )

    return data, labels


# =========================================================
# BUILD MODEL
# =========================================================

def build_model():

    print("\n[INFO] Loading MobileNetV2...")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(
            IMG_SIZE,
            IMG_SIZE,
            3
        ),
        include_top=False,
        weights="imagenet"
    )

    # Freeze base model initially
    base_model.trainable = False

    inputs = tf.keras.Input(
        shape=(
            IMG_SIZE,
            IMG_SIZE,
            3
        )
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

    x = tf.keras.layers.Dropout(
        0.4
    )(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(
        0.3
    )(x)

    # THREE classes:
    # 0 = fresh
    # 1 = medium
    # 2 = rotten

    outputs = tf.keras.layers.Dense(
        3,
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
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
        0.15
    ),

    tf.keras.layers.RandomZoom(
        0.15
    ),

    tf.keras.layers.RandomContrast(
        0.15
    ),

    tf.keras.layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    )

])


# =========================================================
# CREATE TF.DATA DATASETS
# =========================================================

def create_datasets(
    X_train,
    y_train,
    X_test,
    y_test
):

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (
            X_train,
            y_train
        )
    )

    train_dataset = (
        train_dataset
        .shuffle(
            buffer_size=len(X_train),
            seed=RANDOM_STATE
        )
        .batch(BATCH_SIZE)
        .map(
            lambda x, y: (
                data_augmentation(
                    x,
                    training=True
                ),
                y
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .prefetch(
            tf.data.AUTOTUNE
        )
    )

    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (
            X_test,
            y_test
        )
    )

    validation_dataset = (
        validation_dataset
        .batch(BATCH_SIZE)
        .prefetch(
            tf.data.AUTOTUNE
        )
    )

    return (
        train_dataset,
        validation_dataset
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("        FRUITIEPY - FRUIT FRESHNESS MODEL")
    print("=" * 60)

    # =====================================================
    # LOAD DATA
    # =====================================================

    X, y = load_dataset(
        DATASET_PATH
    )

    if len(X) == 0:

        print(
            "\n[ERROR] No images found!"
        )

        print(
            "\nExpected structure:"
        )

        print(
            """
dataset/
├── fresh/
├── medium/
└── rotten/
            """
        )

        return

    print(
        f"\n[INFO] Total images: {len(X)}"
    )

    # =====================================================
    # CHECK CLASS DISTRIBUTION
    # =====================================================

    print("\n[INFO] Class distribution:")

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        count = np.sum(
            y == index
        )

        print(
            f"{class_name:<10}: {count}"
        )

    # =====================================================
    # TRAIN / VALIDATION SPLIT
    # =====================================================

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

    # =====================================================
    # CLASS WEIGHTS
    # =====================================================

    class_weights_array = compute_class_weight(

        class_weight="balanced",

        classes=np.unique(y_train),

        y=y_train
    )

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight
        in zip(
            np.unique(y_train),
            class_weights_array
        )
    }

    print("\n[INFO] Class weights:")

    for class_id, weight in class_weights.items():

        print(
            f"{CLASS_NAMES[class_id]:<10}: "
            f"{weight:.3f}"
        )

    # =====================================================
    # CREATE DATASETS
    # =====================================================

    (
        train_dataset,
        validation_dataset
    ) = create_datasets(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # =====================================================
    # BUILD MODEL
    # =====================================================

    model, base_model = build_model()

    print("\n[INFO] Model created.")

    model.summary()

    # =====================================================
    # CALLBACKS
    # =====================================================

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
            min_lr=1e-7,
            verbose=1
        ),

        ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )

    ]

    # =====================================================
    # PHASE 1
    # =====================================================

    print("\n" + "=" * 60)
    print("PHASE 1 - TRAINING CLASSIFIER")
    print("=" * 60)

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]
    )

    model.fit(

        train_dataset,

        validation_data=validation_dataset,

        epochs=EPOCHS_PHASE1,

        class_weight=class_weights,

        callbacks=callbacks
    )

    # =====================================================
    # PHASE 2 - FINE TUNING
    # =====================================================

    print("\n" + "=" * 60)
    print("PHASE 2 - FINE TUNING MOBILENETV2")
    print("=" * 60)

    base_model.trainable = True

    # Freeze most layers
    for layer in base_model.layers[:-30]:

        layer.trainable = False

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]
    )

    model.fit(

        train_dataset,

        validation_data=validation_dataset,

        epochs=EPOCHS_PHASE2,

        class_weight=class_weights,

        callbacks=callbacks
    )

    # =====================================================
    # LOAD BEST MODEL
    # =====================================================

    if os.path.exists(
        BEST_MODEL_PATH
    ):

        print(
            "\n[INFO] Loading best model..."
        )

        model = tf.keras.models.load_model(
            BEST_MODEL_PATH
        )

    # =====================================================
    # EVALUATION
    # =====================================================

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    loss, accuracy = model.evaluate(
        validation_dataset,
        verbose=1
    )

    print(
        f"\nValidation Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # =====================================================
    # PREDICTIONS
    # =====================================================

    print(
        "\n[INFO] Generating validation predictions..."
    )

    predictions = model.predict(
        validation_dataset,
        verbose=1
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_test,
            predicted_classes,
            target_names=CLASS_NAMES,
            digits=4
        )
    )

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    cm = confusion_matrix(
        y_test,
        predicted_classes
    )

    print(
        "\n             Predicted"
    )

    print(
        "             Fresh  Medium  Rotten"
    )

    for i, row in enumerate(cm):

        print(
            f"{CLASS_NAMES[i]:<10} "
            f"{row}"
        )

    # =====================================================
    # SAVE MODEL
    # =====================================================

    model.save(
        MODEL_SAVE_PATH
    )

    print(
        f"\n[INFO] Final model saved to:"
        f"\n{MODEL_SAVE_PATH}"
    )

    # =====================================================
    # SAVE CLASS NAMES
    # =====================================================

    with open(
        CLASS_NAMES_PATH,
        "w"
    ) as f:

        json.dump(
            CLASS_NAMES,
            f,
            indent=4
        )

    print(
        f"\n[INFO] Class names saved to:"
        f"\n{CLASS_NAMES_PATH}"
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()