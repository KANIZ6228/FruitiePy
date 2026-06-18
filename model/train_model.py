"""
train_model.py — Improved FruitiePy Training Script

Key improvements over original:
- Realistic shelf life labels (per fruit type, per condition)
- Data augmentation to improve generalization
- Dropout regularization to prevent overfitting
- Two-phase training: frozen base → fine-tuning top layers
- More epochs with early stopping & learning rate reduction
- Saves best model automatically (not just last epoch)
- Prints dataset summary before training

Expected dataset folder structure:
    ../dataset/
        banana/
            fresh/      ← banana images that are fresh
            medium/
            rotten/
        mango/
            fresh/
            medium/
            rotten/
        apple/
            fresh/
            medium/
            rotten/
        ... (any fruit folder name works)

    OR flat structure (original):
    ../dataset/
        fresh/
        medium/
        rotten/
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# CONFIG
# =========================
IMG_SIZE      = 224
BATCH_SIZE    = 32
EPOCHS_PHASE1 = 15   # frozen base
EPOCHS_PHASE2 = 15   # fine-tune top layers
TEST_SIZE     = 0.2
RANDOM_STATE  = 42
MODEL_SAVE_PATH = "fruit_model.h5"
DATASET_PATH    = "../dataset"

# =========================
# REALISTIC SHELF LIFE MAP
# Per-fruit, per-condition (days remaining)
# Based on general produce science estimates
# =========================
FRUIT_SHELF_LIFE = {
    # fruit_name : { condition: days }
    "banana":     {"fresh": 6,  "medium": 3,  "rotten": 0},
    "mango":      {"fresh": 7,  "medium": 3,  "rotten": 0},
    "apple":      {"fresh": 10, "medium": 7,  "rotten": 0},
    "orange":     {"fresh": 14, "medium": 6,  "rotten": 0},
    "strawberry": {"fresh": 4,  "medium": 2,  "rotten": 0},
    "grape":      {"fresh": 7,  "medium": 3,  "rotten": 0},
    "watermelon": {"fresh": 7,  "medium": 3,  "rotten": 0},
    "pineapple":  {"fresh": 5,  "medium": 2,  "rotten": 0},
    "papaya":     {"fresh": 5,  "medium": 2,  "rotten": 0},
    "pear":       {"fresh": 7,  "medium": 3,  "rotten": 0},
    "peach":      {"fresh": 5,  "medium": 2,  "rotten": 0},
    "lemon":      {"fresh": 11, "medium": 10, "rotten": 0},
    "kiwi":       {"fresh": 7,  "medium": 3,  "rotten": 0},
    "cherry":     {"fresh": 5,  "medium": 2,  "rotten": 0},
    "plum":       {"fresh": 5,  "medium": 2,  "rotten": 0},
    "pomegranate":{"fresh": 14, "medium": 6,  "rotten": 0},
    # Default fallback for unknown/flat dataset structure
    "default":    {"fresh": 7,  "medium": 3,  "rotten": 0},
}

# =========================
# DATA AUGMENTATION
# =========================
def augment_image(img):
    """Apply random augmentations to a single image (numpy array, 0-255 range)."""
    img = img.astype(np.float32) / 255.0

    # Random horizontal flip
    if np.random.rand() > 0.5:
        img = np.fliplr(img)

    # Random brightness adjustment
    factor = np.random.uniform(0.7, 1.3)
    img = np.clip(img * factor, 0, 1)

    # Random rotation (±15 degrees)
    angle = np.random.uniform(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine((img * 255).astype(np.uint8), M, (w, h)) / 255.0

    # Random zoom (crop center 85-100% then resize back)
    zoom = np.random.uniform(0.85, 1.0)
    zh, zw = int(h * zoom), int(w * zoom)
    top  = (h - zh) // 2
    left = (w - zw) // 2
    img  = img[top:top+zh, left:left+zw]
    img  = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    return (img * 255.0).astype(np.float32)

# =========================
# DATASET LOADER
# =========================
def load_dataset(dataset_path):
    """
    Supports two folder structures:
      1. Flat:    dataset/fresh/, dataset/medium/, dataset/rotten/
      2. Per-fruit: dataset/mango/fresh/, dataset/apple/rotten/, etc.
    """
    data   = []
    labels = []
    conditions_found = {"fresh", "medium", "rotten"}

    top_level = os.listdir(dataset_path)

    # Detect structure
    is_flat = any(d in conditions_found for d in top_level)

    if is_flat:
        print("[INFO] Detected FLAT dataset structure (fresh/medium/rotten at root)")
        for condition in conditions_found:
            condition_path = os.path.join(dataset_path, condition)
            if not os.path.isdir(condition_path):
                continue
            days = FRUIT_SHELF_LIFE["default"][condition]
            count = 0
            for img_name in os.listdir(condition_path):
                img_path = os.path.join(condition_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32)
                data.append(preprocess_input(img))
                labels.append(float(days))
                # Augment each image once
                data.append(preprocess_input(augment_image(img)))
                labels.append(float(days))
                count += 1
            print(f"  [{condition}] → {days} days | {count} images loaded (+ {count} augmented)")

    else:
        print("[INFO] Detected PER-FRUIT dataset structure (fruit/condition/)")
        for fruit_name in top_level:
            fruit_path = os.path.join(dataset_path, fruit_name)
            if not os.path.isdir(fruit_path):
                continue
            shelf = FRUIT_SHELF_LIFE.get(fruit_name.lower(), FRUIT_SHELF_LIFE["default"])
            for condition in conditions_found:
                condition_path = os.path.join(fruit_path, condition)
                if not os.path.isdir(condition_path):
                    continue
                days = shelf[condition]
                count = 0
                for img_name in os.listdir(condition_path):
                    img_path = os.path.join(condition_path, img_name)
                    img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32)
                data.append(preprocess_input(img))
                labels.append(float(days))
                # Augment each image once
                data.append(preprocess_input(augment_image(img)))
                labels.append(float(days))
                count += 1
                print(f"  [{fruit_name}/{condition}] → {days} days | {count} images (+ {count} augmented)")

    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32)

# =========================
# BUILD MODEL
# =========================
def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Phase 1: frozen

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)          # regularization
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='relu')(x)  # relu ensures >= 0

    model = tf.keras.Model(inputs, outputs)
    return model, base_model

# =========================
# CALLBACKS
# =========================
def get_callbacks(phase):
    return [
        EarlyStopping(
            monitor='val_mae',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=f"best_model_phase{phase}.h5",
            monitor='val_mae',
            save_best_only=True,
            verbose=1
        )
    ]

# =========================
# MAIN TRAINING
# =========================
def main():
    print("=" * 50)
    print("  FruitiePy — Model Training")
    print("=" * 50)

    # Load data
    print("\n[STEP 1] Loading dataset...")
    data, labels = load_dataset(DATASET_PATH)

    if len(data) == 0:
        print("[ERROR] No images found. Check your dataset path and structure.")
        return

    print(f"\n[INFO] Total samples : {len(data)}")
    print(f"[INFO] Label range   : {labels.min():.1f} – {labels.max():.1f} days")
    print(f"[INFO] Label mean    : {labels.mean():.2f} days")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

    # Build model
    print("\n[STEP 2] Building model...")
    model, base_model = build_model()
    model.summary()

    # -------------------------
    # PHASE 1: Train top layers
    # -------------------------
    print("\n[STEP 3] Phase 1 — Training top layers (base frozen)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='huber',        # more robust to outliers than MSE
        metrics=['mae']
    )
    model.fit(
        X_train, y_train,
        epochs=EPOCHS_PHASE1,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=get_callbacks(1),
        verbose=1
    )

    # -------------------------
    # PHASE 2: Fine-tune top layers of base model
    # -------------------------
    print("\n[STEP 4] Phase 2 — Fine-tuning top layers of MobileNetV2...")
    base_model.trainable = True

    # Only unfreeze the last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # lower LR
        loss='huber',
        metrics=['mae']
    )
    model.fit(
        X_train, y_train,
        epochs=EPOCHS_PHASE2,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=get_callbacks(2),
        verbose=1
    )

    # -------------------------
    # EVALUATE & SAVE
    # -------------------------
    print("\n[STEP 5] Evaluating final model...")
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] Test Loss (Huber): {loss:.4f}")
    print(f"[RESULT] Test MAE         : {mae:.4f} days")

    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()