import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE = 224

data = []
labels = []

label_map = {
    "fresh": 7,
    "medium": 3,
    "rotten": 0
}

dataset_path = "../dataset"

for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)

    if category not in label_map:
        continue

    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (224,224))
        img = img / 255.0

        data.append(img)
        labels.append(label_map[category])

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

model.save("fruit_model.h5")

print("Model saved!")