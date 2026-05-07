# 🍎 FruitiePy – AI Fruit Shelf Life Predictor

##  Overview

FruitiePy is an AI-powered computer vision system that predicts the **remaining shelf life of fruits** from images using deep learning.

The system:
- Detects whether the input image contains a fruit
- Predicts how many days the fruit remains edible
- Classifies the fruit condition as Fresh, Medium, or Rotten
- Displays results in a modern AI dashboard web interface

Built using **Flask + TensorFlow + OpenCV + MobileNetV2 + CNN Regression Model**.

---

## 🚀 Features

- 📷 Upload fruit images for prediction
- 🧠 CNN-based regression model (TensorFlow/Keras)
- 🍌 Fruit detection using MobileNetV2 (ImageNet pretrained)
- 🕒 Shelf life prediction (in days)
- 🍃 Condition classification:
  - Fresh 😀
  - Medium 😥
  - Rotten ❌
- 📊 Confidence score display
- 🚫 Automatic non-fruit rejection system
- 🎨 Modern responsive dashboard UI

## 🏗️ System Workflow


User Upload Image
        ↓
Fruit Detection (MobileNetV2)
        ↓
If NOT fruit → Show "Not a Fruit"
        ↓
If fruit → Preprocess image
        ↓
CNN Regression Model
        ↓
Predict Shelf Life (Days)
        ↓
Classify Condition
        ↓
Display Result in Dashboard UI


--How It Works:

User uploads a fruit image
System checks if image is a fruit using MobileNetV2
If valid → image is preprocessed
CNN regression model predicts shelf life in days
Output is classified into:
≥ 5 days → Fresh
2–4 days → Medium
< 2 days → Rotten
Result is displayed in AI dashboard UI

--Fruit Detection Logic

The system uses MobileNetV2 pretrained on ImageNet to verify input images.

If fruit detected → proceed to prediction
If not fruit → show: ❌ Not a Fruit
