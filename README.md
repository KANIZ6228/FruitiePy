# 🍎 FruitiePy - AI-Powered Fruit Shelf Life Prediction System

FruitiePy is a Computational Intelligence project that uses **Artificial Intelligence, Deep Learning, and Computer Vision** to predict the remaining shelf life of fruits from images. The system analyzes uploaded fruit images, estimates the number of days the fruit remains edible, and classifies its condition as **Fresh**, **Medium**, or **Rotten**.

The project was developed using **TensorFlow, MobileNetV2, OpenCV, Flask, HTML, CSS, and JavaScript**.

---

## 📌 Project Overview

Food spoilage and waste are major global concerns. Determining fruit freshness manually can be subjective and inaccurate. FruitiePy addresses this problem by providing an intelligent image-based system capable of assessing fruit condition automatically.

The system performs:

* Fruit detection using MobileNetV2
* Shelf life prediction using a CNN-based regression model
* Freshness classification
* Web-based visualization dashboard

---
### 🎥 Demo Video



**[▶️ Watch Demo](https://youtu.be/5MN8IquZOng)**
## 🚀 Features

✅ Upload fruit images through a web interface

✅ Detect whether the uploaded image contains a fruit

✅ Predict remaining shelf life in days

✅ Classify fruit condition:

* Fresh 😀
* Medium 😥
* Rotten ❌

✅ Display confidence score

✅ Reject non-fruit images automatically

✅ Responsive and user-friendly dashboard

---

## 🧠 Technologies Used

| Technology  | Purpose                    |
| ----------- | -------------------------- |
| Python      | Main programming language  |
| TensorFlow  | Deep learning framework    |
| Keras       | Neural network development |
| MobileNetV2 | Fruit detection            |
| OpenCV      | Image processing           |
| Flask       | Backend web framework      |
| HTML/CSS    | Frontend design            |
| JavaScript  | User interaction           |
| NumPy       | Numerical computations     |

---

## 🔬 Computational Intelligence Concepts

This project incorporates the following Computational Intelligence techniques:

* Artificial Intelligence (AI)
* Deep Learning
* Artificial Neural Networks (ANN)
* Convolutional Neural Networks (CNN)
* Transfer Learning
* Computer Vision
* Pattern Recognition
* Regression Modeling
* Classification

---

## ⚙️ System Workflow

```text
User Uploads Image
        ↓
Fruit Detection (MobileNetV2)
        ↓
If Not Fruit
        ↓
Display "Not a Fruit"
        ↓
If Fruit
        ↓
Image Preprocessing
        ↓
CNN Regression Model
        ↓
Shelf Life Prediction
        ↓
Freshness Classification
        ↓
Display Results
```

---

## 🏗️ Project Structure

```text
FruitiePy/
│
├── app/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│
├── model/
│   └── fruit_model.h5
│
├── dataset/
│   ├── fruit_name/
│   │   ├── fresh/
│   │   ├── medium/
│   │   └── rotten/
│
├── training/
│   └── train_model.py
│
├── requirements.txt
│
└── README.md
```

---

## 📊 Shelf Life Classification

The trained model predicts the remaining shelf life in days.

The prediction is converted into freshness categories:

| Predicted Shelf Life | Condition |
| -------------------- | --------- |
| ≥ 5 Days             | Fresh 😀  |
| 2–4 Days             | Medium 😥 |
| < 2 Days             | Rotten ❌  |

Example:


Prediction: 6.8 Days
Result: Fresh 😀



Prediction: 3.2 Days
Result: Medium 😥



Prediction: 0.9 Days
Result: Rotten ❌



## 📂 Dataset Structure

The dataset is organized by fruit type and freshness condition.


dataset/
│
├── banana/
│   ├── fresh/
│   ├── medium/
│   └── rotten/
│
├── apple/
│   ├── fresh/
│   ├── medium/
│   └── rotten/
│
├── mango/
│   ├── fresh/
│   ├── medium/
│   └── rotten/
```

Each image is labeled according to its freshness condition and associated shelf-life value.



## 🏋️ Model Training

The model is trained using:

* MobileNetV2 Transfer Learning
* Data Augmentation
* Fine-Tuning
* Huber Loss
* Adam Optimizer
* Early Stopping
* Learning Rate Reduction

Data augmentation techniques include:

* Rotation
* Zooming
* Brightness Adjustment
* Horizontal Flipping

These techniques improve model generalization and reduce overfitting.

---

## 💻 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/FruitiePy.git
cd FruitiePy
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Navigate to the application folder and run:

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## 📸 Application Demo

### Upload Fruit Image

Users upload an image through the web dashboard.

### AI Analysis

The system:

1. Detects if the image contains a fruit.
2. Processes the image.
3. Predicts shelf life.
4. Classifies fruit condition.

### Result Dashboard

Displays:

* Shelf Life Prediction
* Freshness Condition
* Confidence Score
* Uploaded Image Preview

---

## 🎯 Applications

FruitiePy can be applied in:

* Smart Agriculture
* Food Quality Monitoring
* Supermarkets
* Food Supply Chains
* Household Food Management
* Food Waste Reduction Programs

---

## 🔮 Future Improvements

* Real-time camera detection
* Mobile application development
* Fruit disease detection
* Cloud deployment
* IoT integration
* Explainable AI visualizations
* Support for additional fruit categories

---

## 📈 Results

The system successfully demonstrates:

* Fruit recognition using transfer learning
* Shelf life prediction using deep learning
* Automated freshness assessment
* Interactive AI-powered dashboard

The project showcases the practical application of Computational Intelligence techniques for solving real-world food quality assessment problems.

---

## 📚 References

1. TensorFlow Documentation
2. OpenCV Documentation
3. Keras Documentation
4. MobileNetV2 Research Paper
5. Deep Learning by Goodfellow, Bengio & Courville
6. Computational Intelligence Course Materials

---

## 👩‍💻 Author

**Kaniz Fatema**

Bachelor of Software Engineering

Universiti Teknologi Malaysia (UTM)

Computational Intelligence Project

---

## ⭐ Acknowledgement

This project was developed as part of the Computational Intelligence course to demonstrate the application of Artificial Intelligence, Deep Learning, and Computer Vision in intelligent food quality assessment systems.
