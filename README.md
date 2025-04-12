# 🌸 Flower Classification with MobileNetV2

This project implements a flower classification system using **transfer learning** and **accelerated learning** techniques based on the pre-trained **MobileNetV2** model. The dataset includes five flower types: *daisy*, *dandelion*, *roses*, *sunflowers*, and *tulips*.

## 👥 Team Members

| Student Number | First Name | Last Name |
|----------------|------------|-----------|
| 11427591       | Mohan      | Hao       |

---

## 📁 Dataset

- Dataset contains five folders (`daisy/`, `dandelion/`, `roses/`, `sunflowers/`, `tulips/`), each with labeled `.jpg` images.
- All images are resized to 224x224 and normalized to the range `[-1, 1]`.
- Labels are one-hot encoded for training.

---

## 🧠 Methods

### 🔁 Transfer Learning
- Loads MobileNetV2 with ImageNet weights and freezes base layers.
- Adds trainable dense layers for classification.
- Trains only the top layers on the flower dataset.

### ⚡ Accelerated Learning
- Extracts features from the base model first.
- Trains a new shallow classifier (Dense + Softmax) on these fixed features.

---

## 📊 Evaluation

Includes the following evaluation methods:
- Confusion Matrices (Training and Testing)
- Classification Report (Precision, Recall, F1-score, Support)
- K-Fold Cross Validation (default 5-fold)
- Accuracy & Macro F1 metrics
- Class Distribution Visualization
- Training History Plots

---

## 📌 How to Use

pre-trained model: MobileNetV2
trained model: .h5 file
testing by using .h5 file: testing.ipynb or testing/py
training file: TransferingLearning.ipynb
