# Calories Prediction using Deep Learning 🏃‍♂️🔥

This project uses a Deep Learning approach to predict the number of calories burned during exercise. Built with TensorFlow/Keras and scikit-learn, the model features robust data preprocessing, automated feature selection, and comprehensive performance evaluation. 

This repository is highly optimized for running on **WSL (Windows Subsystem for Linux)** with **NVIDIA GPU acceleration** (e.g., RTX 40XX).

## 🌟 Key Features
- **Data Preprocessing:** Handles dummy encoding for categorical variables and removes outliers using the Interquartile Range (IQR) method.
- **Feature Selection:** Implements `SequentialFeatureSelector` alongside an `MLPRegressor` to determine the most impactful features dynamically.
- **Deep Learning Model:** A Multi-Layer Perceptron (MLP) built with the Keras 3 `Sequential` API.
- **Visualizations:** Generates a Correlation Heatmap, Training/Validation Loss & MAE curves, Actual vs. Predicted scatter plots, and a smooth Residual plot.

---

## 🛠️ Prerequisites

Before you begin, ensure you have the following setup:
1. **Windows Host:** Up-to-date NVIDIA GPU drivers installed on Windows.
2. **WSL2 Environment:** Fedora Linux (or similar distributions) running under WSL2.
3. **Python:** Python **3.11** is strictly recommended for the best compatibility with TensorFlow and CUDA.

---

## 🚀 Installation & Setup

### 1. System Preparation (Fedora WSL)
Ensure you have Python 3.11 installed on your system:
```bash
sudo dnf update -y
sudo dnf install python3.11 python3.11-devel gcc git -y