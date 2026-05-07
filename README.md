# Calories Prediction using Deep Learning (TensorFlow/Keras)

This project uses a Deep Learning approach to predict calories burned during exercise. The model is built using TensorFlow/Keras and scikit-learn, featuring exploratory data analysis (EDA), outlier removal using IQR, feature selection, and comprehensive model evaluation.

## Environment Setup

This project is optimized to run on **WSL (Windows Subsystem for Linux)** using **Fedora 43** and utilizes an NVIDIA GPU (e.g., RTX 4050) for accelerated training.

### Prerequisites
1. **Windows Host:** Ensure your NVIDIA GPU drivers on Windows are up to date. You **do not** need to install NVIDIA drivers inside WSL.
2. **WSL2:** Make sure you are running Fedora 43 under WSL2.
3. **Python 3.10+** installed on your Fedora WSL environment.

### Installation Steps

1. **Update Fedora packages:**
   ```bash
   sudo dnf update -y
   sudo dnf install python3 python3-pip python3-devel gcc git -y