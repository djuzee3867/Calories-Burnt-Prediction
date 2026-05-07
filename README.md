# This project is part of a research project MJU CSTI2026

# Calories Prediction using Deep Learning

A Deep Learning model built with TensorFlow and Keras to predict calories burned during physical activities. This project is optimized for execution on **Windows Subsystem for Linux (WSL)** with **NVIDIA GPU acceleration**.

## Features
- **Data Pipeline:** Categorical encoding and outlier removal using the Interquartile Range (IQR).
- **Feature Selection:** Automated selection utilizing `SequentialFeatureSelector` and `MLPRegressor`.
- **Model Architecture:** Multi-Layer Perceptron (MLP) via Keras 3.
- **Evaluation:** Comprehensive metrics (MSE, RMSE, MAE, R²) and visualization (Actual vs. Predicted, Residual plots).

## Prerequisites
- **OS:** WSL2 Fedora/Ubuntu Or Fedora Linux.
- **Hardware:** NVIDIA GPU with up-to-date Windows host drivers.
- **Software:** Python 3.11.

## Setup & Installation
**Download** the calories.csv dataset from [Kaggle](https://www.kaggle.com/datasets/ruchikakumbhar/calories-burnt-prediction) and place it in the project's root directory.

**Clone the repository and setup the environment:**
```bash
git clone https://github.com/djuzee3867/Calories-Burnt-Prediction
cd Calories-Burnt-Prediction

python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
## Usage
- **Run** the main pipeline to train the model and generate evaluations
```
python main.py
```


- **View on [djuzeeKaggle](https://www.kaggle.com/code/djuzee/calories-burnt-prediction-mlp-regression)**

## License
All rights reserved
