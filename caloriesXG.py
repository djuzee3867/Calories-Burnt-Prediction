import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor

# Load and Clean Data
dataset = pd.read_csv('calories.csv') 
dataset = dataset.drop(columns=['User_ID'])

dataset['Gender'] = dataset['Gender'].replace({'male': 0, 'female': 1}).astype(int)
dataset = dataset.drop(columns=['Height'])
dataset = dataset.drop(columns=['Body_Temp'])

print(dataset.describe())

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(6, 4))
sns.histplot(dataset['Gender'], bins=2, kde=False, shrink=0.8)
plt.title('Distribution of Gender')
plt.xlabel('Gender (0 = Male, 1 = Female)')
plt.ylabel('Count')
plt.xticks([0, 1], ['Male', 'Female'])
plt.grid(axis='y', alpha=0.75)
plt.show()

correlation_matrix = dataset.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# Preprocessing
x = dataset.drop(columns=['Calories'])
y = dataset['Calories']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Initialize and Train XGBoost 
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.7,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=10,
    random_state=42,
    eval_metric='mae',
    device='cuda' # 
)

model.fit(
    x_train_scaled, y_train,
    eval_set=[(x_train_scaled, y_train), (x_test_scaled, y_test)],
    verbose=False
)
print("เทรนเสร็จสมบูรณ์!")

# Evaluate Model
y_pred = model.predict(x_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse_value = mean_squared_error(y_test, y_pred)

print('\n--- XGBoost Regression Results ---')
print(f"MSE: {mse_value:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² : {r2:.4f}")

# Visualizations
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--', lw=2
)
plt.title('Actual vs Predicted Calories (XGBoost)')
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.show()

# Plotting Training and Validation MAE
results = model.evals_result()
epochs = len(results['validation_0']['mae'])
x_axis = range(0, epochs)

plt.figure(figsize=(12, 6))
plt.plot(x_axis, results['validation_0']['mae'], label='Training MAE')
plt.plot(x_axis, results['validation_1']['mae'], label='Validation MAE')
plt.legend()
plt.ylabel('MAE')
plt.xlabel('Number of Boosting Rounds')
plt.title('XGBoost Training and Validation MAE')
plt.show()

# Overfit Check
print("\n--- Overfit Check ---")
y_train_pred = model.predict(x_train_scaled)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_pred)

print(f"Train MAE: {train_mae:.4f}")
print(f"Test MAE : {test_mae:.4f}")
gap = test_mae - train_mae
print(f"(Test - Train) Gap: {gap:.4f}")

# Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.title('Residuals vs Fitted (XGBoost)')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.show()