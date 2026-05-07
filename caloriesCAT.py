import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from catboost import CatBoostRegressor

dataset = pd.read_csv('calories.csv')
dataset = dataset.drop(columns=['User_ID', 'Height', 'Body_Temp'])
dataset['Gender'] = dataset['Gender'].replace({'male': 0, 'female': 1}).astype(int)

print(dataset.describe())

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

x = dataset.drop(columns=['Calories'])
y = dataset['Calories']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = CatBoostRegressor(
    iterations=400,
    learning_rate=0.05,
    depth=5,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=0,
    early_stopping_rounds=50,
    eval_metric='MAE',
    task_type='GPU'
)

model.fit(x_train_scaled, y_train, eval_set=(x_test_scaled, y_test))

y_pred = model.predict(x_test_scaled)

ev_results = model.get_evals_result()

train_mae = ev_results['learn']['MAE']
val_mae = ev_results['validation']['MAE']
train_rmse = ev_results['learn']['RMSE']
val_rmse = ev_results['validation']['RMSE']

period = model.get_params().get('metric_period', 1)

x_train_mae  = [i * period for i in range(len(train_mae))]
x_val_mae    = [i * period for i in range(len(val_mae))]
x_train_rmse = [i * period for i in range(len(train_rmse))]
x_val_rmse   = [i * period for i in range(len(val_rmse))]

plt.figure(figsize=(12, 6))
plt.plot(x_train_mae, train_mae, label='Training MAE', color='blue')
plt.plot(x_val_mae, val_mae, label='Validation MAE', color='orange')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.title('Training and Validation MAE')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(x_train_rmse, train_rmse, label='Training Loss (RMSE)', color='blue')
plt.plot(x_val_rmse, val_rmse, label='Validation Loss (RMSE)', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss (RMSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse_value = mean_squared_error(y_test, y_pred)

print('\n--- CatBoost Regression Results ---')
print(f"MSE: {mse_value:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² : {r2:.4f}")

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--', lw=2
)
plt.title('Actual vs Predicted Calories (CatBoost)')
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted (CatBoost)')
plt.show()