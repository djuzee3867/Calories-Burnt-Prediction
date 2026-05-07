import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sns
import pandas as pd

from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Load and Preprocess Data
dataset = pd.read_csv('calories.csv') # Changed path from Colab to local
dataset = dataset.drop(columns=['User_ID'])
dataset = pd.get_dummies(dataset, columns=['Gender'], drop_first=True)

# Remove Outliers using IQR
Q1 = dataset['Calories'].quantile(0.25)
Q3 = dataset['Calories'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
dataset = dataset[(dataset['Calories'] >= lower_bound) & (dataset['Calories'] <= upper_bound)]

print(dataset.head())
print(dataset.describe())

# Correlation Matrix
correlation_matrix = dataset.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# Prepare X and Y
x = dataset.drop(columns=['Calories'])
y = dataset['Calories']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 2. Feature Selection using MLPRegressor
mlp_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

sfs = SequentialFeatureSelector(estimator=mlp_model,
                                n_features_to_select='auto',
                                tol=0.001,
                                direction='forward',
                                cv=3,
                                n_jobs=-1)

print("เริ่มคัดตัวแปร (Feature Selection)...")
sfs.fit(x_train_scaled, y_train)
selected_features_mlp = x.columns[sfs.get_support()]

print("Selected Features from MLP Sequential Selection:")
print(selected_features_mlp.tolist())

# Note: The neural network below currently trains on ALL features (x_train_scaled).
# If you want to train ONLY on selected features, use x_train_mlp_selected instead.
x_train_mlp_selected = pd.DataFrame(x_train_scaled, columns=x.columns)[selected_features_mlp]

# 3. Build and Train Deep Learning Model
print("กำลังสร้างและเทรนโมเดล...")
model = Sequential([
    Dense(128, activation='relu', input_shape=(x_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss='mse',
    metrics=['mae']
)

history = model.fit(
    x_train_scaled, y_train,
    validation_split=0.2, # แบ่ง 20% ของ Train มาเช็คผลระหว่างเทรน
    epochs=100,
    batch_size=32,
    verbose=1
)
print("เทรนเสร็จสมบูรณ์!")

# 4. Evaluate Model
y_pred = model.predict(x_test_scaled).flatten()

mae = mean_absolute_error(y_test, y_pred)
mse_value = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse_value)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse_value:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# 5. Visualizations
def plot_history(hist):
    mae_history = hist.history['mae']
    val_mae_history = hist.history['val_mae']
    loss_history = hist.history['loss']
    val_loss_history = hist.history['val_loss']
    epochs_range = range(len(mae_history))

    plt.figure(figsize=(12, 6))

    # กราฟ MAE
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, mae_history, label='Training MAE')
    plt.plot(epochs_range, val_mae_history, label='Validation MAE')
    plt.legend(loc='upper right')
    plt.title('Training and Validation MAE')

    # กราฟ Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss_history, label='Training Loss')
    plt.plot(epochs_range, val_loss_history, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()

plot_history(history)

print("\n--- Overfit Check ---")
train_loss, train_mae = model.evaluate(x_train_scaled, y_train, verbose=0)
test_loss, test_mae = model.evaluate(x_test_scaled, y_test, verbose=0)

print(f"Train MAE: {train_mae:.4f}")
print(f"Test MAE : {test_mae:.4f}")
gap = test_mae - train_mae
print(f"(Test - Train) Gap: {gap:.4f}")

# Scatter Plot True vs Predict
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Calories')
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.show()

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.show()