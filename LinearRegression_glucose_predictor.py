import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from utils import load_data, prepare_features, scale_data

# Load and prepare data
df = load_data('glucose_data.csv')
df = prepare_features(df, n_lags=5)

# Features and target
X = df[[f'lag_{i}' for i in range(1, 6)]]
y = df['glucose_level']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale features
X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluate
rmse = mean_squared_error(y_test, y_pred)
print(f'RMSE: {rmse:.2f}')

# Plot predictions
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Glucose Level Prediction')
plt.xlabel('Time Step')
plt.ylabel('Glucose (mg/dL)')
plt.legend()
plt.tight_layout()
plt.show()

mse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f} mg/dL")
print(f"R² Score: {r2:.2f}")
