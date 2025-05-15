import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# LOAD & FEATURE ENGINEER DATA
df = pd.read_csv("glucose_data.csv", parse_dates=["timestamp"])
df["glucose_smoothed"] = savgol_filter(df["glucose_level"], window_length=11, polyorder=2)

# Time-based features
df["hour"] = df["timestamp"].dt.hour
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# Lag features (similar to LSTM sequence input)
sequence_length = 120
for i in range(1, sequence_length + 1):
    df[f"lag_{i}"] = df["glucose_smoothed"].shift(i)

df.dropna(inplace=True)

# SPLIT INTO FEATURES AND TARGET
features = [col for col in df.columns if col.startswith("lag_") or "hour_" in col]
X = df[features]
y = df["glucose_smoothed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# TRAIN XGBOOST MODEL
model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# PREDICT & EVALUATE
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f} mg/dL")
print(f"R² Score: {r2:.2f}")

# PLOT RESULTS
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title("XGBoost Glucose Prediction")
plt.xlabel("Time Step")
plt.ylabel("Glucose Level (mg/dL)")
plt.legend()
plt.tight_layout()
plt.show()

