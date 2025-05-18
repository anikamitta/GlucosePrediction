import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from scipy.signal import savgol_filter

# Load and Process the Data
df = pd.read_csv("glucose_data.csv",parse_dates=["timestamp"])
glucose_values = df["glucose_level"].values.reshape(-1,1)

# Normalize the Data
scaler = MinMaxScaler()
glucose_scaled = scaler.fit_transform(glucose_values)
df["glucose_smoothed"] = savgol_filter(df["glucose_level"], window_length=11, polyorder=2)

# Create Sequences
def create_sequences(data, seq_length):
    X, y = [ ], [ ]
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Generate Sequences and Split Data
sequence_length = 120
X, y = create_sequences(glucose_scaled, sequence_length)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM Model
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer = 'adam', loss = 'mse')

# Train the Model
model.fit(X_train, y_train, epochs = 20, validation_data = (X_test, y_test), verbose=1)

# Make Predictions and Inverse Transform
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_actual = scaler.inverse_transform(y_test)

# Plot Results
plt.figure(figsize=(10, 5))
plt.plot(y_test_actual, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title("LSTM Glucose Prediction")
plt.xlabel("Time Step")
plt.ylabel("Glucose Level (mg/dL)")
plt.legend()
plt.tight_layout()
plt.show()

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
print(f"RMSE: {rmse:.2f} mg/dL")

r2 = r2_score(y_test_actual, y_pred)
print(f"R² Score: {r2:.2f}")