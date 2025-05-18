import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    df.sort_values('timestamp', inplace=True)
    return df

def prepare_features(df, n_lags=5):
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = df['glucose_level'].shift(i)
    df.dropna(inplace=True)
    return df

def scale_data(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
