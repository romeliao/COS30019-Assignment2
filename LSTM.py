import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def train_lstm_model(X_train_path, y_train_path, X_test_path, y_test_path, output_dir):
    """
    Train an LSTM model for traffic condition prediction.

    Args:
        X_train_path (str): Path to the training features CSV file.
        y_train_path (str): Path to the training labels CSV file.
        X_test_path (str): Path to the testing features CSV file.
        y_test_path (str): Path to the testing labels CSV file.
        output_dir (str): Directory to save the processed datasets.

    Returns:
        model: Trained LSTM model.
    """
    # Load the training and testing data
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    # Handle NaN and infinite values in X_train and X_test
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    # Filter numeric columns only
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    # Align columns in X_test with X_train
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Scale the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape the data for LSTM (samples, timesteps, features)
    X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(y_train.shape[1]))  # Adjusted to match the number of target variables

    # Compile the model
    optimizer = Adam(clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model
    print("[INFO] Training the LSTM model...")
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=1)

    # Generate predictions
    predictions = model.predict(X_test_scaled)

    # Check for NaN or infinite values in predictions
    if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
        raise ValueError("Predictions contain NaN or infinite values.")

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print(f"[INFO] Mean Squared Error on Test Data: {mse}")

    # Save the model in the 'models' folder at the root level
    models_dir = os.path.join(os.path.dirname(output_dir), "models")
    os.makedirs(models_dir, exist_ok=True)  # Create the directory if it doesn't exist
    model_path = os.path.join(models_dir, "lstm_traffic_model.h5")
    model.save(model_path)
    print(f"[INFO] LSTM model saved to {model_path}")

    return model