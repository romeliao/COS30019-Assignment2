import os
import numpy as np  
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def train_random_forest_model(X_train_path, y_train_path, X_test_path, y_test_path, output_dir):
    """
    Train a Random Forest model for traffic condition prediction.

    Args:
        X_train_path (str): Path to the training features CSV file.
        y_train_path (str): Path to the training labels CSV file.
        X_test_path (str): Path to the testing features CSV file.
        y_test_path (str): Path to the testing labels CSV file.
        output_dir (str): Directory to save the trained model.

    Returns:
        model: Trained Random Forest model.
        predictions: Predictions made on X_test.
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

    # Initialize Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    print("[INFO] Training Random Forest model...")
    model.fit(X_train, y_train)

    # Generate predictions
    predictions = model.predict(X_test)

    # Check for NaN or infinite values in predictions
    if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
        raise ValueError("Predictions contain NaN or infinite values.")

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print(f"[INFO] Mean Squared Error on Test Data: {mse}")

    # Save the model in the 'models' folder at the root level
    models_dir = os.path.join(os.path.dirname(output_dir), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "random_forest_traffic_model.pkl")
    joblib.dump(model, model_path)
    print(f"[INFO] Random Forest model saved to {model_path}")

    return model, predictions
