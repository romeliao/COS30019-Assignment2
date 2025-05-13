import os
import numpy as np  
import pandas as pd 
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt  # Add this import for plotting

def train_xgboost_model(X_train_path, y_train_path, X_test_path, y_test_path, output_dir, models_predictions):
    """
    Train an XGBoost model for traffic condition prediction.

    Args:
        X_train_path (str): Path to the training features CSV file.
        y_train_path (str): Path to the training labels CSV file.
        X_test_path (str): Path to the testing features CSV file.
        y_test_path (str): Path to the testing labels CSV file.
        output_dir (str): Directory to save the trained model.

    Returns:
        model: Trained XGBoost model.
        predictions: Predictions made on X_test.
    """
    # Load the training and testing data
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()  # Ensure it's a Series
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()

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

    # Initialize XGBoost Regressor with tuned hyperparameters
    eval_metric = "mae"  # Set your desired evaluation metric here
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric=eval_metric,  # Dynamically set the evaluation metric
        early_stopping_rounds=10
    )

    # Train the model with evaluation set
    print("[INFO] Training XGBoost model...")
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )

    # Extract training and validation metrics
    results = model.evals_result()

    # Dynamically determine the metric name for plotting
    metric_name = eval_metric.upper()  # Convert to uppercase for better readability

    # Plot training and validation metrics
    print(f"[INFO] Plotting training and validation {metric_name}...")
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(results['validation_0'][eval_metric]) + 1)
    plt.plot(epochs, results['validation_0'][eval_metric], label=f"Train {metric_name}")
    plt.plot(epochs, results['validation_1'][eval_metric], label=f"Validation {metric_name}")
    plt.title(f"Training and Validation {metric_name}")
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid()
    plt.show()

    # Generate predictions
    predictions = model.predict(X_test)

    # Check for NaN or infinite values in predictions
    if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
        raise ValueError("Predictions contain NaN or infinite values.")

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print(f"[INFO] Mean Squared Error on Test Data: {mse}")

    # Plot XGBoost predictions vs actual values
    print("[INFO] Plotting XGBoost predictions vs actual values...")
    plt.figure(figsize=(10, 6))
    if len(y_test.shape) == 1:  # Handle single target variable
        plt.plot(y_test.values[:100], label="Actual", linestyle='dashed')  # First 100 samples
        plt.plot(predictions[:100], label="Predicted")
    else:  # Handle multiple target variables
        for i in range(min(5, y_test.shape[1])):  # Plot up to 5 target variables
            plt.plot(y_test.iloc[:100, i].values, label=f"Actual_{i}", linestyle='dashed')  # First 100 samples
            plt.plot(predictions[:100, i], label=f"Predicted_{i}")
    plt.title("XGBoost Predictions vs Actual Values")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()

    # Save the models
    models_dir = os.path.join(os.path.dirname(output_dir), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "xgboost_traffic_model.pkl")
    joblib.dump(model, model_path)
    print(f"[INFO] XGBoost model saved to {model_path}")

    return model, predictions