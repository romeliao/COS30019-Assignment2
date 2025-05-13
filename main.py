import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datapreprocessing import preprocess_data
from LSTM import train_lstm_model
from random_forest import train_random_forest_model
from xgboost_model import train_xgboost_model

if __name__ == "__main__":
    # Define the input spreadsheet path, traffic data path, and output directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    spreadsheet_path = os.path.join(base_dir, "Scats Data October 2006.xls")
    traffic_data_path = os.path.join(base_dir, "Traffic_Count_Locations_with_LONG_LAT.csv")
    output_dir = os.path.join(base_dir, "processed_dataset")
    models_predictions = os.path.join(base_dir, "models_predictions")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Run the preprocessing function
    print("[INFO] Starting data preprocessing...")
    try:
        preprocess_data(spreadsheet_path, output_dir, traffic_data_path)
    except Exception as e:
        print(f"[ERROR] Data preprocessing failed: {e}")
        exit(1)

    # Define paths to processed datasets
    X_train_path = os.path.join(output_dir, "X_train.csv")
    y_train_path = os.path.join(output_dir, "y_train.csv")
    X_test_path = os.path.join(output_dir, "X_test.csv")
    y_test_path = os.path.join(output_dir, "y_test.csv")

    # Check that required files exist
    for path in [X_train_path, y_train_path, X_test_path, y_test_path]:
        if not os.path.exists(path):
            print(f"[ERROR] Missing required dataset: {path}")
            exit(1)

    # Train the LSTM model and show sample predictions
    print("[INFO] Training the LSTM model...")
    try:
        lstm_model = train_lstm_model(X_train_path, y_train_path, X_test_path, y_test_path, output_dir)
    except Exception as e:
        print(f"[ERROR] LSTM model training failed: {e}")
    else:
        # Load test data for prediction preview
        print("[INFO] Loading test data for prediction preview...")
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path)

        # Generate predictions using the trained LSTM model
        print("[INFO] Generating predictions with the LSTM model...")
        try:
            # Ensure X_test contains only numeric data
            X_test = X_test.select_dtypes(include=[np.number])

            # Convert X_test to a NumPy array and reshape it for LSTM input
            X_test_scaled = X_test.values.astype(np.float32)  # Ensure the data type is float32
            X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

            # Generate predictions
            predictions = lstm_model.predict(X_test_reshaped)

            # Display sample predictions
            print("\n[INFO] LSTM Sample Predictions vs Actual:")
            for i in range(min(5, len(predictions))):
                actual_value = y_test.iloc[i].values if len(y_test.columns) > 1 else [y_test.values[i]]
                predicted_value = float(predictions[i][0]) if len(predictions[i]) == 1 else predictions[i][0]
                print(f"Predicted: {predicted_value:.4f} | Actual: {actual_value[0]}")
        except Exception as e:
            print(f"[ERROR] Failed to generate predictions: {e}")




    # Train the XGBoost model and save predictions
    print("[INFO] Training the XGBoost model...")
    try:
        xgb_model, xgb_predictions = train_xgboost_model(X_train_path, y_train_path, X_test_path, y_test_path, output_dir,models_predictions)

        # Save predictions to a CSV file
        if len(xgb_predictions.shape) > 1 and xgb_predictions.shape[1] > 1:
            # Multi-dimensional predictions (e.g., multiple target variables)
            xgb_predictions_df = pd.DataFrame(xgb_predictions, columns=[f"Prediction_{i}" for i in range(xgb_predictions.shape[1])])
        else:
            # Single-dimensional predictions
            xgb_predictions_df = pd.DataFrame(xgb_predictions, columns=["Prediction"])

        xgb_predictions_df.to_csv(os.path.join(models_predictions, "xgboost_predictions.csv"), index=False)
        print(f"[INFO] XGBoost predictions saved to {os.path.join(output_dir, 'xgboost_predictions.csv')}")

        # Optionally, display a few rows of the predictions
        print("\n[INFO] Sample XGBoost Predictions:")
        print(xgb_predictions_df.head())

    except Exception as e:
        print(f"[ERROR] XGBoost model training failed: {e}")
