import os
import pandas as pd
from datapreprocessing import preprocess_data
from LSTM import train_lstm_model, generate_lstm_predictions
from xgboost_model import train_xgboost_model
from gru import GRUModel

if __name__ == "__main__":
    # Define the input spreadsheet path, traffic data path, and output directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    spreadsheet_path = os.path.join(base_dir, "Scats Data October 2006.xls")
    traffic_data_path = os.path.join(base_dir, "Traffic_Count_Locations_with_LONG_LAT.csv")
    output_dir = os.path.join(base_dir, "processed_dataset")
    models_predictions = os.path.join(base_dir, "models_predictions")
    models_dir = os.path.join(base_dir, "models")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Run the preprocessing function
    print("[INFO] Starting data preprocessing...")
    try:
        preprocess_data(spreadsheet_path, output_dir, traffic_data_path)
    except Exception as e:
        print(f"[ERROR] Data preprocessing failed: {e}")
        exit(1)

    # Define paths to processed datasets (use only the original, unscaled data)
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
        # Generate predictions using the trained LSTM model
        try:
            predictions = generate_lstm_predictions(lstm_model, X_test_path, y_test_path)
        except Exception as e:
            print(f"[ERROR] Failed to generate predictions: {e}")

    # Train the XGBoost model and save predictions
    print("[INFO] Training the XGBoost model...")
    try:
        xgb_model, xgb_predictions = train_xgboost_model(X_train_path, y_train_path, X_test_path, y_test_path, output_dir, models_predictions)

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

    # Train the GRU model and save predictions
    print("[INFO] Training the GRU model...")
    try:
        model = GRUModel()
        X_train, y_train, X_test, y_test, scats_test = model.load_and_prepare_data(output_dir)
        
        model.train(X_train, y_train, epochs = 10, batch_size = 64)
        
        mse, mae = model.evaluate(X_test, y_test)
        
        print(f"MSE: {mse:.2f}, MAE: {mae:.2f}")
        
        model.save_model(os.path.join(base_dir, "traffic_gru_model.keras"))
        
        # intended route, change this and the best routes value if looking for a different route)
        scats_sites = ["2000", "3122"]
        
        predicted_flows = model.predict_flows_for_scats(scats_sites, X_test, scats_test)
        scats_data = model.load_scats_coordinates(os.path.join(output_dir, "X_test.csv"))
        
        # Convert keys to strings
        scats_data = {str(k): v for k, v in scats_data.items()}
        predicted_flows = {str(k): v for k, v in predicted_flows.items()}

        # Build graph and find routes
        graph = model.build_scats_graph(scats_data, predicted_flows)
        best_routes = model.find_optimal_routes(graph, "2000", "3122", k=3)

    except Exception as e:
        print(f"Error: {e}")
