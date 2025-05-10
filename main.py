import os
import pandas as pd
from datapreprocessing import preprocess_data
from LSTM import train_lstm_model

if __name__ == "__main__":
    # Define the input spreadsheet path, traffic data path, and output directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    spreadsheet_path = os.path.join(base_dir, "Scats Data October 2006.xls")
    traffic_data_path = os.path.join(base_dir, "Traffic_Count_Locations_with_LONG_LAT.csv")
    output_dir = os.path.join(base_dir, "processed_dataset")

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

    # Train the LSTM model and get predictions
    print("[INFO] Training the LSTM model...")
    try:
        model = train_lstm_model(X_train_path, y_train_path, X_test_path, y_test_path, output_dir)
    except Exception as e:
        print(f"[ERROR] LSTM model training failed: {e}")
        exit(1)

    # Load test labels and features to display predictions
    print("[INFO] Loading test data for prediction preview...")
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    # Filter y_test to keep only columns V00 to V95 if they exist
    v_columns = [f"V{str(i).zfill(2)}" for i in range(96)]
    available_v_columns = [col for col in v_columns if col in y_test.columns]
    if available_v_columns:
        y_test = y_test[available_v_columns]

    # Predict and show sample output
    from tensorflow.keras.models import load_model
    import numpy as np

    models_dir = os.path.join(base_dir, "models")
    model_path = os.path.join(models_dir, "lstm_traffic_model.h5")

    if os.path.exists(model_path):
        model = load_model(model_path)

        from sklearn.preprocessing import MinMaxScaler
        import joblib

        scaler_path = os.path.join(models_dir, "lstm_scaler.save")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            scaler = MinMaxScaler()
            X_test = X_test.select_dtypes(include='number')
            scaler.fit(X_test)

        X_test_scaled = scaler.transform(X_test.select_dtypes(include='number'))
        X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

        predictions = model.predict(X_test_scaled)

        print("\n[INFO] Sample Predictions vs Actual:")
        for i in range(min(5, len(predictions))):
            actual_value = y_test.iloc[i].values if len(y_test.columns) > 1 else [y_test.values[i]]
            print(f"Predicted: {predictions[i][0]:.4f} | Actual: {actual_value[0]}")
    else:
        print("[ERROR] Trained model not found. Skipping prediction preview.")
