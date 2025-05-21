import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from datapreprocessing import preprocess_data
from LSTM import LSTMModel
from xgboost_model import XGBoostModel
from gru import GRUModel
from search_algorithms.AStar import a_star_search, nx_to_edge, get_coords

if __name__ == "__main__":

    # Optimal travel route and time
    # Origin and destination SCATS sites
    Origin = "2000"
    Destination = "4035"

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

    # Train the LSTM model and save predictions
    print("[INFO] Training the LSTM model...")
    try:
        lstm_model = LSTMModel()
        X_train, y_train, X_test, y_test, scats_test = lstm_model.load_and_prepare_data(output_dir)

        # Train the model
        lstm_model.train(X_train, y_train, epochs=10, batch_size=32)

        # Evaluate with updated method signature
        train_mse, train_mae = lstm_model.evaluate(X_train, y_train, X_test, y_test)
        print(f"[LSTM] Train MAE: {train_mae:.2f}, Test MAE: {train_mae:.2f}")

        # Save model
        lstm_model.save_model(os.path.join(models_dir, "traffic_lstm_model.keras"))

        # Predict flows
        scats_sites = [Origin, Destination]
        predicted_flows = lstm_model.predict_flows_for_scats(scats_sites, X_test, scats_test)

        # Load SCATS coordinates and build graph
        scats_data = lstm_model.load_scats_coordinates(os.path.join(output_dir, "X_test.csv"))
        scats_data = {str(k): v for k, v in scats_data.items()}
        predicted_flows = {str(k): v for k, v in predicted_flows.items()}

        graph = lstm_model.build_scats_graph(scats_data, predicted_flows)
        best_routes = lstm_model.find_optimal_routes(graph, Origin, Destination, k=5)

        # A* Search
        edges = nx_to_edge(graph)
        coords = get_coords(scats_data)
        path, nodes_created = a_star_search(Origin, {Destination}, edges, coords)
        print("[LSTM] A* Path:", path)
        print("[LSTM] Nodes Created:", nodes_created)

    except Exception as e:
            print(f"[ERROR] LSTM model training or prediction failed: {e}")

    # XGBoost Section
    print("[INFO] Training the XGBoost model...")
    try:
        xgb_model = XGBoostModel(timesteps=96)  # Include timesteps here
        X_train, y_train, X_test, y_test, scats_test = xgb_model.load_and_prepare_data(output_dir)

        xgb_model.train(X_train, y_train)

        # Updated evaluation with diagrams
        mse, mae = xgb_model.evaluate(X_test, y_test, X_train, y_train)
        print(f"[XGBoost] MSE: {mse:.2f}, MAE: {mae:.2f}")

        scats_sites = [Origin, Destination]
        predicted_flows = xgb_model.predict_flows_for_scats(scats_sites, X_test, scats_test)

        scats_data = xgb_model.load_scats_coordinates(X_test_path)
        scats_data = {str(k): v for k, v in scats_data.items()}
        predicted_flows = {str(k): v for k, v in predicted_flows.items()}

        graph = xgb_model.build_scats_graph(scats_data, predicted_flows)
        best_routes = xgb_model.find_optimal_routes(graph, Origin, Destination, k=5)

        # A* search
        edges = nx_to_edge(graph)
        coords = get_coords(scats_data)
        path, nodes_created = a_star_search(Origin, {Destination}, edges, coords)
        print("[XGBoost] A* Path:", path)
        print("[XGBoost] Nodes Created:", nodes_created)

    except Exception as e:
        print(f"[ERROR] XGBoost model training or prediction failed: {e}")

    # Train the GRU model and save predictions
    print("[INFO] Training the GRU model...")
    try:
        model = GRUModel()
        X_train, y_train, X_test, y_test, scats_test = model.load_and_prepare_data(output_dir)

        # Model training
        model.train(X_train, y_train, epochs=10, batch_size=64)

        # Evaluate the model
        mse, mae = model.evaluate(X_test, y_test, X_train, y_train)
        print(f"[GRU] MSE: {mse:.2f}, MAE: {mae:.2f}")

        # Save the trained model
        model.save_model(os.path.join(models_dir, "traffic_gru_model.keras"))

        # Predict flows for SCATS sites
        scats_sites = [Origin, Destination]
        predicted_flows = model.predict_flows_for_scats(scats_sites, X_test, scats_test)

        # Load SCATS coordinates and build the graph
        scats_data = model.load_scats_coordinates(os.path.join(output_dir, "X_test.csv"))
        scats_data = {str(k): v for k, v in scats_data.items()}
        predicted_flows = {str(k): v for k, v in predicted_flows.items()}

        graph = model.build_scats_graph(scats_data, predicted_flows)
        best_routes = model.find_optimal_routes(graph, Origin, Destination, k=5)

        # Perform A* search
        edges = nx_to_edge(graph)
        coords = get_coords(scats_data)
        origin = Origin
        destinations = {Destination}

        path, nodes_created = a_star_search(origin, destinations, edges, coords)
        print("[GRU] A* Path:", path)
        print("[GRU] Nodes Created:", nodes_created)
    except Exception as e:
        print(f"[ERROR] GRU model training or prediction failed: {e}")
        
