import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for TensorFlow

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from math import radians, sin, cos, sqrt, atan2

class GRUModel:
    # Initialize the GRU model, scaler, and model architecture
    def __init__(self, timesteps=96, n_features=185):
        self.timesteps = timesteps
        self.n_features = n_features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._build_model()

    # Build and compile the GRU neural network
    def _build_model(self):
        model = Sequential([
            Input(shape=(self.timesteps, self.n_features)),
            GRU(128, return_sequences=True),
            Dropout(0.3),
            GRU(64, return_sequences=True),
            Dropout(0.3),
            GRU(32),
            Dropout(0.3),
            Dense(self.timesteps)  # output shape: (batch_size, timesteps)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    # Train the model with early stopping and checkpointing
    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.1):
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
            ModelCheckpoint('best_model.keras', save_best_only=True, verbose=1)
        ]
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        # Plot training and validation loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.show()

    # Evaluate the model and plot predictions and MAE
    def evaluate(self, X_test, y_test, X_train, y_train):
        preds_test = self.model.predict(X_test)
        preds_train = self.model.predict(X_train)

        # Inverse scale predictions and targets
        preds_test_inv = self.scaler.inverse_transform(preds_test.reshape(-1, 1)).flatten()
        y_test_inv = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        preds_train_inv = self.scaler.inverse_transform(preds_train.reshape(-1, 1)).flatten()
        y_train_inv = self.scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()

        mse_test = mean_squared_error(y_test_inv, preds_test_inv)
        mae_test = mean_absolute_error(y_test_inv, preds_test_inv)
        mae_train = mean_absolute_error(y_train_inv, preds_train_inv)

        # Plot predictions vs actual for the first test sample
        plt.figure(figsize=(15, 5))
        plt.plot(y_test_inv[:self.timesteps], label='Actual')
        plt.plot(preds_test_inv[:self.timesteps], label='Predicted')
        plt.title('Sample Flow Prediction on Test Set')
        plt.xlabel('Time Steps')
        plt.ylabel('Flow')
        plt.legend()
        plt.show()

        # Plot MAE for train and test
        plt.figure(figsize=(6, 6))
        plt.bar(['Train', 'Test'], [mae_train, mae_test], color=['blue', 'orange'])
        plt.title('Mean Absolute Error: Train vs Test')
        plt.ylabel('MAE')
        plt.show()

        return mse_test, mae_test

    # Load and preprocess data from CSV files
    def load_and_prepare_data(self, data_dir):
        X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
        y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))

        # Find and standardize SCATS column name
        scats_col = next((c for c in X_train.columns if 'scats' in c.lower()), None)
        if scats_col is None:
            raise ValueError("SCATS Number column not found in X_train")
        X_train = X_train.rename(columns={scats_col: 'SCATS_Number'})
        scats_numbers = X_train['SCATS_Number'].copy()

        # Feature engineering for date/time columns
        if 'Date' in X_train.columns:
            X_train['Date'] = pd.to_datetime(X_train['Date'])
            X_train['hour'] = X_train['Date'].dt.hour
            X_train['day_of_week'] = X_train['Date'].dt.dayofweek
            X_train['is_weekend'] = X_train['day_of_week'].isin([5, 6]).astype(int)
            X_train.drop('Date', axis=1, inplace=True)

        # One-hot encode categorical variables except SCATS_Number
        X_train = pd.get_dummies(X_train.drop('SCATS_Number', axis=1))

        # Convert to numpy arrays
        X_train = X_train.astype('float32').values
        y_train = y_train.astype('float32').values

        # Reshape y_train if needed
        if y_train.ndim == 2:
            y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)

        # Repeat X features for each timestep
        X_train = np.tile(X_train[:, np.newaxis, :], (1, y_train.shape[1], 1))

        # Split into train and test sets (80/20 split)
        split_idx = int(0.8 * len(X_train))
        X_train_split, X_test_split = X_train[:split_idx], X_train[split_idx:]
        y_train_split, y_test_split = y_train[:split_idx], y_train[split_idx:]
        scats_test_split = scats_numbers.values[split_idx:]

        # Scale y data
        y_train_scaled = self.scaler.fit_transform(y_train_split.reshape(-1, 1)).reshape(y_train_split.shape)
        y_test_scaled = self.scaler.transform(y_test_split.reshape(-1, 1)).reshape(y_test_split.shape)

        return X_train_split, y_train_scaled, X_test_split, y_test_scaled, scats_test_split

    # Save the trained model to a file
    def save_model(self, filepath):
        self.model.save(filepath)

    # Load a trained model from a file
    def load_model(self, filepath):
        self.model = keras_load_model(filepath)

    # Load SCATS coordinates from a CSV file
    def load_scats_coordinates(self, test_csv_path):
        df_test = pd.read_csv(test_csv_path)
        scats_data = (
            df_test[["SCATS Number", "NB_LATITUDE", "NB_LONGITUDE"]]
            .drop_duplicates("SCATS Number")
            .set_index("SCATS Number")
            .rename(columns={"NB_LATITUDE": "Latitude", "NB_LONGITUDE": "Longitude"})
            .to_dict(orient="index")
        )
        return scats_data

    # Convert predicted flow to estimated speed (km/h)
    def flow_to_speed(self, flow):
        speed = 50 - (flow / 60)  # Simpler linear relationship
        return min(max(speed, 5), 60)

    # Calculate travel time (in seconds) for a segment
    def calculate_travel_time(self, distance_km, predicted_flow):
        speed = self.flow_to_speed(predicted_flow)
        travel_time = (distance_km / speed)
        return travel_time * 3600 + 30   # 30s intersection delay

    # Predict average traffic flow for each SCATS site in the list
    def predict_flows_for_scats(self, scats_sites, X_test, scats_test):
        flows = {}
        scats_test = np.array(scats_test).astype(str)

        for scats in scats_sites:
            mask = scats_test == scats
            if not np.any(mask):
                flows[scats] = 500  # Default flow if no data
                continue

            X_scats = X_test[mask]

            preds = [self.model.predict(X_scats[i:i+1], verbose=0)[0][0] for i in range(len(X_scats))]
            mean_pred_scaled = np.mean(preds).reshape(-1, 1)
            flows[scats] = self.scaler.inverse_transform(mean_pred_scaled)[0, 0]

        return flows

    # Calculate the great-circle distance between two points (Haversine formula)
    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    # Build a weighted graph of SCATS sites using predicted flows as edge weights
    def build_scats_graph(self, scats_data, predicted_flows):
        G = nx.Graph()
        for origin, origin_data in scats_data.items():
            for dest, dest_data in scats_data.items():
                if origin_data != dest:
                    distance = self.haversine(
                        origin_data["Latitude"], origin_data["Longitude"],
                        dest_data["Latitude"], dest_data["Longitude"]
                    )
                    flow = predicted_flows.get(dest, 500)
                    travel_time = self.calculate_travel_time(distance, flow)
                    G.add_edge(origin, dest, weight = travel_time)  # Weight in seconds
        return G

    # Find the top-k shortest routes between two SCATS sites using travel time as weight
    def find_optimal_routes(self, graph, origin_scats, dest_scats, k=3):
        routes = []
        for path in nx.shortest_simple_paths(graph, origin_scats, dest_scats, weight="weight"):
            total_time = sum(graph[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            routes.append((path, total_time))
            if len(routes) == k:
                break

        # Print routes and their total travel time in minutes
        for i, (path, total_seconds) in enumerate(routes, 1):
            print(f"Route {i}: {' -> '.join(path)} | Time: {total_seconds / 60:.1f} mins")
        return routes