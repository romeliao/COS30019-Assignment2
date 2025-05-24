from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import networkx as nx
from math import radians, sin, cos, sqrt, atan2

class LSTMModel:
    def __init__(self, timesteps=96, n_features=185):
        # Initialize model parameters and scaler
        self.timesteps = timesteps
        self.n_features = n_features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._build_model()
    
    def _build_model(self):
        # Build the LSTM neural network architecture
        model = Sequential([
            Input(shape=(self.timesteps, self.n_features)),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(self.timesteps)
        ])
        model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
        return model

    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.1):
        # Train the LSTM model with early stopping and model checkpointing
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ModelCheckpoint(os.path.join("models", "best_lstm_model.keras"), save_best_only=True)
        ]
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                 validation_split=validation_split, callbacks=callbacks, verbose=1)
        # Plot training and validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.legend()
        plt.show()

    def evaluate(self, X_train, y_train, X_test, y_test):
        # Evaluate model performance on train and test sets

        # Predict on test set
        preds_test = self.model.predict(X_test, verbose=0)
        preds_test_inv = self.scaler.inverse_transform(preds_test.reshape(-1, 1))
        y_test_inv = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        mae_test = mean_absolute_error(y_test_inv, preds_test_inv)

        # Predict on train set
        preds_train = self.model.predict(X_train, verbose=0)
        preds_train_inv = self.scaler.inverse_transform(preds_train.reshape(-1, 1))
        y_train_inv = self.scaler.inverse_transform(y_train.reshape(-1, 1))
        mae_train = mean_absolute_error(y_train_inv, preds_train_inv)

        mse = mean_squared_error(y_test_inv, preds_test_inv)
        mae = mean_absolute_error(y_test_inv, preds_test_inv)

        # Plot actual vs predicted flows for test set
        plt.figure(figsize=(15, 5))
        plt.plot(y_test_inv[:self.timesteps], label='Actual')
        plt.plot(preds_test_inv[:self.timesteps], label='Predicted')
        plt.title('Flow Prediction')
        plt.ylabel('Flow')
        plt.xlabel('Time Steps')
        plt.legend()
        plt.show()

        # Plot MAE for train and test sets
        plt.figure(figsize=(6, 6))
        plt.bar(['Train', 'Test'], [mae_train, mae_test], color=['skyblue', 'salmon'])
        plt.title('Final MAE: Train vs Test')
        plt.ylabel('MAE')
        plt.show()

        return mae_train, mae_test

    def load_and_prepare_data(self, data_dir):
        # Load and preprocess training data
        X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
        y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))

        # Find SCATS column
        scats_col = next((c for c in X_train.columns if 'scats' in c.lower()), None)
        if scats_col is None:
            raise ValueError("SCATS column not found in dataset.")
        X_train = X_train.rename(columns={scats_col: 'SCATS_Number'})
        scats_numbers = X_train['SCATS_Number'].copy()

        # Feature engineering for date/time
        if 'Date' in X_train.columns:
            X_train['Date'] = pd.to_datetime(X_train['Date'])
            X_train['hour'] = X_train['Date'].dt.hour
            X_train['day_of_week'] = X_train['Date'].dt.dayofweek
            X_train['is_weekend'] = X_train['day_of_week'].isin([5,6]).astype(int)
            X_train.drop('Date', axis=1, inplace=True)

        # One-hot encode categorical variables and convert to numpy arrays
        X_train = pd.get_dummies(X_train.drop('SCATS_Number', axis=1))
        X_train = X_train.astype('float32').values
        y_train = y_train.astype('float32').values

        # Reshape y_train if needed
        if y_train.ndim == 2:
            y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
        # Tile X_train to match y_train's time dimension
        X_train = np.tile(X_train[:, np.newaxis, :], (1, y_train.shape[1], 1))

        # Split into train and test sets
        split = int(0.8 * len(X_train))
        X_train, X_test = X_train[:split], X_train[split:]
        y_train, y_test = y_train[:split], y_train[split:]
        scats_test = scats_numbers.values[split:]

        # Scale y values
        y_train = self.scaler.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
        y_test = self.scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

        return X_train, y_train, X_test, y_test, scats_test

    def save_model(self, filepath):
        # Save the trained model to disk
        self.model.save(filepath)

    def load_scats_coordinates(self, test_csv_path):
        # Load SCATS site coordinates from CSV
        df_test = pd.read_csv(test_csv_path)
        scats_data = (
            df_test[["SCATS Number", "NB_LATITUDE", "NB_LONGITUDE"]]
            .drop_duplicates("SCATS Number")
            .set_index("SCATS Number")
            .rename(columns={"NB_LATITUDE": "Latitude", "NB_LONGITUDE": "Longitude"})
            .to_dict(orient="index")
        )
        return scats_data

    def flow_to_speed(self, flow):
        # Convert traffic flow to speed using a simple linear relationship
        speed = 50 - (flow / 60)  # Simpler linear relationship
        return min(max(speed, 5), 60)   

    def calculate_travel_time(self, distance_km, predicted_flow):
        # Calculate travel time (in seconds) for a given distance and predicted flow
        speed = self.flow_to_speed(predicted_flow)
        travel_time = (distance_km / speed)
        return  travel_time * 3600 + 30   # 30s intersection delay
        
    def predict_flows_for_scats(self, scats_sites, X_test, scats_test):
        # Predict mean flow for each SCATS site in the test set
        flows = {}
        scats_test = np.array(scats_test).astype(str)

        for scats in scats_sites:
            mask = scats_test== scats
            if not np.any(mask):
                flows[scats] = 500  # Default flow if no data
                continue

            X_scats = X_test[mask]

            preds = [self.model.predict(X_scats[i:i+1], verbose=0)[0][0] for i in range(len(X_scats))]
            mean_pred_scaled = np.mean(preds).reshape(-1, 1)
            flows[scats] = self.scaler.inverse_transform(mean_pred_scaled)[0, 0]

        return flows
        
    def haversine(self, lat1, lon1, lat2, lon2):
        # Calculate the great-circle distance between two points on the Earth
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    def build_scats_graph(self, scats_data, predicted_flows):
        # Build a graph of SCATS sites with travel time as edge weights
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
        
    def find_optimal_routes(self, graph, origin_scats, dest_scats, k=3):
        # Find k shortest routes between two SCATS sites using travel time as weight
        routes = []
        for path in nx.shortest_simple_paths(graph, origin_scats, dest_scats, weight="weight"):
            total_time = sum(graph[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            routes.append((path, total_time))
            if len(routes) == k:
                break

        # Print routes and their travel times in minutes
        for i, (path, total_seconds) in enumerate(routes, 1):
            print(f"Route {i}: {' -> '.join(path)} | Time: {total_seconds / 60:.1f} mins")
        return routes