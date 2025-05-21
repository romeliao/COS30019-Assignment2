import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from math import radians, sin, cos, sqrt, atan2

class XGBoostModel:
    def __init__(self, timesteps=96):
        self.timesteps = timesteps
        self.model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
        self.scaler = MinMaxScaler()

    def load_and_prepare_data(self, data_dir):
        X = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
        y = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
        
        scats_col = next((c for c in X.columns if 'scats' in c.lower()), None)
        X = X.rename(columns={scats_col: 'SCATS_Number'})
        scats_numbers = X['SCATS_Number'].copy()

        if 'Date' in X.columns:
            X['Date'] = pd.to_datetime(X['Date'])
            X['hour'] = X['Date'].dt.hour
            X['day_of_week'] = X['Date'].dt.dayofweek
            X['is_weekend'] = X['day_of_week'].isin([5,6]).astype(int)
            X.drop('Date', axis=1, inplace=True)

        X = pd.get_dummies(X.drop('SCATS_Number', axis=1))
        y = y.mean(axis=1).values  # Reduce to 1D for regression

        X = X.astype('float32').values
        y = y.astype('float32')

        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1)).flatten()

        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y_scaled[:split], y_scaled[split:]
        scats_test = scats_numbers.values[split:]
        
        return X_train, y_train, X_test, y_test, scats_test

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def save_model(self, filepath):
        """Save the trained XGBoost model to a file."""
        self.model.save_model(filepath)
        print(f"[INFO] XGBoost model saved to {filepath}")

    def load_model(self, filepath):
        """Load a pre-trained XGBoost model from a file."""
        self.model.load_model(filepath)
        print(f"[INFO] XGBoost model loaded from {filepath}")

    def evaluate(self, X_test, y_test, X_train=None, y_train=None):
        preds_test = self.model.predict(X_test)
        preds_test_inv = self.scaler.inverse_transform(preds_test.reshape(-1, 1)).flatten()
        y_test_inv = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        mae_test = mean_absolute_error(y_test_inv, preds_test_inv)

        # Diagram 2: Test Prediction vs Actual
        plt.figure(figsize=(15, 5))
        plt.plot(y_test_inv[:self.timesteps], label='Actual')
        plt.plot(preds_test_inv[:self.timesteps], label='Predicted')
        plt.title('Flow Prediction')
        plt.ylabel('Flow')
        plt.xlabel('Time Steps')
        plt.legend()
        plt.show()

        # Calculate train MAE if training data is provided
        if X_train is not None and y_train is not None:
            preds_train = self.model.predict(X_train)
            preds_train_inv = self.scaler.inverse_transform(preds_train.reshape(-1, 1)).flatten()
            y_train_inv = self.scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
            mae_train = mean_absolute_error(y_train_inv, preds_train_inv)
        else:
            mae_train = 0

        # Diagram 3: MAE Comparison
        plt.figure(figsize=(6, 6))
        plt.bar(['Train', 'Test'], [mae_train, mae_test], color=['skyblue', 'salmon'])
        plt.title('Final MAE: Train vs Test')
        plt.ylabel('MAE')
        plt.show()

        mse = mean_squared_error(y_test_inv, preds_test_inv)
        return mse, mae_test

    def predict_flows_for_scats(self, scats_sites, X_test, scats_test):
        flows = {}
        scats_test = np.array(scats_test).astype(str)
        
        for scats in scats_sites:
            mask = scats_test == scats
            if not np.any(mask):
                flows[scats] = 500
                continue

            X_scats = X_test[mask]
            preds = self.model.predict(X_scats)
            mean_pred = np.mean(preds).reshape(-1, 1)
            flows[scats] = self.scaler.inverse_transform(mean_pred)[0, 0]

        return flows

    def flow_to_speed(self, flow):
        if flow <= 1800:
            speed = 50 - (flow / 60)
        else:
            speed = 20 - (flow - 1800) / 100
        return min(max(speed, 5), 60)

    def calculate_travel_time(self, distance_km, predicted_flow):
        speed = self.flow_to_speed(predicted_flow)
        travel_time = (distance_km / speed) * 3600 + 30  # Seconds
        return travel_time

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

    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    def build_scats_graph(self, scats_data, predicted_flows):
        G = nx.Graph()
        for origin, o_data in scats_data.items():
            for dest, d_data in scats_data.items():
                if origin != dest:
                    dist = self.haversine(o_data["Latitude"], o_data["Longitude"], d_data["Latitude"], d_data["Longitude"])
                    flow = predicted_flows.get(dest, 500)
                    time = self.calculate_travel_time(dist, flow)
                    G.add_edge(origin, dest, weight=time)
        return G

    def find_optimal_routes(self, graph, origin_scats, dest_scats, k=3):
        routes = []
        for path in nx.shortest_simple_paths(graph, origin_scats, dest_scats, weight="weight"):
            total_time = sum(graph[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            routes.append((path, total_time))
            if len(routes) == k:
                break
        for i, (path, total_sec) in enumerate(routes, 1):
            print(f"Route {i}: {' -> '.join(path)} | Time: {total_sec/60:.1f} mins")
        return routes