import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from math import radians, sin, cos, sqrt, atan2
        
class GRUModel:
    #initialising instances of the model 
    def __init__(self, timesteps = 96, n_features = 185):
        self.timesteps = timesteps
        self.n_features = n_features
        self.scaler = MinMaxScaler(feature_range = (0, 1))
        self.model = self._build_model()
    
    #building model using sequential 
    def _build_model(self):
        model = Sequential([
            Input(shape = (self.timesteps, self.n_features)),
            GRU(128, return_sequences = True),
            Dropout(0.3),
            GRU(64, return_sequences = True),
            Dropout(0.3),
            GRU(32),
            Dropout(0.3),
            Dense(self.timesteps)
        ])
        
        model.compile(optimizer = Adam(0.001), loss = 'mse', metrics = ['mae'])
        return model
        
    def train(self, X_train, y_train, epochs = 100, batch_size = 32, validation_split = 0.1):
        #monitoring algorithms to prevent stalling and saves the best model
        callbacks = [EarlyStopping(patience = 15, restore_best_weights = True), ModelCheckpoint('best_model.keras', save_best_only = True)]
        
        #plots training validation
        history = self.model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_split = validation_split, callbacks = callbacks, verbose = 1)
        
        plt.figure(figsize = (10, 5))
        plt.plot(history.history['loss'], label = 'Training Loss')
        plt.plot(history.history['val_loss'], label = 'Validation Loss')
        plt.title('Model Training History')
        plt.legend()
        plt.show()
    
    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test, verbose = 0)

        # inverses the test
        predictions_inv = self.scaler.inverse_transform(predictions.reshape(-1, 1))
        y_test_inv = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # margin of error calculations
        mse = mean_squared_error(y_test_inv, predictions_inv)
        mae = mean_absolute_error(y_test_inv, predictions_inv)
        
        # Plot first sample (reshape back to original timesteps)
        plt.figure(figsize = (15, 5))
        plt.plot(y_test_inv[:self.timesteps], label = 'Actual')
        plt.plot(predictions_inv[:self.timesteps], label = 'Predicted')
        plt.title('Flow Prediction')
        plt.ylabel('Flow')
        plt.xlabel('Time Steps')
        plt.legend()
        plt.show()
    
        return mse, mae
    
    def load_and_prepare_data(self, data_dir):
        # Load data
        X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
        y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
        
        scats_col = next((c for c in X_train.columns if 'scats' in c.lower()), None)
            
        # Standardize column name
        X_train = X_train.rename(columns={scats_col: 'SCATS_Number'}) #CHECK HERE IF FAIL
        scats_numbers = X_train['SCATS_Number'].copy()
        
        # Process features
        if 'Date' in X_train.columns:
            X_train['Date'] = pd.to_datetime(X_train['Date'])
            X_train['hour'] = X_train['Date'].dt.hour
            X_train['day_of_week'] = X_train['Date'].dt.dayofweek
            X_train['is_weekend'] = X_train['day_of_week'].isin([5,6]).astype(int)
            X_train.drop('Date', axis=1, inplace=True)

        X_train = pd.get_dummies(X_train.drop('SCATS_Number', axis=1))
        
        # numpy values
        X_train = X_train.astype('float32').values
        y_train = y_train.astype('float32').values
        
        # Reshape y to (samples, timesteps, 1)
        if y_train.ndim == 2:  # If shape is (samples, timesteps)
            y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
        X_train = np.tile(X_train[:, np.newaxis, :], (1, y_train.shape[1], 1))
        
        # Split data
        split = int(0.8 * len(X_train))
        X_train, X_test = X_train[:split], X_train[split:]
        y_train, y_test = y_train[:split], y_train[split:]
        scats_test = scats_numbers.values[split:]
        
        # Scaling
        y_train = self.scaler.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
        y_test = self.scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
        
        return X_train, y_train, X_test, y_test, scats_test
    
    def save_model(self, filepath):
        self.model.save(filepath)
    
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

    def flow_to_speed(self, flow):
        if flow <= 1800:  # Adjusted capacity
            speed = 50 - (flow / 60)  # Simpler linear relationship
        else:
            speed = 20 - (flow - 1800) / 100
        return min(max(speed, 5), 60)

    def calculate_travel_time(self, distanceance_km, predicted_flow):
        speed = self.flow_to_speed(predicted_flow)
        travel_time = (distanceance_km / speed)
        return  travel_time * 3600 + 30   # 30s intersection delay
    
    def predict_flows_for_scats(self, scats_sites, X_test, scats_test):
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
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

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
    
    def find_optimal_routes(self, graph, origin_scats, dest_scats, k = 3):
        routes = []
        for path in nx.shortest_simple_paths(graph, origin_scats, dest_scats, weight = "weight"):
            total_time = sum(graph[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            routes.append((path, total_time))
            if len(routes) == k:
                break

        # Print converted to minutes
        for i, (path, total_seconds) in enumerate(routes, 1):
            print(f"Route {i}: {' → '.join(path)} | Time: {total_seconds / 60:.1f} mins")
        return routes
