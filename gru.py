import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

class TrafficGRUPredictor:
    def __init__(self, timesteps=96, n_features=186):
        self.timesteps = timesteps
        self.n_features = n_features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential([
            Input(shape=(self.timesteps, self.n_features)),
            GRU(128, return_sequences=True),
            Dropout(0.3),
            GRU(64, return_sequences=True),
            Dropout(0.3),
            GRU(32),
            Dropout(0.3),
            Dense(self.timesteps)  # Output for all timesteps
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        model.summary()
        return model
        
    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.1):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ModelCheckpoint('best_gru_model.h5', monitor='val_loss', save_best_only=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()
    
    def evaluate(self, X_test, y_test):
        # Reshape y_test to 2D for inverse transform
        original_shape = y_test.shape
        y_test_flat = y_test.reshape(-1, 1)
        
        predictions = self.model.predict(X_test)
        predictions_flat = predictions.reshape(-1, 1)
        
        # Inverse transform
        predictions_inv = self.scaler.inverse_transform(predictions_flat).reshape(original_shape)
        y_test_inv = self.scaler.inverse_transform(y_test_flat).reshape(original_shape)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_inv, predictions_inv)
        mae = mean_absolute_error(y_test_inv, predictions_inv)
        
        # Plot first sample
        plt.figure(figsize=(15, 5))
        plt.plot(y_test_inv[0], label='Actual')
        plt.plot(predictions_inv[0], label='Predicted')
        plt.title('Traffic Flow Prediction')
        plt.ylabel('Flow (vehicles/hour)')
        plt.xlabel('Time Steps (15-min intervals)')
        plt.legend()
        plt.show()
        
        return mse, mae
    
    def load_and_prepare_data(self, data_dir):
        try:
            # Load data
            X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
            y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
            
            # Process features
            if 'Date' in X_train.columns:
                X_train['Date'] = pd.to_datetime(X_train['Date'])
                X_train['hour'] = X_train['Date'].dt.hour
                X_train['day_of_week'] = X_train['Date'].dt.dayofweek
                X_train['is_weekend'] = X_train['day_of_week'].isin([5,6]).astype(int)
                X_train.drop('Date', axis=1, inplace=True)
            
            # One-hot encode categoricals
            cat_cols = X_train.select_dtypes(include=['object']).columns
            if not cat_cols.empty:
                X_train = pd.get_dummies(X_train, columns=cat_cols)
            
            # Convert to numpy
            X_train = X_train.astype('float32').values
            y_train = y_train.astype('float32').values
            
            # Reshape y to (samples, timesteps, 1)
            y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
            
            # Tile X features across timesteps
            X_train = np.tile(X_train[:, np.newaxis, :], (1, y_train.shape[1], 1))
            
            # Split data
            split = int(0.8 * X_train.shape[0])
            X_train, X_test = X_train[:split], X_train[split:]
            y_train, y_test = y_train[:split], y_train[split:]
            
            # Scale targets
            y_train = self.scaler.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
            y_test = self.scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
            
            print(f"\nData loaded - X_train: {X_train.shape}, y_train: {y_train.shape}")
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            print(f"\nError loading data: {str(e)}")
            raise
    
    def save_model(self, filepath):
        self.model.save(filepath)
    
    def load_model(self, filepath):
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_dataset")
    
    print("=== GRU Traffic Prediction ===")
    try:
        predictor = TrafficGRUPredictor()
        X_train, y_train, X_test, y_test = predictor.load_and_prepare_data(PROCESSED_DATA_DIR)
        
        print("\n=== Training ===")
        predictor.train(X_train, y_train, epochs=50, batch_size=64)
        
        print("\n=== Evaluation ===")
        mse, mae = predictor.evaluate(X_test, y_test)
        print(f"MSE: {mse:.2f}, MAE: {mae:.2f}")
        
        model_path = os.path.join(BASE_DIR, "traffic_gru_model.h5")
        predictor.save_model(model_path)
        print(f"\nModel saved to: {model_path}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")