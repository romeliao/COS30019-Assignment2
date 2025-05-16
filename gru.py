import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

class GRUModel:
    #initialising instances of the model 
    def __init__(self, timesteps = 96, n_features = 186):
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
            
            # numpy values
            X_train = pd.get_dummies(X_train).astype('float32').values
            y_train = y_train.astype('float32').values
            
            # Reshape y to (samples, timesteps, 1)
            y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
            X_train = np.tile(X_train[:, np.newaxis, :], (1, y_train.shape[1], 1))
            
            # Split data
            split = int(0.8 * len(X_train))
            X_train, X_test = X_train[:split], X_train[split:]
            y_train, y_test = y_train[:split], y_train[split:]
            
            # Scaling
            y_train = self.scaler.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
            y_test = self.scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
            
            print(f"\nX_train: {X_train.shape}, y_train: {y_train.shape}")
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            print(f"\nError: {e}")
            raise
    
    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = load_model(filepath)