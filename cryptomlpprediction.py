"""
Cryptocurrency Price Prediction and Trading Signal Generation

This module implements a price prediction system for cryptocurrencies using MLP and LSTM models.
It supports data preprocessing, model training, evaluation, and signal generation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import os
import logging

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class CryptoPredictor:
    """
    A class for cryptocurrency price prediction and trading signal generation.
    
    Attributes:
        coin (str): Name of the cryptocurrency
        data_path (str): Path to the CSV file containing historical data
        df (pd.DataFrame): Processed DataFrame
        scaler_X, scaler_y (MinMaxScaler): Scalers for features and target
        X_train, X_test, y_train, y_test: Train-test split data
        models (dict): Trained models
        predictions (dict): Model predictions
        signals (dict): Trading signals
        performances (dict): Model performance metrics
    """
    
    def __init__(self, coin_name, data_path):
        self.coin = coin_name
        self.data_path = data_path
        self.df = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_dates = None
        self.test_actual = None
        self.models = {}
        self.predictions = {}
        self.signals = {}
        self.performances = {}
        self.feature_names = []
        os.makedirs(f'outputs/{coin_name}', exist_ok=True)
        logging.info(f"Initialized CryptoPredictor for {coin_name}")
    
    def load_data(self):
        """Load and preprocess cryptocurrency data from CSV."""
        logging.info(f"Loading data for {self.coin} from {self.data_path}")
        df = pd.read_csv(self.data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            logging.warning(f"Found {missing_count} missing values. Filling with forward/backward fill.")
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        df = df.reset_index(drop=True)
        self.df = df
        logging.info(f"Data loaded. Shape: {df.shape}")
        return df
    
    def prepare_features(self, window_size=1):
        """Prepare features using previous days' data."""
        logging.info(f"Preparing features with window_size={window_size}")
        df = self.df.copy()
        
        for i in range(1, window_size + 1):
            df[f'Open_lag_{i}'] = df['Open'].shift(i)
            df[f'High_lag_{i}'] = df['High'].shift(i)
            df[f'Low_lag_{i}'] = df['Low'].shift(i)
            df[f'Close_lag_{i}'] = df['Close'].shift(i)
            df[f'Volume_lag_{i}'] = df['Volume'].shift(i)
        
        df = df.dropna()
        dates = df['Date']
        actual_close = df['Close']
        lag_columns = [col for col in df.columns if 'lag' in col]
        self.feature_names = lag_columns
        
        X = self.scaler_X.fit_transform(df[lag_columns])
        y = df['Close'].values.reshape(-1, 1)
        y = self.scaler_y.fit_transform(y)
        
        logging.info(f"Features prepared. X shape: {X.shape}, y shape: {y.shape}")
        return X, y, dates, actual_close
    
    def split_data(self, X, y, test_size=0.2):
        """Split data into training and testing sets."""
        logging.info(f"Splitting data with test_size={test_size}")
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        test_dates_idx = int(len(self.df) * (1 - test_size))
        self.test_dates = self.df['Date'].iloc[test_dates_idx:test_dates_idx + len(X_test)].reset_index(drop=True)
        self.test_actual = self.df['Close'].iloc[test_dates_idx:test_dates_idx + len(X_test)].reset_index(drop=True)
        
        logging.info(f"Train shape: X={X_train.shape}, y={y_train.shape}")
        logging.info(f"Test shape: X={X_test.shape}, y={y_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def train_mlp(self, hyperparameter_tuning=False):
        """Train an MLP model with Adam optimizer."""
        logging.info("Training MLP model")
        if hyperparameter_tuning:
            logging.info("Performing hyperparameter tuning for MLP")
            param_grid = {
                'learning_rate_init': [0.001, 0.01]
            }
            mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, solver='adam', random_state=42)
            grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train.ravel())
            model = grid_search.best_estimator_
            logging.info(f"Best MLP parameters: {grid_search.best_params_}")
        else:
            model = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                solver='adam',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42
            )
            model.fit(self.X_train, self.y_train.ravel())
        
        self.models['mlp'] = model
        logging.info("MLP training completed")
        return model
    
    def train_lstm(self, hyperparameter_tuning=False):
        """Train an LSTM model with Adam optimizer."""
        logging.info("Training LSTM model")
        timesteps = 1
        n_features = self.X_train.shape[1]
        X_train_lstm = self.X_train.reshape(self.X_train.shape[0], timesteps, n_features)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        if hyperparameter_tuning:
            logging.info("Performing hyperparameter tuning for LSTM")
            learning_rates = [0.001, 0.01]
            best_val_loss = float('inf')
            best_model = None
            
            for lr in learning_rates:
                model = Sequential([
                    LSTM(units=64, return_sequences=True, input_shape=(timesteps, n_features)),
                    Dropout(0.2),
                    LSTM(units=32),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
                history = model.fit(
                    X_train_lstm, self.y_train,
                    epochs=50, batch_size=32, validation_split=0.2,
                    callbacks=[early_stopping], verbose=0
                )
                val_loss = min(history.history['val_loss'])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model
                    logging.info(f"New best LSTM: lr={lr}, val_loss={val_loss:.4f}")
            model = best_model
        else:
            model = Sequential([
                LSTM(units=64, return_sequences=True, input_shape=(timesteps, n_features)),
                Dropout(0.2),
                LSTM(units=32),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        history = model.fit(
            X_train_lstm, self.y_train,
            epochs=50, batch_size=32, validation_split=0.2,
            callbacks=[early_stopping], verbose=1
        )
        
        plt.figure(figsize=(8, 4))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{self.coin} - LSTM Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'outputs/{self.coin}/lstm_training_history.png')
        plt.close()
        
        self.models['lstm'] = model
        self.models['lstm_reshape'] = {'timesteps': timesteps, 'n_features': n_features}
        logging.info("LSTM training completed")
        return model
    
    def predict_and_evaluate(self, model_name):
        """Make predictions and evaluate model performance."""
        logging.info(f"Evaluating {model_name} model")
        model = self.models[model_name]
        
        if model_name == 'lstm':
            timesteps = self.models['lstm_reshape']['timesteps']
            n_features = self.models['lstm_reshape']['n_features']
            X_test_reshaped = self.X_test.reshape(self.X_test.shape[0], timesteps, n_features)
            predictions = model.predict(X_test_reshaped)
        else:
            predictions = model.predict(self.X_test).reshape(-1, 1)
        
        predictions = self.scaler_y.inverse_transform(predictions)
        actual = self.scaler_y.inverse_transform(self.y_test)
        
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mape = mean_absolute_percentage_error(actual, predictions) * 100
        
        self.predictions[model_name] = predictions
        self.performances[model_name] = {'rmse': rmse, 'mape': mape}
        logging.info(f"{model_name.upper()} - RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        
        return predictions, rmse, mape
    
    def generate_signals(self, model_name, threshold=0.01):
        """Generate trading signals with confidence scores."""
        logging.info(f"Generating signals for {model_name} with threshold={threshold}")
        if model_name not in self.predictions:
            raise ValueError(f"No predictions for {model_name}")
        
        predictions = self.predictions[model_name]
        actual = self.scaler_y.inverse_transform(self.y_test)
        
        signals_df = pd.DataFrame({
            'Date': self.get_test_dates(),
            'Actual': actual.flatten(),
            'Predicted': predictions.flatten(),
            'Close': actual.flatten()
        })
        
        signals_df['Diff_Pct'] = (signals_df['Predicted'] - signals_df['Actual']) / signals_df['Actual'] * 100
        signals_df['Signal'] = 'HOLD'
        signals_df.loc[signals_df['Diff_Pct'] > threshold * 100, 'Signal'] = 'BUY'
        signals_df.loc[signals_df['Diff_Pct'] < -threshold * 100, 'Signal'] = 'SELL'
        
        #  confidence score based on deviation magnitude
        signals_df['Confidence'] = np.abs(signals_df['Diff_Pct']) / (threshold * 100)
        signals_df['Confidence'] = signals_df['Confidence'].clip(0, 1)
        
        self.signals[model_name] = signals_df
        signals_df.to_csv(f'outputs/{self.coin}/{model_name}_signals.csv', index=False)
        logging.info(f"Signals generated: {signals_df['Signal'].value_counts().to_dict()}")
        return signals_df
    
    def export_performance(self):
        """Export model performance metrics to CSV."""
        logging.info("Exporting model performance metrics")
        comparison = pd.DataFrame({
            'Model': list(self.performances.keys()),
            'RMSE': [self.performances[model]['rmse'] for model in self.performances],
            'MAPE': [self.performances[model]['mape'] for model in self.performances]
        })
        comparison.to_csv(f'outputs/{self.coin}/model_performance.csv', index=False)
        logging.info(f"Performance metrics exported to outputs/{self.coin}/model_performance.csv")
        return comparison
    
    def get_test_dates(self):
        """Return test set dates."""
        if self.test_dates is None:
            raise ValueError("Test dates not available")
        return self.test_dates
    
    def get_test_actual(self):
        """Return test set actual prices."""
        if self.test_actual is None:
            raise ValueError("Test actual values not available")
        return self.test_actual
    
    def get_feature_names(self):
        """Return feature names."""
        return self.feature_names