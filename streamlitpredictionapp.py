"""
Streamlit App for Cryptocurrency Price Prediction

This module provides a simple web interface for predicting cryptocurrency prices and generating trading signals
using MLP and LSTM models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
from io import BytesIO
from cryptomlpprediction import CryptoPredictor

#  predefined dataset paths
PREDEFINED_DATASETS = {
    "Bitcoin": "bitcoin.csv",
    "BNB": "BNB.csv",
    "Dogecoin": "dogecoin.csv",
    "Solana": "solana.csv"
}

# Dataset to symbol mapping
DATASET_SYMBOLS = {
    "Bitcoin": "BTC",
    "BNB": "BNB",
    "Dogecoin": "DOGE",
    "Solana": "SOL"
}

# Set page configuration
st.set_page_config(
    page_title="Crypto Price Prediction",
    page_icon="📈",
    layout="centered"
)

#  CSS
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    h1, h2, h3 { color: #1e3a8a; }
    .metric-card { background-color: #ffffff; border-radius: 5px; padding: 15px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; }
    .metric-value { font-size: 20px; font-weight: bold; color: #1e3a8a; }
    .metric-label { font-size: 12px; color: #64748b; }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main function to run the Streamlit app."""
    st.title("Cryptocurrency Price Prediction & Signals")
    st.markdown("Load data, train MLP or LSTM models, and generate predictions with trading signals.")
    
    tab1, tab2, tab3 = st.tabs(["Data Loading", "Model Training", "Predictions & Signals"])
    
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = {}
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    
    # Tab 1: Data Loading
    with tab1:
        st.header("Load Data")
        data_source = st.radio("Data Source", ["Predefined Dataset", "Upload CSV"])
        
        if data_source == "Predefined Dataset":
            selected_dataset = st.selectbox("Select Dataset", list(PREDEFINED_DATASETS.keys()))
            coin_name = DATASET_SYMBOLS[selected_dataset]
            dataset_path = PREDEFINED_DATASETS[selected_dataset]
            window_size = st.slider("Window Size (Days)", 1, 10, 1, key="window_predefined")
            
            if st.button("Load Data"):
                with st.spinner("Loading data..."):
                    try:
                        predictor = CryptoPredictor(coin_name, dataset_path)
                        df = predictor.load_data()
                        X, y, dates, actual_close = predictor.prepare_features(window_size=window_size)
                        X_train, X_test, y_train, y_test = predictor.split_data(X, y)
                        
                        st.session_state.predictor = predictor
                        st.session_state.data_loaded = True
                        
                        st.subheader("Dataset Preview")
                        st.dataframe(df.head())
                        st.subheader("Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Records", df.shape[0])
                        col2.metric("Date Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
                        col3.metric("Min Price", f"${df['Close'].min():.2f}")
                        col4.metric("Max Price", f"${df['Close'].max():.2f}")
                        
                        st.subheader("Price History")
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.plot(df['Date'], df['Close'])
                        ax.set_title(f"{coin_name} Price History")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Price")
                        st.pyplot(fig)
                        
                        buf = BytesIO()
                        fig.savefig(buf, format="png")
                        st.download_button("Download Plot", buf, f"{coin_name}_history.png", "image/png")
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            coin_name = st.text_input("Coin Symbol", "BTC")
            window_size = st.slider("Window Size (Days)", 1, 10, 1, key="window_custom")
            
            if uploaded_file and st.button("Load Data"):
                with st.spinner("Loading data..."):
                    try:
                        temp_file_path = f"temp_{coin_name}_data.csv"
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        predictor = CryptoPredictor(coin_name, temp_file_path)
                        df = predictor.load_data()
                        X, y, dates, actual_close = predictor.prepare_features(window_size=window_size)
                        X_train, X_test, y_train, y_test = predictor.split_data(X, y)
                        
                        st.session_state.predictor = predictor
                        st.session_state.data_loaded = True
                        
                        st.subheader("Dataset Preview")
                        st.dataframe(df.head())
                        st.subheader("Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Records", df.shape[0])
                        col2.metric("Date Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
                        col3.metric("Min Price", f"${df['Close'].min():.2f}")
                        col4.metric("Max Price", f"${df['Close'].max():.2f}")
                        
                        st.subheader("Price History")
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.plot(df['Date'], df['Close'])
                        ax.set_title(f"{coin_name} Price History")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Price")
                        st.pyplot(fig)
                        
                        buf = BytesIO()
                        fig.savefig(buf, format="png")
                        st.download_button("Download Plot", buf, f"{coin_name}_history.png", "image/png")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    # Tab 2: Model Training
    with tab2:
        st.header("Train Models")
        if not st.session_state.data_loaded:
            st.warning("Load data first in the 'Data Loading' tab.")
        else:
            predictor = st.session_state.predictor
            st.markdown("**Note**: Both MLP and LSTM use the Adam optimizer to minimize prediction errors.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("MLP")
                if st.button("Train MLP"):
                    with st.spinner("Training MLP..."):
                        try:
                            predictor.train_mlp(hyperparameter_tuning=False)
                            predictions, rmse, mape = predictor.predict_and_evaluate('mlp')
                            st.session_state.models_trained['mlp'] = True
                            st.markdown(f"**RMSE**: {rmse:.2f} (average error of ${rmse:.2f} per prediction)<br>**MAPE**: {mape:.2f}% (average error of {mape:.2f}% of the price)", unsafe_allow_html=True)
                            st.success("MLP trained!")
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            with col2:
                st.subheader("LSTM")
                if st.button("Train LSTM"):
                    with st.spinner("Training LSTM..."):
                        try:
                            predictor.train_lstm(hyperparameter_tuning=False)
                            predictions, rmse, mape = predictor.predict_and_evaluate('lstm')
                            st.session_state.models_trained['lstm'] = True
                            st.markdown(f"**RMSE**: {rmse:.2f} (average error of ${rmse:.2f} per prediction)<br>**MAPE**: {mape:.2f}% (average error of {mape:.2f}% of the price)", unsafe_allow_html=True)
                            st.success("LSTM trained!")
                            st.image(f"outputs/{predictor.coin}/lstm_training_history.png", caption="LSTM Training Loss")
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            st.subheader("Model Summary")
            if any(st.session_state.models_trained.values()):
                st.markdown("**MLP Model**: The Multi-Layer Perceptron (MLP) predicts the next day's closing price of the cryptocurrency based on historical data, such as the previous day's open, high, low, close prices, and trading volume. It uses a neural network with two hidden layers (100 and 50 neurons) and the Adam optimizer to learn patterns in the data. The model's performance is measured by RMSE (e.g., 3216.53, meaning predictions are off by about $3216 on average) and MAPE (e.g., 5.99%, meaning predictions are off by about 6% of the actual price). These metrics indicate how close the predicted prices are to the actual prices, with lower values being better.")
                if 'lstm' in st.session_state.models_trained:
                    st.markdown("**LSTM Model**: The Long Short-Term Memory (LSTM) model also predicts the next day's closing price but is designed to capture time-series patterns in the data. It uses two LSTM layers (64 and 32 units) with dropout to prevent overfitting, optimized by Adam. Its performance is similarly measured by RMSE and MAPE. For example, an RMSE of 3216.53 and MAPE of 5.99% suggest that predictions are reasonably accurate for volatile cryptocurrencies like Bitcoin, though less so for lower-priced coins like BNB due to the dollar-based RMSE.")
    
    # Tab 3: Predictions & Signals
    with tab3:
        st.header("Predictions & Signals")
        if not st.session_state.data_loaded or not any(st.session_state.models_trained.values()):
            st.warning("Load data and train at least one model first.")
        else:
            predictor = st.session_state.predictor
            trained_models = [m for m in st.session_state.models_trained if st.session_state.models_trained[m]]
            selected_model = st.selectbox("Select Model", trained_models, format_func=lambda x: {'mlp': 'MLP', 'lstm': 'LSTM'}[x])
            threshold = st.slider("Signal Threshold (%)", 0.0, 5.0, 1.0, 0.1)
            
            if st.button("Generate Predictions"):
                with st.spinner("Generating predictions..."):
                    try:
                        predictions, rmse, mape = predictor.predict_and_evaluate(selected_model)
                        signals_df = predictor.generate_signals(selected_model, threshold/100)
                        
                        st.subheader("Performance")
                        col1, col2 = st.columns(2)
                        col1.markdown(f"**RMSE**: {rmse:.2f} (average error of ${rmse:.2f} per prediction)")
                        col2.markdown(f"**MAPE**: {mape:.2f}% (average error of {mape:.2f}% of the price)")
                        
                        st.subheader("Prediction Table")
                        st.dataframe(signals_df[['Date', 'Actual', 'Predicted', 'Signal', 'Confidence']].style.format({
                            'Actual': '{:.2f}',
                            'Predicted': '{:.2f}',
                            'Confidence': '{:.2%}'
                        }))
                        
                        st.subheader("Price Plot")
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.plot(signals_df['Date'], signals_df['Actual'], label='Actual', color='blue')
                        ax.plot(signals_df['Date'], signals_df['Predicted'], label='Predicted', color='green', linestyle='--')
                        buy_signals = signals_df[signals_df['Signal'] == 'BUY']
                        sell_signals = signals_df[signals_df['Signal'] == 'SELL']
                        if not buy_signals.empty:
                            ax.scatter(buy_signals['Date'], buy_signals['Close'], color='green', marker='^', s=100, label='Buy')
                        if not sell_signals.empty:
                            ax.scatter(sell_signals['Date'], sell_signals['Close'], color='red', marker='v', s=100, label='Sell')
                        ax.set_title(f"{predictor.coin} Predictions")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Price")
                        ax.legend()
                        st.pyplot(fig)
                        
                        buf = BytesIO()
                        signals_df.to_csv(buf, index=False)
                        st.download_button("Download Predictions", buf, f"{predictor.coin}_predictions.csv", "text/csv")
                        
                      
                        if st.button("Export Performance"):
                            comparison = predictor.export_performance()
                            st.dataframe(comparison)
                            buf = BytesIO()
                            comparison.to_csv(buf, index=False)
                            st.download_button("Download Performance", buf, f"{predictor.coin}_performance.csv", "text/csv")
                    except Exception as e:
                        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()