# NeuroCrypto Predictor 🚀📈 
**NeuroCrypto Predictor** is a machine learning-powered web application designed to forecast cryptocurrency closing prices. 
Built with **Streamlit** and **Scikit-Learn**, it leverages a Multi-layer Perceptron (MLP) Neural Network to analyze daily market metrics and provide actionable price predictions.
## 🌟 Features * **Multi-Coin Support**: Predict closing prices for major cryptocurrencies including **Bitcoin (BTC)**,
#**Binance Coin (BNB)**, **Dogecoin (DOGE)**, and **Solana (SOL)**. * **AI-Powered Predictions**: Uses the `MLPRegressor` 
#(Neural Network) model to learn from historical data patterns. * **Interactive Web Interface**: 
#A user-friendly Streamlit dashboard that allows you to input daily metrics (Open, High, Low, Volume) and get instant results. 
#* **Real-time Data Processing**: Automatically scales user input using standard scaling techniques for accurate model inference.  ## 🛠️ Tech Stack * **Python**: Core programming language. * **Streamlit**: For the interactive web application interface. * **Scikit-Learn**: For implementing the MLP Regressor neural network model.
#* **Pandas**: For data manipulation and CSV handling. 
## 📂 Project Structure * `streamlitpredictionapp.py`: The main frontend application file. * `cryptomlpprediction.py`: The backend logic containing the machine learning model training and prediction functions. * `data/`: Contains historical datasets (`bitcoin.csv`, `BNB.csv`, `dogecoin.csv`, `solana.csv`) used for training the model. 
## 🚀 How to Run 
#1.  **Clone the repository**:     ```bash     git clone [https://github.com/yourusername/neurocrypto-predictor.git](https://github.com/yourusername/neurocrypto-predictor.git)     
#cd neurocrypto-predictor    
#``` 2.  **Install dependencies**: 
#```bash     pip install streamlit pandas scikit-learn  
#``` 3.  **Run the application**:     ```bash     streamlit run streamlitpredictionapp.py     ```
