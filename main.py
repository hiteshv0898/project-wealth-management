import numpy as np
import yfinance as yf
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.models.hyperparameter_tuner import run_hyperparameter_tuning, build_model
from src.visualization.plotter import plot_predictions, plot_loss_curve, plot_residuals
import logging
from src.utils.anomaly_detection import detect_dbscan_anomalies
import talib
import optuna

# Configure Logging
logging.basicConfig(level=logging.INFO, format='ℹ️ %(asctime)s - ℹ️ %(levelname)s - %(message)s')

def fetch_data(ticker, start_date, end_date):
    logging.info(f"📦 Fetching data for {ticker} from {start_date} to {end_date}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        logging.info(f"✅ Successfully fetched {len(data)} records for {ticker}.")
        return data
    except Exception as e:
        logging.error(f"❗ Error fetching data for {ticker}: {e}")
        return None

def prepare_data_with_features(data, sequence_length=60):
    logging.info("🔎 Preparing data with additional features...")

    # Ensure 'Close' is passed as a 1D numpy array
    close_prices = data['Close'].values

    # Ensure it's a 1D array (reshape if necessary)
    close_prices = close_prices.reshape(-1)  # Explicitly reshape to 1D
    logging.info(f"Shape of close_prices: {close_prices.shape}")  # Check shape of close_prices

    # Calculate technical indicators (example: SMA, EMA, RSI)
    try:
        data['SMA'] = talib.SMA(close_prices, timeperiod=14)  # Simple Moving Average
        data['EMA'] = talib.EMA(close_prices, timeperiod=14)  # Exponential Moving Average
        data['RSI'] = talib.RSI(close_prices, timeperiod=14)  # Relative Strength Index
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}")

    # Handle NaN values by forward filling or dropping them (depending on the situation)
    data = data.dropna()

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['Close', 'SMA', 'EMA', 'RSI']])

    X, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i])
        y.append(data_scaled[i, 0])  # 'Close' price as the target

    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    logging.info(f"✅ Data scaled using MinMaxScaler. Shape: {X_train.shape}, {X_val.shape}")
    return X_train, y_train, X_val, y_val, scaler

def evaluate_model(y_true, y_pred):
    logging.info("📊 Evaluating Model Performance...")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    logging.info(f"✅ RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
    return rmse, mae, mape

def load_checkpoint(ticker):
    checkpoint_path = f'checkpoints/{ticker}_study.pkl'
    if os.path.exists(checkpoint_path):
        try:
            logging.info(f"🔄 Loading checkpoint from {checkpoint_path}...")
            study = joblib.load(checkpoint_path)
            if isinstance(study, optuna.study.Study):
                logging.info("✅ Study loaded successfully.")
                return study
            else:
                logging.error(f"❗ Checkpoint content type: {type(study)}")
                logging.error("❗ Checkpoint is not a valid Optuna study. Starting a new study.")
                return None
        except Exception as e:
            logging.error(f"❗ Error loading checkpoint: {e}", exc_info=True)
            return None
    else:
        logging.info(f"⚠️ No checkpoint found for {ticker}. Starting a new study.")
        return None

def save_checkpoint(ticker, study):
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = f'checkpoints/{ticker}_study.pkl'

    if isinstance(study, optuna.study.Study):
        joblib.dump(study, checkpoint_path)
        logging.info(f"💾 Checkpoint saved to {checkpoint_path}")
    else:
        logging.error("❗ Attempted to save an invalid study object. Ensure the study is returned correctly.")

def main():
    print("🚀 Starting Wealth Management Agent...")

    # Ticker details
    tickers = ['AAPL', 'TSLA', 'GOOGL', 'MSFT']
    start_date = '2023-01-01'
    end_date = '2024-01-01'

    all_data = {}
    for ticker in tickers:
        data = fetch_data(ticker, start_date, end_date)
        if data is not None:
            all_data[ticker] = data

    # Select the first ticker for training (you can loop through for all)
    ticker = tickers[0]
    if ticker not in all_data:
        logging.error("❗ No data available. Exiting...")
        return

    data = all_data[ticker]
    X_train, y_train, X_val, y_val, scaler = prepare_data_with_features(data)

    # Load or Perform Hyperparameter Tuning using Optuna
    print("🧪 Running hyperparameter tuning...")
    study = load_checkpoint(ticker)
    if study is None:
        study = run_hyperparameter_tuning(X_train, y_train, X_val, y_val)
        save_checkpoint(ticker, study)

    # Ensure study is valid before accessing best_params
    if isinstance(study, optuna.study.Study):
        try:
            best_params = study.best_params
            print(f"✅ Best Parameters Selected: {best_params}")
        except AttributeError as e:
            logging.error(f"❗ Error accessing best_params: {e}", exc_info=True)
            return
        except Exception as e:
            logging.error(f"❗ Unexpected error occurred: {e}", exc_info=True)
            return

    # Train Final Model
    model = build_model(best_params['num_units'], best_params['dropout_rate'],
                        best_params['learning_rate'], X_train.shape[1:])
    print("🚦 Training with best parameters...")
    history = model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'],
                        validation_data=(X_val, y_val), verbose=1)

    # Visualize Loss Curve
    plot_loss_curve(history)

    # Predict and Evaluate
    print("📈 Predicting stock prices...")
    predictions = model.predict(X_val)
    predictions_reshaped = predictions.reshape(-1, 1)  # Ensure predictions have the correct shape

    # Reshape y_val to match the number of features (4 in this case)
    y_val_reshaped = np.concatenate([y_val.reshape(-1, 1), np.zeros_like(y_val.reshape(-1, 1)), np.zeros_like(y_val.reshape(-1, 1)), np.zeros_like(y_val.reshape(-1, 1))], axis=1)

    # Now apply inverse transform
    actual_rescaled = scaler.inverse_transform(y_val_reshaped)[:, 0]  # Get 'Close' column only
    predictions_rescaled = scaler.inverse_transform(np.concatenate([predictions_reshaped, np.zeros_like(predictions_reshaped), np.zeros_like(predictions_reshaped), np.zeros_like(predictions_reshaped)], axis=1))[:, 0]

    # Evaluate Model
    evaluate_model(actual_rescaled, predictions_rescaled)

    # Perform Anomaly Detection
    anomalies = detect_dbscan_anomalies(actual_rescaled, predictions_rescaled)

    # Visualizations
    plot_predictions(actual_rescaled, predictions_rescaled, anomalies)
    plot_residuals(actual_rescaled, predictions_rescaled)

    print("🎉 Model training and evaluation complete!")

if __name__ == "__main__":
    main()