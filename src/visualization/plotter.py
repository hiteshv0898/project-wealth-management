import matplotlib.pyplot as plt
from src.utils.logger import setup_logger
logger = setup_logger(__name__)

def plot_predictions(actual, predictions, anomalies):
    logger.info("📊 Plotting predictions and anomalies...")
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual', color='blue')
    plt.plot(predictions, label='Predicted', color='green')

    if anomalies is None or len(anomalies) == 0:
        logger.info("No anomalies detected. Skipping anomaly plotting.")
        anomalies = []

    if len(anomalies) > 0:
        logger.info(f"Plotting {len(anomalies)} anomalies.")
        plt.scatter(anomalies, actual[anomalies], color='orange', marker='o', label='Anomalies')

    plt.title('Actual vs Predicted Stock Prices with Anomalies')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def plot_stock_data(actual, predicted):
    print("📊 Plotting actual vs. predicted stock prices...")
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(actual, color='blue', label='Actual Stock Price')
        plt.plot(predicted, color='red', label='Predicted Stock Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()
        print("✅ Plot generated successfully.")
    except Exception as e:
        print(f"❌ Error while plotting: {e}")

def plot_loss_curve(history):
    print("📊 Plotting Loss Curve...")
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()
    print("✅ Loss Curve plotted successfully.")

def plot_residuals(actual, predicted):
    print("📊 Plotting Residuals...")
    residuals = actual - predicted
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(residuals)), residuals, color='red', alpha=0.5)
    plt.axhline(0, color='blue', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()
    print("✅ Residual Plot plotted successfully.")