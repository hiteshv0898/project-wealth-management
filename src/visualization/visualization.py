from src.utils.logger import setup_logger
import matplotlib.pyplot as plt

logger = setup_logger(__name__)

def plot_predictions(data, predictions, anomalies):
    logger.info("📊 Generating predictions and anomaly plots.")
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Actual', color='blue')
    plt.plot(predictions, label='Predicted', color='red')
    plt.scatter(anomalies, data[anomalies], color='orange', marker='o', label='Anomalies')
    plt.legend()
    plt.title('Stock Price Prediction and Anomalies')
    plt.show()
    logger.info("✅ Prediction plot displayed successfully.")

def plot_loss_curve(history):
    logger.info("📊 Plotting training and validation loss curves.")
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.legend()
    plt.title('Training and Validation Loss Curve')
    plt.show()
    logger.info("✅ Loss curve displayed successfully.")