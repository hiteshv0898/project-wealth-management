from src.visualization import plot_predictions, plot_loss_curve, plot_portfolio_allocation
from src.anomaly_detection import detect_anomalies
from src.portfolio_optimization import run_portfolio_optimization
from src.live_streaming import LiveDataStreamer
from src.evaluation import ModelEvaluator
from src.model_monitor import ModelMonitor
from src.alerts import send_email_alert

def main():
    print("🚀 Starting Wealth Management Agent")

    # Visualizations
    plot_predictions()
    plot_loss_curve()
    plot_portfolio_allocation()

    # Anomaly Detection
    detect_anomalies()

    # Portfolio Optimization
    run_portfolio_optimization()

    # Model Monitoring (Dummy Data)
    y_true = [220, 265, 155, 380]
    y_pred = [218, 260, 157, 375]

    print("🛡️ Monitoring Model Performance...")
    monitor = ModelMonitor(threshold_rmse=5.0)
    monitor.monitor_and_retrain(y_true, y_pred, model=None, data=None)

    # Live Data Streaming
    streamer = LiveDataStreamer()
    streamer.start()

if __name__ == "__main__":
    main()