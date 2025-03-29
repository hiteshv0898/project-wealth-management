from src.evaluation import ModelEvaluator
from src.alerts import send_email_alert

class ModelMonitor:
    def __init__(self, threshold_rmse=5.0):
        self.threshold_rmse = threshold_rmse

    def monitor_and_retrain(self, y_true, y_pred, model, data):
        evaluator = ModelEvaluator()
        evaluation_results = evaluator.evaluate_model(y_true, y_pred)

        if evaluation_results['RMSE'] > self.threshold_rmse:
            print("⚠️ Model performance degraded. Initiating retraining...")
            send_email_alert("Model Degradation Alert", "RMSE exceeded threshold. Retraining initiated.")
            model.fit(data['X_train'], data['y_train'], epochs=50, batch_size=64)
            print("✅ Model retrained successfully.")
        else:
            print("✅ Model performance is within acceptable range. No retraining required.")