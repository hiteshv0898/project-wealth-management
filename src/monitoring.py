import numpy as np
from src.evaluation import ModelEvaluator
from src.fine_tuning import LSTMConfig

class ModelMonitor:
    def __init__(self, threshold_rmse=5.0):
        self.threshold_rmse = threshold_rmse
        self.config = LSTMConfig()

    def monitor_and_retrain(self, y_true, y_pred, model, data):
        """
        Monitor model performance and retrain if necessary.
        Args:
            y_true (np.ndarray): Actual values.
            y_pred (np.ndarray): Predicted values.
            model (tf.keras.Model): Current LSTM model.
            data (np.ndarray): Training data for retraining.
        """
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(y_true, y_pred)

        if results['RMSE'] > self.threshold_rmse:
            print("⚠️ RMSE exceeds threshold. Retraining triggered.")

            # Adjust for oversmoothing if detected
            if results['MAPE'] > 15:
                self.config.adjust_for_oversmoothing()

            # Add a layer if RMSE remains high
            if results['RMSE'] > 7.0:
                self.config.add_additional_layers()

            # Rebuild and retrain the model
            new_model = self.config.build_model(input_shape=(data.shape[1], 1))
            new_model.fit(data['X_train'], data['y_train'], epochs=self.config.epochs, batch_size=self.config.batch_size)

            print("✅ Model Retrained Successfully.")
            return new_model
        else:
            print("✅ Model performance is within acceptable range. No retraining required.")
            return model