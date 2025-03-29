import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ModelEvaluator:
    @staticmethod
    def evaluate_model(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        print(f"✅ RMSE: {rmse:.4f}")
        print(f"✅ MAE: {mae:.4f}")
        print(f"✅ MAPE: {mape:.4f}%")
        return {"RMSE": rmse, "MAE": mae, "MAPE": mape}