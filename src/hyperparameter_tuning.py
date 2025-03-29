import optuna
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

class HyperparameterTuner:
    def __init__(self, data, lookback=60):
        """
        Initialize the HyperparameterTuner with data and a lookback window.
        """
        self.data = data
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def objective(self, trial):
        """
        Objective function for Bayesian Optimization using Optuna.
        """
        # Define search space for hyperparameters
        num_units = trial.suggest_int('num_units', 50, 200)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

        # Build LSTM model
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(self.lookback, 1)),
            tf.keras.layers.LSTM(num_units, return_sequences=True),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.LSTM(num_units),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')

        # Preprocess Data
        data = self.data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Train model
        model.fit(X, y, epochs=20, batch_size=batch_size, verbose=0)

        # Evaluate with RMSE
        predictions = model.predict(X)
        loss = np.sqrt(np.mean((self.scaler.inverse_transform(predictions) -
                                self.scaler.inverse_transform(y.reshape(-1, 1)))**2))

        return loss

    def tune_hyperparameters(self, n_trials=30):
        """
        Run Bayesian Optimization to find the best hyperparameters with Early Stopping and Time Limit.
        """
        print("🔎 Starting Hyperparameter Optimization using Bayesian Search...")

        # Apply Early Stopping with Median Pruner
        pruner = optuna.pruners.PatientPruner(
            optuna.pruners.MedianPruner(),
            patience=5
        )

        def callback(study, trial):
            # Stop if loss threshold is met
            if study.best_value and study.best_value < 4.0:
                print("✅ Early Stopping: Loss below threshold (4.0).")
                study.stop()

        # Run Optimization with Time Limit
        study = optuna.create_study(direction='minimize', pruner=pruner)
        study.optimize(self.objective, n_trials=n_trials, timeout=600, callbacks=[callback])

        print(f"✅ Best Hyperparameters: {study.best_params}")
        return study.best_params