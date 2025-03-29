import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

class LSTMPredictor:
    def __init__(self, data, lookback=60):
        self.data = data
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.X_train = None
        self.y_train = None

    def preprocess_data(self):
        # Scale the data
        scaled_data = self.scaler.fit_transform(self.data['Close'].values.reshape(-1, 1))

        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Store training data
        self.X_train, self.y_train = X, y

    def build_model(self, num_units=100, dropout_rate=0.2, learning_rate=0.001):
        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(num_units, return_sequences=True, input_shape=(self.lookback, 1)),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.LSTM(num_units, return_sequences=False),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

    def train_model(self, epochs=50, batch_size=64):
        if self.X_train is None or self.y_train is None:
            self.preprocess_data()

        self.build_model()
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)

    def predict(self):
        if self.model is None:
            raise ValueError("Model is not built or trained.")

        predictions = self.model.predict(self.X_train)
        return self.scaler.inverse_transform(predictions)