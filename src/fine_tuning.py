import numpy as np
import tensorflow as tf

class LSTMConfig:
    def __init__(self):
        # Configurable hyperparameters
        self.units = 100
        self.dropout_rate = 0.2
        self.num_layers = 2
        self.learning_rate = 0.001
        self.batch_size = 64
        self.epochs = 50

    def adjust_for_oversmoothing(self):
        print("🔎 Adjusting to reduce oversmoothing...")
        if self.dropout_rate > 0.1:
            self.dropout_rate -= 0.05
        self.units += 50
        print(f"✅ New Units: {self.units}, New Dropout: {self.dropout_rate}")

    def add_additional_layers(self):
        print("➕ Adding an additional LSTM layer...")
        self.num_layers += 1
        print(f"✅ Total Layers: {self.num_layers}")

    def build_model(self, input_shape):
        """
        Build an LSTM model with the current configuration.
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=input_shape))

        for i in range(self.num_layers):
            model.add(tf.keras.layers.LSTM(self.units, return_sequences=(i < self.num_layers-1)))
            model.add(tf.keras.layers.Dropout(self.dropout_rate))

        model.add(tf.keras.layers.Dense(1))

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        print("✅ LSTM Model Built with the following config:")
        print(vars(self))

        return model