import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

def train_lstm_model(ticker):
    try:
        data = yf.Ticker(ticker).history(period='5y')['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        x_train, y_train = [], []
        for i in range(60, len(data_scaled)):
            x_train.append(data_scaled[i-60:i])
            y_train.append(data_scaled[i])

        x_train, y_train = np.array(x_train), np.array(y_train)

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=10, batch_size=32)

        if not os.path.exists('models'):
            os.makedirs('models')

        model.save(f'models/{ticker}_lstm_model.h5')
        print(f"Model for {ticker} saved successfully.")
    except Exception as e:
        print(f"Error training LSTM model for {ticker}: {e}")