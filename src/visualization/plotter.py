import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(actual, predicted):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label="Actual", linestyle='-', marker='o')
    plt.plot(predicted, label="Predicted", linestyle='--', marker='x')
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Stock Price Prediction")
    plt.legend()
    plt.show()