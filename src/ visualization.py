import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_predictions():
    print("📊 Visualizing Actual vs Predicted Stock Prices...")
    np.random.seed(42)
    time_steps = np.arange(0, 100, 1)
    actual_prices = np.sin(time_steps / 10) + np.random.normal(0, 0.1, len(time_steps))
    predicted_prices = np.sin(time_steps / 10) + np.random.normal(0, 0.2, len(time_steps))

    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, actual_prices, label='Actual Prices', linewidth=2)
    plt.plot(time_steps, predicted_prices, label='Predicted Prices', linestyle='--', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Stock Prices')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss_curve():
    print("📊 Visualizing Training vs Validation Loss...")
    epochs = np.arange(1, 51)
    train_loss = np.random.uniform(0.1, 0.3, 50)
    val_loss = np.random.uniform(0.1, 0.4, 50)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, label='Validation Loss', linewidth=2, linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_portfolio_allocation():
    print("📊 Visualizing Portfolio Allocation...")
    portfolio_data = pd.DataFrame({'Asset': ['AAPL', 'TSLA', 'GOOGL', 'MSFT'],
                                   'Investment': [30000, 30000, 20000, 20000]})
    plt.figure(figsize=(8, 8))
    plt.pie(portfolio_data['Investment'], labels=portfolio_data['Asset'], autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Portfolio Allocation')
    plt.show()