import threading
import time
import random
import matplotlib.pyplot as plt
from collections import deque

class LiveDataStreamer:
    def __init__(self):
        self.assets = ['AAPL', 'TSLA', 'GOOGL', 'MSFT']
        self.prices = {asset: random.uniform(150, 400) for asset in self.assets}
        self.data = {asset: deque(maxlen=100) for asset in self.assets}
        self.running = True

    def simulate_price_change(self, price):
        return price + random.uniform(-1, 1) * price * 0.005

    def update_prices(self):
        while self.running:
            for asset in self.assets:
                self.prices[asset] = self.simulate_price_change(self.prices[asset])
                self.data[asset].append(self.prices[asset])
            time.sleep(1)

    def start(self):
        print("🚀 Starting Live Data Stream...")
        threading.Thread(target=self.update_prices, daemon=True).start()
        for _ in range(30):
            self.plot_data()
            time.sleep(1)

    def plot_data(self):
        plt.clf()
        for asset, price_data in self.data.items():
            plt.plot(price_data, label=asset)
        plt.legend(loc='upper left')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Live Data Streaming - Simulated Stock Prices')
        plt.pause(0.1)

    def stop(self):
        self.running = False
        print("⏹️ Stopping Live Data Stream...")