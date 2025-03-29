import time
import schedule

class MonitoringAgent:
    def __init__(self, portfolio, stop_loss_percent=5, profit_target_percent=10):
        """
        Initialize Monitoring Agent.
        """
        self.portfolio = portfolio
        self.stop_loss_percent = stop_loss_percent
        self.profit_target_percent = profit_target_percent

        # Check if 'Current Price' column exists
        if 'Current Price' not in self.portfolio.columns:
            raise ValueError("❌ 'Current Price' column missing in portfolio data. Please add it before monitoring.")

        self.initial_prices = {row['Asset']: row['Current Price'] for _, row in self.portfolio.iterrows()}

    def check_prices(self):
        """
        Check for price anomalies based on stop-loss and profit target thresholds.
        """
        print("🔎 Checking prices for anomalies...")
        for _, row in self.portfolio.iterrows():
            asset = row['Asset']
            current_price = row['Current Price']
            initial_price = self.initial_prices[asset]
            price_change = ((current_price - initial_price) / initial_price) * 100

            print(f"{asset} is stable at ₹{current_price:.2f} with a {price_change:.2f}% change.")

            stop_loss_price = initial_price * (1 - self.stop_loss_percent / 100)
            profit_target_price = initial_price * (1 + self.profit_target_percent / 100)

            if current_price <= stop_loss_price:
                print(f"🚨 ALERT: {asset} hit the stop-loss price of ₹{stop_loss_price:.2f}. Consider selling.")
            elif current_price >= profit_target_price:
                print(f"🎉 ALERT: {asset} reached the profit target price of ₹{profit_target_price:.2f}. Consider taking profits.")
            else:
                print(f"INFO: {current_price} is within the safe range. Holding position.")

    def start_monitoring(self, run_time_minutes=1):
        """
        Start monitoring for the specified duration (in minutes).
        """
        print("✅ Initial prices fetched:", self.initial_prices)
        print("✅ Monitoring agent started for", run_time_minutes, "minute(s)...")

        # Schedule price check every minute
        schedule.every(1).minutes.do(self.check_prices)

        # Run for a limited time
        end_time = time.time() + (run_time_minutes * 60)
        while time.time() < end_time:
            schedule.run_pending()
            time.sleep(1)

        print("⏹ Monitoring session completed.")