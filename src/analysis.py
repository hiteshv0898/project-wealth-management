import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from src.utils import get_live_data

class PortfolioAnalyzer:
    def __init__(self, portfolio, risk_tolerance):
        self.portfolio = portfolio
        self.risk_tolerance = risk_tolerance

    def perform_analysis(self):
        results = []
        for index, row in self.portfolio.iterrows():
            data = get_live_data(row['Asset'])
            if data:
                current_price = data.history(period='1d')['Close'].iloc[-1]
                results.append({'Asset': row['Asset'], 'Investment': row['Investment'], 'Current Price': current_price})
        df = pd.DataFrame(results)
        print("Portfolio Analysis Results:")
        print(df)

        self.calculate_sharpe_ratio(df)
        self.calculate_risk_adjusted_return(df)
        self.optimize_allocation(df)
        self.recommend_investments(df)

    def calculate_sharpe_ratio(self, df):
        print("Calculating Sharpe Ratio...")
        returns = []
        for asset in df['Asset']:
            data = get_live_data(asset)
            if data:
                try:
                    historical_data = pd.DataFrame(data.history(period='1y')['Close'].pct_change().dropna())
                    mean_return = historical_data.mean().iloc[0]
                    std_dev = historical_data.std().iloc[0]
                    sharpe_ratio = (mean_return - 0.04) / std_dev
                    returns.append({'Asset': asset, 'Sharpe Ratio': sharpe_ratio})
                except Exception as e:
                    print(f"Error processing {asset}: {e}")
        print(pd.DataFrame(returns))

    def calculate_risk_adjusted_return(self, df):
        print("Calculating Risk-Adjusted Return...")
        risk_adjusted_results = []
        for asset in df['Asset']:
            data = get_live_data(asset)
            if data:
                try:
                    historical_data = pd.DataFrame(data.history(period='1y')['Close'].pct_change().dropna())
                    mean_return = historical_data.mean().iloc[0]
                    std_dev = historical_data.std().iloc[0]
                    risk_adjusted_return = mean_return / std_dev
                    risk_adjusted_results.append({'Asset': asset, 'Risk-Adjusted Return': risk_adjusted_return})
                except Exception as e:
                    print(f"Error processing {asset}: {e}")
        print(pd.DataFrame(risk_adjusted_results))

    def optimize_allocation(self, df):
        print("Optimizing portfolio using Monte Carlo Simulation...")
        num_assets = len(df)
        num_simulations = 10000

        returns = []
        risks = []
        weights_list = []

        prices = []
        for asset in df['Asset']:
            data = get_live_data(asset)
            if data:
                try:
                    historical_data = data.history(period='1y')['Close'].dropna()
                    prices.append(historical_data)
                except Exception as e:
                    print(f"Error fetching historical data for {asset}: {e}")

        prices_df = pd.concat(prices, axis=1).dropna()
        returns_df = prices_df.pct_change().dropna()

        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()

        for _ in range(num_simulations):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            portfolio_return = np.sum(weights * mean_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            returns.append(portfolio_return)
            risks.append(portfolio_risk)
            weights_list.append(weights)

        sharpe_ratios = np.array(returns) / np.array(risks)
        max_sharpe_idx = np.argmax(sharpe_ratios)

        print("Optimal Portfolio Allocation:")
        print({df['Asset'][i]: round(weights_list[max_sharpe_idx][i] * 100, 2) for i in range(num_assets)})

    def recommend_investments(self, df):
        print("Generating Investment Recommendations using Machine Learning...")
        predictions = []
        for asset in df['Asset']:
            data = get_live_data(asset)
            if data:
                try:
                    historical_data = data.history(period='5y')['Close'].dropna()
                    X = np.arange(len(historical_data)).reshape(-1, 1)
                    y = historical_data.values

                    model = LinearRegression()
                    model.fit(X, y)
                    predicted_price = model.predict([[len(historical_data) + 30]])[0]

                    current_price = historical_data.iloc[-1]
                    recommendation = "Buy" if predicted_price > current_price else "Sell"

                    predictions.append({'Asset': asset, 'Predicted Price': predicted_price, 'Current Price': current_price, 'Recommendation': recommendation})
                except Exception as e:
                    print(f"Error predicting for {asset}: {e}")
        print(pd.DataFrame(predictions))