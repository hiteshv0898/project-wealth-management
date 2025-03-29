import pandas as pd
from src.analysis import PortfolioAnalyzer
from src.utils import get_live_data

class WealthManagementAgent:
    def __init__(self, initial_capital, investment_allocation, risk_tolerance):
        self.initial_capital = initial_capital
        self.investment_allocation = investment_allocation
        self.risk_tolerance = risk_tolerance
        self.portfolio = self.create_portfolio()

    def create_portfolio(self):
        portfolio = []
        for asset, percentage in self.investment_allocation.items():
            amount = self.initial_capital * (percentage / 100)
            portfolio.append({'Asset': asset, 'Investment': amount})
        return pd.DataFrame(portfolio)

    def analyze_portfolio(self):
        analyzer = PortfolioAnalyzer(self.portfolio, self.risk_tolerance)
        analyzer.perform_analysis()

    def recommend_investments(self):
        print("Fetching market data and recommending investments...")
        print("Investment recommendations will be added in the next phase.")