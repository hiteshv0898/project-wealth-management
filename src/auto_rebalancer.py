import pandas as pd
import numpy as np
from src.analysis import PortfolioAnalyzer

class AutoRebalancer:
    def __init__(self, portfolio_data, risk_tolerance='moderate'):
        """
        Initialize the AutoRebalancer with portfolio data and risk tolerance level.
        """
        self.portfolio_data = portfolio_data
        self.risk_tolerance = risk_tolerance
        self.analyzer = PortfolioAnalyzer(portfolio_data, risk_tolerance)

    def calculate_optimized_allocation(self):
        """
        Calculates the optimized portfolio allocation using Sharpe ratio.
        If all Sharpe ratios are negative, apply a risk-minimizing strategy.
        """
        print("🔎 Performing portfolio optimization using Sharpe ratio...")
        self.analyzer.perform_analysis()

        # Calculate Sharpe ratio
        sharpe_ratios = self.analyzer.calculate_sharpe_ratio(self.portfolio_data)

        # Check if all Sharpe ratios are negative
        if not sharpe_ratios or all(x['Sharpe Ratio'] <= 0 for x in sharpe_ratios):
            print("⚠️ All Sharpe ratios are negative. Switching to risk-minimizing strategy.")
            return self.risk_minimizing_allocation()

        # Perform allocation based on positive Sharpe ratios
        optimized_allocation = {}
        total_sharpe = sum(x['Sharpe Ratio'] for x in sharpe_ratios if x['Sharpe Ratio'] > 0)

        for data in sharpe_ratios:
            asset = data['Asset']
            sharpe_ratio = data['Sharpe Ratio']
            if sharpe_ratio > 0 and total_sharpe > 0:
                optimized_allocation[asset] = (sharpe_ratio / total_sharpe) * 100
            else:
                optimized_allocation[asset] = 0

        print("✅ Optimized Allocation (Based on Sharpe Ratio):", optimized_allocation)
        return optimized_allocation

    def risk_minimizing_allocation(self):
        """
        Applies a risk-minimizing strategy by assigning equal weights to all assets.
        """
        print("🔎 Applying Risk-Minimizing Strategy...")
        equal_weight = 100 / len(self.portfolio_data)
        return {row['Asset']: equal_weight for _, row in self.portfolio_data.iterrows()}

    def rebalance_portfolio(self):
        """
        Provides recommendations for portfolio rebalancing based on optimized allocation.
        """
        print("🔁 Rebalancing portfolio based on optimized allocation...")
        optimized_allocation = self.calculate_optimized_allocation()

        for asset, new_allocation in optimized_allocation.items():
            current_investment = self.portfolio_data.loc[self.portfolio_data['Asset'] == asset, 'Investment'].values[0]
            target_investment = new_allocation / 100 * self.portfolio_data['Investment'].sum()
            adjustment = target_investment - current_investment

            if adjustment > 0:
                print(f"💡 Buy {asset}: Adjust by ₹{adjustment:.2f}")
            elif adjustment < 0:
                print(f"⚠️ Sell {asset}: Adjust by ₹{abs(adjustment):.2f}")
            else:
                print(f"✅ {asset} is well-balanced.")

        print("🔎 Portfolio rebalancing completed.")