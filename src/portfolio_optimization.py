import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def run_portfolio_optimization():
    print("📈 Performing Portfolio Optimization...")
    assets = ['AAPL', 'TSLA', 'GOOGL', 'MSFT']
    mean_returns = np.array([0.12, 0.18, 0.15, 0.10])
    cov_matrix = np.array([[0.005, -0.002, 0.004, 0.003],
                           [-0.002, 0.007, 0.002, 0.001],
                           [0.004, 0.002, 0.006, 0.002],
                           [0.003, 0.001, 0.002, 0.005]])

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(assets)))
    initial_weights = np.array([0.25, 0.25, 0.25, 0.25])

    result = minimize(portfolio_volatility, initial_weights, bounds=bounds, constraints=constraints)
    optimal_weights = result.x

    print("✅ Optimal Portfolio Weights:")
    for asset, weight in zip(assets, optimal_weights):
        print(f"{asset}: {weight:.2%}")