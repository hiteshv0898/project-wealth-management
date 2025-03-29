import unittest
from src.agent import WealthManagementAgent

class TestWealthManagementAgent(unittest.TestCase):
    def test_portfolio_creation(self):
        allocation = {'AAPL': 50, 'TSLA': 50}
        agent = WealthManagementAgent(100000, allocation, risk_tolerance='moderate')
        self.assertEqual(len(agent.portfolio), 2)

if __name__ == "__main__":
    unittest.main()