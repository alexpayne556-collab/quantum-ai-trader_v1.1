"""
Smoke test for Portfolio Manager module
"""
import numpy as np
from portfolio_manager_optimal import PortfolioManagerService

def test_portfolio_manager_smoke():
    service = PortfolioManagerService()
    returns = np.random.randn(252, 50) * 0.02
    weights = service.optimize_hrp(returns)
    assert np.isclose(sum(weights.values()), 1.0), "Weights do not sum to 1.0"
    risk_metrics = service.calculate_risk_metrics(returns[:, 0])
    print("HRP Weights:", weights)
    print("Risk Metrics:", risk_metrics)
    print("Portfolio Manager smoke test passed!")

if __name__ == "__main__":
    test_portfolio_manager_smoke()
