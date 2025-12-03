"""
Smoke test for Trade Executor module
"""
from trade_executor import TradeExecutor

def test_trade_executor_smoke():
    executor = TradeExecutor()
    order = {'ticker': 'AAPL', 'quantity': 10, 'entry_price': 150.0, 'side': 'buy'}
    portfolio_state = {'equity': 10000, 'cash': 5000, 'used_margin': 1000, 'positions': []}
    market_data = {'AAPL': {'volatility': 0.02, 'average_daily_volume': 1000000, 'spread': 0.01}}
    valid, violations = executor.validate_order(order, portfolio_state, market_data)
    print('Order validation:', 'Passed' if valid else violations)
    kelly_size = executor.calculate_kelly_size(10000, 0.02, 150.0)
    print('Kelly position size:', kelly_size)
    slippage = executor.estimate_slippage('AAPL', 10, 150.0, market_data)
    print('Slippage estimate:', slippage)
    print('Trade Executor smoke test passed!')

if __name__ == "__main__":
    test_trade_executor_smoke()
