"""
Smoke test for Signal Service module
"""
import numpy as np
from signal_service import SignalService

def test_signal_service_smoke():
    # Generate synthetic data: 100 samples, 60 timesteps, 50 features
    X = np.random.rand(100, 60, 50)
    service = SignalService(input_size=50, seq_len=60)
    preds = service.predict(X)
    assert preds.shape[0] == 100, "Prediction shape mismatch"
    lower, upper = service.confidence_intervals(X)
    print("Signal predictions:", preds)
    print(f"Confidence interval (95%): {lower:.4f} to {upper:.4f}")
    print("Signal Service smoke test passed!")

if __name__ == "__main__":
    test_signal_service_smoke()
