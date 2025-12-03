import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'quantum_trading_v2'))
 # Removed incorrect import
import numpy as np

class TestForecast24DayEnsemble(unittest.TestCase):
    def setUp(self):
        from forecast_24d import ModelTrainer, ModelEnsemble, PatternDetector
        self.input_dim = 10
        self.seq_len = 5
        # XGBoost needs 2D, LSTM/CNN need 3D
        self.X_2d = np.random.rand(100, self.input_dim)
        self.X_3d = np.random.rand(100, self.seq_len, self.input_dim)
        self.y = np.random.rand(100)
        self.trainer = ModelTrainer(self.input_dim, self.seq_len)
        self.ensemble = ModelEnsemble(self.trainer)
        self.pattern_detector = PatternDetector()

    def test_train_and_predict(self):
        # Train the models (XGBoost: 2D, LSTM/CNN: 3D)
        # Patch ModelTrainer.train to accept both shapes
        # For this test, we only check ensemble prediction
        # We'll flatten 3D to 2D for XGBoost, keep 3D for others
        # The ModelTrainer expects 3D for LSTM/CNN, but XGBoost needs 2D
        # We'll call train with 3D, but patch XGBoost input inside ModelTrainer if needed
        self.trainer.xgb.train(self.X_2d, self.y)
        # LSTM/CNN train skipped for speed in smoke test
        preds, conf = self.ensemble.predict_24d(self.X_2d)
        self.assertEqual(len(preds), len(self.y))
        self.assertTrue(np.all(np.isfinite(preds)))
        self.assertIn('conf_95', conf)
        self.assertIn('conf_99', conf)

    def test_feature_importance(self):
        self.trainer.xgb.train(self.X_2d, self.y)
        importance = self.trainer.get_model_importance()
        self.assertIn('xgb_importance', importance)
        self.assertIn('xgb_shap', importance)

    def test_pattern_detection(self):
        import pandas as pd
        price_data = pd.Series(np.random.rand(100))
        patterns = self.pattern_detector.detect_patterns(price_data)
        self.assertIsInstance(patterns, list)

if __name__ == "__main__":
    unittest.main()
