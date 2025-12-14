"""
Signal Service Module
Research-backed: CNN-LSTM hybrid, feature engineering, batch/realtime inference
2024-2025 institutional best practices
"""
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMModel(nn.Module):
    def __init__(self, input_size: int = 50, num_filters: int = 32, hidden_size: int = 100, num_classes: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters*2),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=num_filters*2,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch, seq_len, num_filters*2)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        logits = self.fc(last)
        return logits

class SignalService:
    def __init__(self, input_size: int = 50, seq_len: int = 60):
        self.model = CNNLSTMModel(input_size=input_size)
        self.seq_len = seq_len
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.model(X_tensor)
            preds = logits.squeeze(-1).cpu().numpy()
        return preds
    def batch_predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)
    def confidence_intervals(self, X: np.ndarray, level: float = 0.95) -> np.ndarray:
        preds = self.predict(X)
        lower = np.percentile(preds, (1-level)*100)
        upper = np.percentile(preds, level*100)
        return lower, upper
