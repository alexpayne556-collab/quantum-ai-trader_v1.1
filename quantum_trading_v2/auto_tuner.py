"""
AUTO-TUNER MODULE
=================
Dynamically adjusts model hyperparameters based on training results.
"""
import numpy as np
import copy

class AutoTuner:
    """
    Auto-tunes QuantumForecastConfig hyperparameters based on validation loss and accuracy.
    """
    def __init__(self, config, min_lr=1e-5, max_lr=1e-3, min_batch=8, max_batch=64):
        self.config = copy.deepcopy(config)
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.history = []

    def update(self, val_loss, val_acc):
        """
        Adjust config based on validation metrics.
        """
        self.history.append({'val_loss': val_loss, 'val_acc': val_acc})
        # Example logic: if loss not improving, reduce lr, increase batch size
        if len(self.history) < 2:
            return self.config
        prev = self.history[-2]
        curr = self.history[-1]
        # If loss increased, reduce learning rate
        if curr['val_loss'] > prev['val_loss']:
            self.config.learning_rate = max(self.min_lr, self.config.learning_rate * 0.7)
        # If accuracy improved, try increasing batch size
        if curr['val_acc'] > prev['val_acc']:
            self.config.batch_size = min(self.max_batch, self.config.batch_size + 8)
        # If accuracy drops, decrease batch size
        if curr['val_acc'] < prev['val_acc']:
            self.config.batch_size = max(self.min_batch, self.config.batch_size - 8)
        return self.config

    def get_config(self):
        return copy.deepcopy(self.config)
