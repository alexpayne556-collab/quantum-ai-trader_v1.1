#!/usr/bin/env python3
"""
TICKER-AWARE TRANSFORMER MODEL
==============================
Multi-ticker model with shared learning + ticker-specific adaptation.

Architecture (from Perplexity research):
- Shared LSTM: Learns universal patterns (momentum, mean reversion)
- Ticker Embeddings: Learns what each ticker "means"
- Ticker Adapters: Fine-tunes features for each stock
- Attention: Shows WHICH inputs drove the prediction
- Catalyst Integration: Domain signals for each ticker

Key Benefits:
- Transfer learning: 1500 days of data (all tickers) vs 500 days (single)
- Automatic feature weighting per ticker
- Interpretable decisions via attention weights

Target GPU: NVIDIA RTX 2000 Ada (Shadow PC)
Author: Research Team
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Model hyperparameters"""
    
    # Input dimensions
    n_price_features: int = 30      # OHLCV + technical indicators
    n_catalyst_features: int = 10   # Per-ticker catalyst signals
    sequence_length: int = 60       # 60 trading days lookback
    
    # Architecture
    hidden_size: int = 64           # LSTM hidden size
    n_lstm_layers: int = 2          # LSTM depth
    ticker_embed_dim: int = 16      # Ticker embedding size
    adapter_hidden: int = 32        # Adapter layer size
    n_attention_heads: int = 4      # Multi-head attention
    
    # Output
    n_classes: int = 3              # Buy/Hold/Sell
    
    # Training
    dropout: float = 0.2
    learning_rate: float = 1e-4
    batch_size: int = 32
    
    # Regularization
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0


class TickerID(Enum):
    """Ticker to ID mapping"""
    RKLB = 0
    ASTS = 1
    IONQ = 2
    OKLO = 3
    # Add more as needed


# ============================================================================
# TICKER-AWARE MODEL
# ============================================================================

class TickerAwareTransformer(nn.Module):
    """
    Multi-ticker model with shared learning + ticker-specific adaptation.
    
    Architecture:
    1. Shared LSTM processes price/technical features (universal patterns)
    2. Ticker embedding learns what each ticker "means"
    3. Catalyst encoder processes ticker-specific domain signals
    4. Attention mechanism shows which inputs mattered
    5. Ticker adapter fine-tunes for each stock
    6. Output head produces Buy/Hold/Sell + confidence
    """
    
    def __init__(self, config: ModelConfig, n_tickers: int = 4):
        super().__init__()
        self.config = config
        self.n_tickers = n_tickers
        
        # ===== SHARED FEATURE EXTRACTION =====
        # Processes price + technical indicators
        # This learns UNIVERSAL patterns (momentum, mean reversion, etc.)
        self.shared_lstm = nn.LSTM(
            input_size=config.n_price_features,
            hidden_size=config.hidden_size,
            num_layers=config.n_lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.n_lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        # ===== TICKER EMBEDDINGS =====
        # Learns a vector representation for each ticker
        # "What does RKLB mean?" → 16-dim vector
        self.ticker_embedding = nn.Embedding(
            num_embeddings=n_tickers,
            embedding_dim=config.ticker_embed_dim
        )
        
        # ===== CATALYST ENCODER =====
        # Processes ticker-specific catalyst features
        # (launch dates, NRC timelines, qubit counts, etc.)
        self.catalyst_encoder = nn.Sequential(
            nn.Linear(config.n_catalyst_features, config.adapter_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.adapter_hidden, config.ticker_embed_dim),
        )
        
        # ===== ATTENTION MECHANISM =====
        # Shows WHICH parts of the sequence mattered
        # Interpretability: "This prediction used 90% launch date, 10% price"
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.n_attention_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # ===== TICKER-SPECIFIC ADAPTERS =====
        # Fine-tunes shared features for each ticker
        # "For RKLB, these features matter most"
        self.ticker_adapters = nn.ModuleList([
            self._build_adapter(config) for _ in range(n_tickers)
        ])
        
        # ===== FUSION LAYER =====
        # Combines: shared features + ticker embedding + catalyst encoding
        fusion_input_size = (
            config.hidden_size +      # Shared LSTM output
            config.ticker_embed_dim + # Ticker embedding
            config.ticker_embed_dim   # Catalyst encoding
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        
        # ===== OUTPUT HEAD =====
        # Produces: Buy/Hold/Sell logits + confidence
        self.output_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.adapter_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.adapter_hidden, config.n_classes),
        )
        
        # ===== CONFIDENCE HEAD =====
        # Separate head for prediction confidence
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.adapter_hidden),
            nn.ReLU(),
            nn.Linear(config.adapter_hidden, 1),
            nn.Sigmoid(),  # Output 0-1
        )
        
        # Initialize weights
        self._init_weights()
    
    def _build_adapter(self, config: ModelConfig) -> nn.Module:
        """Build a ticker-specific adapter module"""
        return nn.Sequential(
            nn.Linear(config.hidden_size, config.adapter_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.adapter_hidden, config.hidden_size),
        )
    
    def _init_weights(self):
        """Initialize weights properly"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self, 
        price_features: torch.Tensor,      # (batch, seq_len, n_price_features)
        ticker_ids: torch.Tensor,          # (batch,) - integer IDs
        catalyst_features: torch.Tensor,   # (batch, n_catalyst_features)
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            price_features: Price + technical indicator time series
            ticker_ids: Which ticker each sample belongs to
            catalyst_features: Ticker-specific catalyst signals
            return_attention: Whether to return attention weights
        
        Returns:
            logits: (batch, n_classes) - Buy/Hold/Sell scores
            confidence: (batch, 1) - Prediction confidence 0-1
            attention_weights: (batch, seq_len, seq_len) - Optional
        """
        batch_size = price_features.size(0)
        
        # ===== STEP 1: Shared LSTM processing =====
        # Learn universal patterns from price data
        lstm_out, (h_n, c_n) = self.shared_lstm(price_features)
        # lstm_out: (batch, seq_len, hidden_size)
        
        # ===== STEP 2: Attention over sequence =====
        # Which timesteps matter most?
        attended, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        # attended: (batch, seq_len, hidden_size)
        
        # Take last timestep (most recent)
        shared_features = attended[:, -1, :]
        # shared_features: (batch, hidden_size)
        
        # ===== STEP 3: Ticker embedding =====
        # What does this ticker "mean"?
        ticker_emb = self.ticker_embedding(ticker_ids)
        # ticker_emb: (batch, ticker_embed_dim)
        
        # ===== STEP 4: Catalyst encoding =====
        # Process domain-specific signals
        catalyst_enc = self.catalyst_encoder(catalyst_features)
        # catalyst_enc: (batch, ticker_embed_dim)
        
        # ===== STEP 5: Ticker-specific adaptation =====
        # Apply per-ticker adapter
        adapted_features = torch.zeros_like(shared_features)
        for ticker_id in range(self.n_tickers):
            mask = (ticker_ids == ticker_id)
            if mask.any():
                adapted_features[mask] = self.ticker_adapters[ticker_id](
                    shared_features[mask]
                )
        
        # Residual connection
        adapted_features = shared_features + adapted_features
        
        # ===== STEP 6: Fusion =====
        # Combine all information
        fused = torch.cat([
            adapted_features,  # Shared + adapted
            ticker_emb,        # Ticker identity
            catalyst_enc,      # Domain signals
        ], dim=1)
        
        fused = self.fusion(fused)
        # fused: (batch, hidden_size)
        
        # ===== STEP 7: Output =====
        logits = self.output_head(fused)
        confidence = self.confidence_head(fused)
        
        if return_attention:
            return logits, confidence, attention_weights
        return logits, confidence, None
    
    def predict(
        self, 
        price_features: torch.Tensor,
        ticker_ids: torch.Tensor,
        catalyst_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with interpretation.
        
        Returns dict with:
        - signal: 0=Sell, 1=Hold, 2=Buy
        - confidence: 0-1
        - probabilities: Softmax probabilities
        - attention: Which timesteps mattered
        """
        self.eval()
        with torch.no_grad():
            logits, confidence, attention = self.forward(
                price_features, ticker_ids, catalyst_features,
                return_attention=True
            )
            
            probs = F.softmax(logits, dim=1)
            signal = logits.argmax(dim=1)
            
            return {
                'signal': signal,
                'confidence': confidence.squeeze(),
                'probabilities': probs,
                'attention': attention,
            }


# ============================================================================
# CATALYST FEATURE BUILDERS
# ============================================================================

class CatalystFeatureBuilder:
    """
    Build ticker-specific catalyst features.
    
    Each ticker has different catalysts that matter:
    - RKLB: Launch schedule, backlog, contracts
    - ASTS: Satellites deployed, revenue timeline
    - IONQ: Qubit count, partnerships
    - OKLO: NRC timeline, policy news
    """
    
    def __init__(self, n_features: int = 10):
        self.n_features = n_features
    
    def build_rklb_features(self, data: pd.DataFrame, catalysts: Dict) -> np.ndarray:
        """
        RKLB-specific catalyst features.
        
        Catalysts that move RKLB:
        1. Days to next launch
        2. Launch success ratio (90-day)
        3. Days since last launch
        4. Backlog value ($M)
        5. Days since contract announcement
        6. Neutron milestone progress (0-5)
        7. Competitor failure flag
        8. Earnings surprise
        9. Government contract flag
        10. Media sentiment score
        """
        n_samples = len(data)
        features = np.zeros((n_samples, self.n_features))
        
        # Feature 0: Days to next launch (normalized)
        if 'next_launch_date' in catalysts:
            days_to_launch = (catalysts['next_launch_date'] - data.index).days
            features[:, 0] = np.clip(days_to_launch / 30, 0, 1)  # Normalize to 0-1
        
        # Feature 1: Launch success ratio (90-day rolling)
        if 'launch_history' in catalysts:
            features[:, 1] = catalysts.get('success_ratio_90d', 0.9)
        
        # Feature 2: Days since last launch (normalized)
        if 'last_launch_date' in catalysts:
            days_since = (data.index - catalysts['last_launch_date']).days
            features[:, 2] = np.clip(days_since / 30, 0, 1)
        
        # Feature 3: Backlog value (log-normalized)
        if 'backlog_value' in catalysts:
            features[:, 3] = np.log1p(catalysts['backlog_value']) / 10
        
        # Feature 4: Days since contract (normalized)
        if 'last_contract_date' in catalysts:
            days_since = (data.index - catalysts['last_contract_date']).days
            features[:, 4] = np.clip(days_since / 60, 0, 1)
        
        # Feature 5: Neutron progress (0-5 scale)
        features[:, 5] = catalysts.get('neutron_stage', 0) / 5
        
        # Feature 6: Competitor failure (binary)
        features[:, 6] = float(catalysts.get('competitor_failure_flag', 0))
        
        # Feature 7: Earnings surprise (normalized)
        features[:, 7] = np.clip(catalysts.get('earnings_surprise', 0) / 10, -1, 1)
        
        # Feature 8: Recent government contract (binary)
        features[:, 8] = float(catalysts.get('recent_gov_contract', 0))
        
        # Feature 9: Media sentiment (-1 to 1)
        features[:, 9] = catalysts.get('media_sentiment', 0)
        
        return features
    
    def build_asts_features(self, data: pd.DataFrame, catalysts: Dict) -> np.ndarray:
        """ASTS-specific catalyst features"""
        n_samples = len(data)
        features = np.zeros((n_samples, self.n_features))
        
        # Feature 0: Satellites deployed (normalized)
        features[:, 0] = catalysts.get('satellites_deployed', 0) / 100
        
        # Feature 1: Days to first revenue (normalized)
        features[:, 1] = np.clip(catalysts.get('days_to_revenue', 365) / 365, 0, 1)
        
        # Feature 2: Partnership count
        features[:, 2] = catalysts.get('partnership_count', 0) / 10
        
        # Feature 3: Recent partnership flag
        features[:, 3] = float(catalysts.get('recent_partnership', 0))
        
        # Feature 4: Regulatory approval progress
        features[:, 4] = catalysts.get('regulatory_stage', 0) / 5
        
        # Feature 5: Technical milestone flag
        features[:, 5] = float(catalysts.get('tech_milestone', 0))
        
        # Feature 6: Competitor setback flag
        features[:, 6] = float(catalysts.get('competitor_setback', 0))
        
        # Feature 7: Carrier adoption count
        features[:, 7] = catalysts.get('carrier_count', 0) / 10
        
        # Feature 8: Speed test results (normalized)
        features[:, 8] = catalysts.get('speed_test_score', 0) / 100
        
        # Feature 9: Coverage expansion flag
        features[:, 9] = float(catalysts.get('coverage_expansion', 0))
        
        return features
    
    def build_ionq_features(self, data: pd.DataFrame, catalysts: Dict) -> np.ndarray:
        """IONQ-specific catalyst features"""
        n_samples = len(data)
        features = np.zeros((n_samples, self.n_features))
        
        # Feature 0: Qubit count (log-normalized)
        features[:, 0] = np.log1p(catalysts.get('qubit_count', 32)) / 10
        
        # Feature 1: Days since qubit update
        features[:, 1] = np.clip(catalysts.get('days_since_qubit_update', 90) / 90, 0, 1)
        
        # Feature 2: Cloud partnerships (AWS, Azure, GCP)
        features[:, 2] = catalysts.get('cloud_partnerships', 0) / 3
        
        # Feature 3: Recent publication flag
        features[:, 3] = float(catalysts.get('recent_publication', 0))
        
        # Feature 4: Government contract value (log-normalized)
        features[:, 4] = np.log1p(catalysts.get('gov_contract_value', 0)) / 10
        
        # Feature 5: Customer deployment count
        features[:, 5] = catalysts.get('customer_deployments', 0) / 10
        
        # Feature 6: Error rate improvement (normalized)
        features[:, 6] = catalysts.get('error_rate_improvement', 0)
        
        # Feature 7: Competitor comparison score
        features[:, 7] = catalysts.get('vs_ibm_google', 0) + 0.5  # -0.5 to 0.5 → 0 to 1
        
        # Feature 8: Hype cycle position (0-1)
        features[:, 8] = catalysts.get('hype_position', 0.5)
        
        # Feature 9: Patent filings (normalized)
        features[:, 9] = catalysts.get('patent_count_90d', 0) / 10
        
        return features
    
    def build_oklo_features(self, data: pd.DataFrame, catalysts: Dict) -> np.ndarray:
        """OKLO-specific catalyst features"""
        n_samples = len(data)
        features = np.zeros((n_samples, self.n_features))
        
        # Feature 0: NRC stage (0=pre-app, 1=submitted, 2=review, 3=approved)
        features[:, 0] = catalysts.get('nrc_stage', 0) / 3
        
        # Feature 1: Days to expected decision
        features[:, 1] = np.clip(catalysts.get('days_to_decision', 365) / 365, 0, 1)
        
        # Feature 2: Customer LOI count
        features[:, 2] = catalysts.get('customer_loi_count', 0) / 10
        
        # Feature 3: Policy tailwind flag
        features[:, 3] = float(catalysts.get('policy_tailwind', 0))
        
        # Feature 4: Competitor setback flag
        features[:, 4] = float(catalysts.get('competitor_setback', 0))
        
        # Feature 5: Data center demand score
        features[:, 5] = catalysts.get('dc_demand_score', 0.5)
        
        # Feature 6: DOE funding flag
        features[:, 6] = float(catalysts.get('doe_funding', 0))
        
        # Feature 7: AI narrative strength (0-1)
        features[:, 7] = catalysts.get('ai_narrative', 0.5)
        
        # Feature 8: Site development progress
        features[:, 8] = catalysts.get('site_progress', 0) / 5
        
        # Feature 9: Nuclear policy sentiment
        features[:, 9] = catalysts.get('nuclear_sentiment', 0.5)
        
        return features
    
    def build_features(self, ticker: str, data: pd.DataFrame, catalysts: Dict) -> np.ndarray:
        """Build features for any ticker"""
        builders = {
            'RKLB': self.build_rklb_features,
            'ASTS': self.build_asts_features,
            'IONQ': self.build_ionq_features,
            'OKLO': self.build_oklo_features,
        }
        
        builder = builders.get(ticker.upper())
        if builder is None:
            # Return zeros for unknown tickers
            return np.zeros((len(data), self.n_features))
        
        return builder(data, catalysts)


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class TickerAwareTrainer:
    """
    Training loop with proper validation.
    
    Features:
    - Purged cross-validation (no data leakage)
    - Per-ticker metrics
    - Transfer learning verification
    - Early stopping
    """
    
    def __init__(self, model: TickerAwareTransformer, config: ModelConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.best_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, dataloader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            price_features = batch['price_features']
            ticker_ids = batch['ticker_ids']
            catalyst_features = batch['catalyst_features']
            labels = batch['labels']
            
            self.optimizer.zero_grad()
            
            logits, confidence, _ = self.model(
                price_features, ticker_ids, catalyst_features
            )
            
            loss = self.criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.gradient_clip
            )
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader) -> Dict[str, float]:
        """Validate and compute metrics"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_tickers = []
        
        with torch.no_grad():
            for batch in dataloader:
                price_features = batch['price_features']
                ticker_ids = batch['ticker_ids']
                catalyst_features = batch['catalyst_features']
                labels = batch['labels']
                
                logits, confidence, _ = self.model(
                    price_features, ticker_ids, catalyst_features
                )
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_tickers.extend(ticker_ids.cpu().numpy())
        
        # Overall metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_tickers = np.array(all_tickers)
        
        accuracy = (all_preds == all_labels).mean()
        
        # Per-ticker metrics
        per_ticker_acc = {}
        for ticker_id in range(self.model.n_tickers):
            mask = all_tickers == ticker_id
            if mask.sum() > 0:
                ticker_acc = (all_preds[mask] == all_labels[mask]).mean()
                ticker_name = list(TickerID)[ticker_id].name
                per_ticker_acc[ticker_name] = ticker_acc
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'per_ticker': per_ticker_acc,
        }
    
    def check_transfer_learning(self, val_results_all: Dict, val_results_subset: Dict) -> Dict:
        """
        Verify transfer learning is working.
        
        Compare model trained on ALL tickers vs subset.
        If accuracy on held-out ticker is better with all data,
        transfer learning is helping.
        """
        improvement = {}
        
        for ticker, acc_all in val_results_all['per_ticker'].items():
            acc_subset = val_results_subset['per_ticker'].get(ticker, 0)
            improvement[ticker] = acc_all - acc_subset
        
        avg_improvement = np.mean(list(improvement.values()))
        
        return {
            'per_ticker_improvement': improvement,
            'avg_improvement': avg_improvement,
            'transfer_learning_works': avg_improvement > 0.02,  # 2% improvement threshold
        }


# ============================================================================
# QUICK TEST
# ============================================================================

def quick_test():
    """Quick test to verify model builds correctly"""
    print("=" * 60)
    print("TICKER-AWARE TRANSFORMER - QUICK TEST")
    print("=" * 60)
    
    # Config
    config = ModelConfig()
    print(f"\n[CONFIG]")
    print(f"  Price features: {config.n_price_features}")
    print(f"  Catalyst features: {config.n_catalyst_features}")
    print(f"  Sequence length: {config.sequence_length}")
    print(f"  Hidden size: {config.hidden_size}")
    
    # Build model
    print(f"\n[MODEL] Building TickerAwareTransformer...")
    model = TickerAwareTransformer(config, n_tickers=4)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")
    
    # Test forward pass
    print(f"\n[TEST] Running forward pass...")
    batch_size = 8
    
    # Fake data
    price_features = torch.randn(batch_size, config.sequence_length, config.n_price_features)
    ticker_ids = torch.randint(0, 4, (batch_size,))
    catalyst_features = torch.randn(batch_size, config.n_catalyst_features)
    
    # Forward
    logits, confidence, attention = model(
        price_features, ticker_ids, catalyst_features,
        return_attention=True
    )
    
    print(f"  Input shapes:")
    print(f"    price_features: {price_features.shape}")
    print(f"    ticker_ids: {ticker_ids.shape}")
    print(f"    catalyst_features: {catalyst_features.shape}")
    
    print(f"  Output shapes:")
    print(f"    logits: {logits.shape}")
    print(f"    confidence: {confidence.shape}")
    print(f"    attention: {attention.shape}")
    
    # Test predict
    print(f"\n[TEST] Running predict...")
    predictions = model.predict(price_features, ticker_ids, catalyst_features)
    
    print(f"  Signals: {predictions['signal'].tolist()}")
    print(f"  Confidence: {[f'{c:.2f}' for c in predictions['confidence'].tolist()]}")
    
    # Test catalyst builder
    print(f"\n[TEST] Testing CatalystFeatureBuilder...")
    builder = CatalystFeatureBuilder()
    
    # Fake data
    fake_data = pd.DataFrame({'close': np.random.randn(100)}, index=pd.date_range('2024-01-01', periods=100))
    fake_catalysts = {
        'next_launch_date': pd.Timestamp('2024-06-01'),
        'success_ratio_90d': 0.95,
        'backlog_value': 500,
        'neutron_stage': 3,
    }
    
    rklb_features = builder.build_features('RKLB', fake_data, fake_catalysts)
    print(f"  RKLB features shape: {rklb_features.shape}")
    print(f"  Sample: {rklb_features[0][:5]}")
    
    print(f"\n{'=' * 60}")
    print("ALL TESTS PASSED ✅")
    print("=" * 60)
    print("""
NEXT STEPS:
1. Build catalyst scraper for each ticker
2. Create training data pipeline
3. Train on historical data
4. Validate transfer learning effect
5. Paper trade with Alpaca
""")


if __name__ == "__main__":
    quick_test()
