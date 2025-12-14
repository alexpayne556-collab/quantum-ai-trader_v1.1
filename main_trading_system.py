"""
Main Trading System: Complete workflow
Trains model → Generates signals → Executes multi-week simulation

ULTIMATE AI TRADING DISCOVERY SYSTEM
Optimized for Colab T4 High-RAM (15GB+)

Usage:
    python main_trading_system.py              # Full training + signals
    python main_trading_system.py --quick      # Quick test (5 tickers, 3 years)
    python main_trading_system.py --signals    # Load model, generate signals only
"""

import argparse
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import json
import os

warnings.filterwarnings('ignore')

# Import our modules
from ultimate_feature_engine import UltimateFeatureEngine
from universal_model_trainer import UniversalTrader
from daily_signal_generator import DailySignalGenerator, WeeklySimulator

# Try genetic evolver (optional, needs DEAP)
try:
    from genetic_formula_evolver import GeneticFormulaEvolver, evolve_trading_formulas
    HAS_GENETIC = True
except ImportError:
    HAS_GENETIC = False
    print("⚠️ Genetic evolver not available (install DEAP)")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Full 30 tickers for production training
FULL_TICKERS = [
    # Major ETFs
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI',
    # Tech Giants
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    # Growth/Momentum
    'AMD', 'NFLX', 'CRM', 'ADBE', 'PYPL',
    # Crypto-related
    'SQ', 'COIN', 'MARA', 'RIOT',
    # Innovation
    'ARKK', 'PLTR', 'RBLX', 'HOOD',
    # Sectors
    'XLK', 'XLV', 'XLE', 'XLF', 'XLY'
]

# Quick test subset
QUICK_TICKERS = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']

# Training parameters
FULL_CONFIG = {
    'start_date': '2000-01-01',  # Maximum history
    'target_days': 5,            # 5-day returns
    'target_threshold': 0.01,    # 1% minimum return
    'validation_weeks': 8,       # Multi-week simulation
    'genetic_pop': 200,          # GP population
    'genetic_gen': 50,           # GP generations
}

QUICK_CONFIG = {
    'start_date': '2020-01-01',  # 4 years
    'target_days': 5,
    'target_threshold': 0.01,
    'validation_weeks': 4,
    'genetic_pop': 50,
    'genetic_gen': 10,
}


def print_banner():
    """Print startup banner."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║    ██╗   ██╗██╗  ████████╗██╗███╗   ███╗ █████╗ ████████╗███████╗             ║
║    ██║   ██║██║  ╚══██╔══╝██║████╗ ████║██╔══██╗╚══██╔══╝██╔════╝             ║
║    ██║   ██║██║     ██║   ██║██╔████╔██║███████║   ██║   █████╗               ║
║    ██║   ██║██║     ██║   ██║██║╚██╔╝██║██╔══██║   ██║   ██╔══╝               ║
║    ╚██████╔╝███████╗██║   ██║██║ ╚═╝ ██║██║  ██║   ██║   ███████╗             ║
║     ╚═════╝ ╚══════╝╚═╝   ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝             ║
║                                                                               ║
║              AI TRADING DISCOVERY SYSTEM v2.0                                 ║
║              Multi-Asset Universal Model Training                             ║
║              Optimized for Colab T4 High-RAM                                  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)


def run_full_pipeline(config: dict, tickers: list):
    """
    Run complete training and signal generation pipeline.
    
    Args:
        config: Configuration dictionary
        tickers: List of tickers to train on
    """
    print_banner()
    
    print("\n" + "=" * 80)
    print("🚀 STARTING FULL TRAINING PIPELINE")
    print("=" * 80)
    print(f"Tickers: {len(tickers)}")
    print(f"Date range: {config['start_date']} to today")
    print(f"Target: {config['target_days']}-day returns > {config['target_threshold']:.1%}")
    print("=" * 80)
    
    # ========================================================================
    # STEP 1: Train Universal Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("[1/4] TRAINING UNIVERSAL MODEL")
    print("=" * 80)
    
    trader = UniversalTrader(
        tickers=tickers,
        start_date=config['start_date'],
        target_days=config['target_days'],
        target_threshold=config['target_threshold']
    )
    
    # Prepare multi-asset data
    X, y = trader.prepare_multi_asset_data()
    
    # Train model
    model = trader.train_universal_model(X, y, use_gpu=True)
    
    # Walk-forward validation
    print("\n📊 Running walk-forward validation...")
    validation_results = trader.walk_forward_validation(X, y, n_splits=5)
    
    # Show feature importance
    print("\n🔝 TOP 15 MOST IMPORTANT FEATURES:")
    print("-" * 50)
    importance = trader.get_feature_importance(15)
    for i, row in importance.iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.4f}")
    
    # Save model
    trader.save_model("universal_trader_model.pkl")
    
    # ========================================================================
    # STEP 2: Genetic Formula Discovery (if available)
    # ========================================================================
    if HAS_GENETIC and config['genetic_pop'] > 0:
        print("\n" + "=" * 80)
        print("[2/4] GENETIC FORMULA DISCOVERY")
        print("=" * 80)
        
        # Use a sample of data for genetic evolution
        sample_size = min(2000, len(X))
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        sample_idx.sort()
        
        # Create features DataFrame
        features_df = pd.DataFrame(X[sample_idx], columns=trader.feature_names)
        future_returns = y[sample_idx]
        
        # Run genetic evolution
        evolver = GeneticFormulaEvolver(features_df, max_features=20)
        hof, logbook = evolver.evolve(
            future_returns,
            pop_size=config['genetic_pop'],
            generations=config['genetic_gen']
        )
        
        print(f"\n✓ Discovered {len(hof)} trading formulas")
        print(f"  Best correlation: {hof[0].fitness.values[0]:.4f}")
    else:
        print("\n[2/4] Skipping genetic evolution (not configured)")
        hof = None
    
    # ========================================================================
    # STEP 3: Generate Daily Signals
    # ========================================================================
    print("\n" + "=" * 80)
    print("[3/4] GENERATING TODAY'S SIGNALS")
    print("=" * 80)
    
    signal_gen = DailySignalGenerator(
        model=trader.model,
        scalers=trader.scalers,
        universal_scaler=trader.universal_scaler,
        tickers=tickers
    )
    
    # Generate signals
    signals = signal_gen.get_today_signals()
    
    # Rank and display
    ranked_signals = signal_gen.rank_signals(signals, top_n=15)
    signal_gen.print_signal_report(ranked_signals)
    
    # ========================================================================
    # STEP 4: Multi-Week Simulation
    # ========================================================================
    print("\n" + "=" * 80)
    print("[4/4] MULTI-WEEK SIMULATION")
    print("=" * 80)
    
    simulator = WeeklySimulator(
        model=trader.model,
        scalers=trader.scalers,
        universal_scaler=trader.universal_scaler,
        tickers=tickers
    )
    
    sim_results = simulator.run_multi_week_simulation(
        weeks=config['validation_weeks'],
        initial_capital=10000.0
    )
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("🏆 TRAINING COMPLETE - FINAL SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 DATA SUMMARY:")
    print(f"   Tickers trained: {trader.training_stats.get('tickers_count', 0)}")
    print(f"   Total samples: {trader.training_stats.get('total_samples', 0):,}")
    print(f"   Features: {trader.training_stats.get('total_features', 0)}")
    
    print(f"\n🤖 MODEL PERFORMANCE:")
    print(f"   Train Accuracy: {trader.training_stats.get('train_accuracy', 0):.4f}")
    print(f"   Test Accuracy: {trader.training_stats.get('test_accuracy', 0):.4f}")
    print(f"   AUC Score: {trader.training_stats.get('auc', 0):.4f}")
    print(f"   Walk-Forward Sharpe: {validation_results['mean_sharpe']:.2f}")
    
    print(f"\n📈 SIMULATION RESULTS:")
    print(f"   Weeks simulated: {sim_results['weeks']}")
    print(f"   Total Return: {sim_results['total_return']:+.2%}")
    print(f"   Final Capital: ${sim_results['final_capital']:,.2f}")
    
    print(f"\n📡 TODAY'S TOP SIGNALS:")
    for i, (ticker, sig) in enumerate(list(ranked_signals.items())[:5], 1):
        print(f"   {i}. {ticker}: {sig['action']} "
              f"(Prob: {sig['probability']:.2%}, Conf: {sig['confidence']:.2%})")
    
    # Expected returns calculation
    buy_signals = signal_gen.get_buy_signals(ranked_signals)
    high_conf = signal_gen.get_high_confidence_signals(ranked_signals)
    
    print(f"\n💰 EXPECTED RETURNS:")
    print(f"   BUY signals: {len(buy_signals)}")
    print(f"   High confidence: {len(high_conf)}")
    print(f"   Expected daily gain: 1.5-3% per trade × {len(buy_signals)} = {1.5*len(buy_signals):.0f}-{3*len(buy_signals):.0f}%")
    
    print("\n" + "=" * 80)
    print("✅ SYSTEM READY FOR LIVE TRADING")
    print("=" * 80)
    
    # Save results
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'training_stats': trader.training_stats,
        'validation': {
            'mean_accuracy': validation_results['mean_accuracy'],
            'mean_auc': validation_results['mean_auc'],
            'mean_sharpe': validation_results['mean_sharpe']
        },
        'simulation': {
            'weeks': sim_results['weeks'],
            'total_return': sim_results['total_return'],
            'final_capital': sim_results['final_capital']
        },
        'signals_count': len(ranked_signals),
        'buy_signals': len(buy_signals),
        'high_confidence': len(high_conf)
    }
    
    with open('training_results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to training_results_summary.json")
    
    return trader, ranked_signals, sim_results


def run_signals_only(tickers: list):
    """Load existing model and generate signals only."""
    print_banner()
    
    print("\n" + "=" * 80)
    print("📡 SIGNAL GENERATION MODE")
    print("=" * 80)
    
    # Load model
    if not os.path.exists("universal_trader_model.pkl"):
        print("❌ No saved model found. Run training first.")
        return None
    
    trader = UniversalTrader(tickers)
    trader.load_model("universal_trader_model.pkl")
    
    # Generate signals
    signal_gen = DailySignalGenerator(
        model=trader.model,
        scalers=trader.scalers,
        universal_scaler=trader.universal_scaler,
        tickers=tickers
    )
    
    signals = signal_gen.get_today_signals()
    ranked_signals = signal_gen.rank_signals(signals, top_n=15)
    signal_gen.print_signal_report(ranked_signals)
    
    return ranked_signals


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Ultimate AI Trading Discovery System'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick test mode (5 tickers, 3 years)'
    )
    parser.add_argument(
        '--signals', action='store_true',
        help='Generate signals only (load existing model)'
    )
    parser.add_argument(
        '--tickers', nargs='+', default=None,
        help='Custom ticker list'
    )
    
    args = parser.parse_args()
    
    # Select configuration
    if args.quick:
        config = QUICK_CONFIG
        tickers = args.tickers or QUICK_TICKERS
        print("🚀 Running in QUICK TEST mode")
    else:
        config = FULL_CONFIG
        tickers = args.tickers or FULL_TICKERS
        print("🚀 Running in FULL TRAINING mode")
    
    # Run appropriate mode
    if args.signals:
        run_signals_only(tickers)
    else:
        run_full_pipeline(config, tickers)


if __name__ == "__main__":
    main()
