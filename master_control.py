#!/usr/bin/env python3
"""
🚀 QUANTUM AI TRADER - MASTER CONTROL
=====================================
One script to rule them all.

Commands:
  python master_control.py train      - Train AI in Colab (shows instructions)
  python master_control.py arena      - Run championship arena
  python master_control.py compete    - Start AI vs Human dashboard
  python master_control.py status     - Show graduation status
  python master_control.py research   - Generate research report
  python master_control.py signals    - Get today's signals
"""

import sys
import os
import json
from datetime import datetime, timedelta
import subprocess

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def show_banner():
    """Show banner"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🚀 QUANTUM AI TRADER v4.0 - ALPHAGO EDITION                       ║
║                                                                      ║
║   "I fear not the man who has practiced 10,000 kicks once,          ║
║    but I fear the man who has practiced one kick 10,000 times."     ║
║                                                          - Bruce Lee ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def cmd_train():
    """Show training instructions"""
    print("""
🎮 TRAINING MODE - USE GOOGLE COLAB FOR GPU POWER
=================================================

Steps:
1. Go to: https://colab.research.google.com
2. Upload: ALPHAGO_TRADER.ipynb
3. Enable GPU: Runtime → Change runtime type → GPU
4. Run all cells
5. Download trained models when complete

Files to download:
  • alphago_trader_brain.pt    - Main trained model
  • best_brain_winrate.pt      - Best win rate model
  • strategy_dna.json          - The secret sauce rules
  • todays_predictions.json    - Today's signals

After download, place files in this directory and run:
  python master_control.py signals
""")


def cmd_arena():
    """Run championship arena"""
    print("\n⚔️  STARTING CHAMPIONSHIP ARENA...")
    
    try:
        from championship_arena import main as arena_main
        arena, results = arena_main()
    except ImportError:
        print("Running championship_arena.py...")
        subprocess.run([sys.executable, 'championship_arena.py'])


def cmd_compete():
    """Start competition dashboard"""
    print("\n🏆 STARTING AI vs HUMAN COMPETITION DASHBOARD...")
    print("Open browser to: http://localhost:5001")
    
    try:
        subprocess.run([sys.executable, 'competition_dashboard.py'])
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


def cmd_status():
    """Show graduation status"""
    try:
        from graduation_system import GraduationTracker, TrainingLogger
        
        tracker = GraduationTracker()
        tracker.display_status()
        
        logger = TrainingLogger()
        stats = logger.get_session_stats()
        
        print("\n📊 TRAINING HISTORY:")
        print(f"  Total sessions: {stats['total_sessions']}")
        print(f"  Total episodes: {stats['total_episodes']}")
        
    except Exception as e:
        print(f"Error: {e}")


def cmd_research():
    """Generate research report"""
    print("\n📚 GENERATING RESEARCH REPORT...")
    
    try:
        from graduation_system import ResearchEngine
        
        engine = ResearchEngine()
        report = engine.generate_report()
        
        print("\n" + "="*70)
        print("📊 RESEARCH REPORT")
        print("="*70)
        
        stats = report['session_stats']
        print(f"\nTotal Sessions: {stats['total_sessions']}")
        print(f"Total Episodes: {stats['total_episodes']}")
        
        patterns = report['win_patterns']
        print(f"\nWin Rate: {patterns['win_rate']*100:.1f}%")
        print(f"Avg Win: {patterns.get('avg_win', 0):.2f}%")
        print(f"Avg Loss: {patterns.get('avg_loss', 0):.2f}%")
        
        print("\nBest Strategies:")
        for s in report['best_strategies'][:5]:
            print(f"  • {s['strategy']}: {s['avg_return']:.2f}% return")
        
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
        print(f"\n✅ Full report saved to: research_report.json")
        
    except Exception as e:
        print(f"Error: {e}")


def cmd_signals():
    """Get today's signals"""
    print("\n🎯 GETTING TODAY'S SIGNALS...")
    
    # Check if we have predictions file
    predictions_file = 'todays_predictions.json'
    strategy_dna_file = 'strategy_dna.json'
    
    if os.path.exists(predictions_file):
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
        
        # Show strategy DNA
        if os.path.exists(strategy_dna_file):
            with open(strategy_dna_file, 'r') as f:
                dna = json.load(f)
            
            print("\n🧬 STRATEGY DNA (Secret Sauce):")
            print(f"  DIP BUY: {dna['dip_buy']['min_drop']*100:.0f}% drop + RSI<{dna['dip_buy']['max_rsi']}")
            print(f"  PROFIT: {dna['profit_take']['target']*100:.0f}% target")
            print(f"  CUT LOSS: {dna['cut_loss']['max_loss']*100:.0f}% max loss")
        
        # Show signals
        print("\n" + "="*70)
        print("🎯 TODAY'S SIGNALS")
        print("="*70)
        
        buys = [p for p in predictions if p['action'] == 'BUY']
        dips = [p for p in predictions if p.get('dip_buy')]
        
        if dips:
            print("\n🎯 DIP BUY OPPORTUNITIES (ACT ON THESE):")
            for p in dips:
                print(f"  {p['ticker']:5s} ${p['price']:>7.2f} | Down {p['drawdown']:+.1f}% | RSI: {p['rsi']:.0f}")
        
        print(f"\n🟢 BUY SIGNALS ({len(buys)}):")
        for p in buys[:10]:
            sig = f" ← {p.get('signal', '')}" if p.get('signal') else ""
            print(f"  {p['ticker']:5s} ${p['price']:>7.2f} | {p['buy_prob']:.0%} conf{sig}")
        
        sells = [p for p in predictions if p['action'] == 'SELL']
        if sells:
            print(f"\n🔴 SELL/AVOID ({len(sells)}):")
            for p in sells[:5]:
                print(f"  {p['ticker']:5s} ${p['price']:>7.2f} | {p['sell_prob']:.0%} conf")
        
    else:
        print(f"\n⚠️  No predictions file found: {predictions_file}")
        print("Run training in Colab first and download the predictions file.")
        print("\nGenerating LIVE signals from scratch...")
        
        # Run live predictions
        try:
            from alphago_meta_learner import main as meta_main
            engine, results, predictions = meta_main()
        except Exception as e:
            print(f"Error: {e}")
            print("\nTip: Run 'python master_control.py train' for instructions")


def cmd_help():
    """Show help"""
    print("""
🚀 QUANTUM AI TRADER - COMMANDS
================================

  train     - Instructions to train AI in Google Colab
  arena     - Run championship arena (AI vs champion strategies)
  compete   - Start AI vs Human competition dashboard
  status    - Show graduation progress and achievements
  research  - Generate research report from all training data
  signals   - Get today's trading signals
  help      - Show this help message

Examples:
  python master_control.py train
  python master_control.py arena
  python master_control.py compete
  python master_control.py signals

Files:
  ALPHAGO_TRADER.ipynb      - Colab notebook for GPU training
  championship_arena.py     - Fight AI against champion strategies
  competition_dashboard.py  - AI vs Human paper trading dashboard
  graduation_system.py      - Track AI progress from novice to pro
  alphago_meta_learner.py   - Core AI strategy engine
""")


def main():
    show_banner()
    
    if len(sys.argv) < 2:
        cmd_help()
        return
    
    command = sys.argv[1].lower()
    
    commands = {
        'train': cmd_train,
        'arena': cmd_arena,
        'compete': cmd_compete,
        'status': cmd_status,
        'research': cmd_research,
        'signals': cmd_signals,
        'help': cmd_help,
    }
    
    if command in commands:
        commands[command]()
    else:
        print(f"Unknown command: {command}")
        cmd_help()


if __name__ == '__main__':
    main()
