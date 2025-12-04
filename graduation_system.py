"""
🎓 GRADUATION SYSTEM - AI TRAINING TO PRO
==========================================
Complete training pipeline from novice to confident pro.

STAGES:
1. BOOTCAMP - Basic training on historical data
2. SPARRING - Fight against champion strategies  
3. GRADUATION - Must beat 3/4 champions
4. PAPER TRADING - 1 week vs human
5. DEPLOYMENT - Ready for real action

Each run is logged for research and improvement.
"""

import json
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

# ============================================================================
# TRAINING LOG - EVERY RUN IS RECORDED FOR RESEARCH
# ============================================================================

class TrainingLogger:
    """
    Logs every training run for research.
    This builds the dataset for future improvements.
    """
    
    def __init__(self, log_dir='training_logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.current_session = None
    
    def start_session(self, session_type):
        """Start a new training session"""
        self.current_session = {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'session_type': session_type,
            'started_at': datetime.now().isoformat(),
            'episodes': [],
            'metrics': {},
            'notes': [],
        }
        return self.current_session['session_id']
    
    def log_episode(self, episode_data):
        """Log an episode"""
        if self.current_session:
            self.current_session['episodes'].append({
                'timestamp': datetime.now().isoformat(),
                **episode_data
            })
    
    def log_metric(self, name, value):
        """Log a metric"""
        if self.current_session:
            if name not in self.current_session['metrics']:
                self.current_session['metrics'][name] = []
            self.current_session['metrics'][name].append({
                'timestamp': datetime.now().isoformat(),
                'value': value
            })
    
    def add_note(self, note):
        """Add a note"""
        if self.current_session:
            self.current_session['notes'].append({
                'timestamp': datetime.now().isoformat(),
                'note': note
            })
    
    def end_session(self, summary=None):
        """End and save session"""
        if self.current_session:
            self.current_session['ended_at'] = datetime.now().isoformat()
            self.current_session['summary'] = summary or {}
            
            # Save to file
            filename = f"{self.current_session['session_id']}_{self.current_session['session_type']}.json"
            filepath = self.log_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(self.current_session, f, indent=2, default=str)
            
            print(f"📝 Session logged: {filepath}")
            return filepath
    
    def get_all_sessions(self):
        """Get all logged sessions"""
        sessions = []
        for filepath in self.log_dir.glob('*.json'):
            with open(filepath, 'r') as f:
                sessions.append(json.load(f))
        return sorted(sessions, key=lambda x: x['started_at'], reverse=True)
    
    def get_session_stats(self):
        """Get aggregate stats across all sessions"""
        sessions = self.get_all_sessions()
        
        stats = {
            'total_sessions': len(sessions),
            'total_episodes': sum(len(s.get('episodes', [])) for s in sessions),
            'by_type': {},
            'best_results': {},
        }
        
        for session in sessions:
            stype = session.get('session_type', 'unknown')
            if stype not in stats['by_type']:
                stats['by_type'][stype] = 0
            stats['by_type'][stype] += 1
            
            # Track best results
            if 'summary' in session:
                summary = session['summary']
                for key in ['total_return', 'win_rate', 'sharpe']:
                    if key in summary:
                        if key not in stats['best_results'] or summary[key] > stats['best_results'][key]['value']:
                            stats['best_results'][key] = {
                                'value': summary[key],
                                'session_id': session['session_id']
                            }
        
        return stats


# ============================================================================
# GRADUATION TRACKER - PROGRESS TO PRO
# ============================================================================

class GraduationTracker:
    """
    Tracks AI progress from novice to pro.
    """
    
    STAGES = {
        'BOOTCAMP': {
            'description': 'Basic training on historical data',
            'requirements': {
                'min_episodes': 100,
                'min_trades': 50,
            }
        },
        'SPARRING': {
            'description': 'Fight against champion strategies',
            'requirements': {
                'battles': 10,
                'win_rate_vs_benchmark': 0.4,
            }
        },
        'GRADUATION': {
            'description': 'Must beat 3/4 champions',
            'requirements': {
                'champions_beaten': 3,
                'total_return': 0,  # Must be positive
            }
        },
        'PAPER_TRADING': {
            'description': '1 week vs human trades',
            'requirements': {
                'days': 5,
                'trades_executed': 10,
            }
        },
        'PRO': {
            'description': 'Ready for real action!',
            'requirements': {
                'paper_profit': True,
                'human_approval': True,
            }
        }
    }
    
    def __init__(self, progress_file='graduation_progress.json'):
        self.progress_file = progress_file
        self.load_progress()
    
    def load_progress(self):
        """Load progress from file"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                'current_stage': 'BOOTCAMP',
                'started_at': datetime.now().isoformat(),
                'history': [],
                'achievements': [],
                'stats': {
                    'total_episodes': 0,
                    'total_trades': 0,
                    'battles_won': 0,
                    'battles_lost': 0,
                    'champions_beaten': [],
                    'paper_trading_days': 0,
                }
            }
            self.save_progress()
    
    def save_progress(self):
        """Save progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def add_achievement(self, achievement):
        """Add an achievement"""
        if achievement not in self.progress['achievements']:
            self.progress['achievements'].append({
                'name': achievement,
                'achieved_at': datetime.now().isoformat()
            })
            self.save_progress()
            print(f"🏅 ACHIEVEMENT UNLOCKED: {achievement}")
    
    def update_stats(self, **kwargs):
        """Update stats"""
        for key, value in kwargs.items():
            if key in self.progress['stats']:
                if isinstance(self.progress['stats'][key], list):
                    if value not in self.progress['stats'][key]:
                        self.progress['stats'][key].append(value)
                else:
                    self.progress['stats'][key] += value
        self.save_progress()
    
    def check_stage_completion(self):
        """Check if current stage is complete"""
        stage = self.progress['current_stage']
        requirements = self.STAGES[stage]['requirements']
        stats = self.progress['stats']
        
        if stage == 'BOOTCAMP':
            return (stats['total_episodes'] >= requirements['min_episodes'] and
                    stats['total_trades'] >= requirements['min_trades'])
        
        elif stage == 'SPARRING':
            total_battles = stats['battles_won'] + stats['battles_lost']
            if total_battles < requirements['battles']:
                return False
            win_rate = stats['battles_won'] / total_battles
            return win_rate >= requirements['win_rate_vs_benchmark']
        
        elif stage == 'GRADUATION':
            return len(stats['champions_beaten']) >= requirements['champions_beaten']
        
        elif stage == 'PAPER_TRADING':
            return stats['paper_trading_days'] >= requirements['days']
        
        return False
    
    def advance_stage(self):
        """Advance to next stage if requirements met"""
        if not self.check_stage_completion():
            return False
        
        current = self.progress['current_stage']
        stages = list(self.STAGES.keys())
        current_idx = stages.index(current)
        
        if current_idx < len(stages) - 1:
            next_stage = stages[current_idx + 1]
            
            self.progress['history'].append({
                'from_stage': current,
                'to_stage': next_stage,
                'advanced_at': datetime.now().isoformat()
            })
            
            self.progress['current_stage'] = next_stage
            self.save_progress()
            
            print(f"\n🎉 STAGE COMPLETE! Advanced from {current} to {next_stage}")
            self.add_achievement(f"Completed {current}")
            
            return True
        
        return False
    
    def get_status(self):
        """Get current status"""
        stage = self.progress['current_stage']
        requirements = self.STAGES[stage]['requirements']
        stats = self.progress['stats']
        
        status = {
            'stage': stage,
            'description': self.STAGES[stage]['description'],
            'requirements': requirements,
            'current_progress': {},
            'is_complete': self.check_stage_completion(),
        }
        
        # Calculate progress for each requirement
        if stage == 'BOOTCAMP':
            status['current_progress'] = {
                'episodes': f"{stats['total_episodes']}/{requirements['min_episodes']}",
                'trades': f"{stats['total_trades']}/{requirements['min_trades']}",
            }
        elif stage == 'SPARRING':
            total = stats['battles_won'] + stats['battles_lost']
            status['current_progress'] = {
                'battles': f"{total}/{requirements['battles']}",
                'win_rate': f"{stats['battles_won']}/{total if total > 0 else 1}",
            }
        elif stage == 'GRADUATION':
            status['current_progress'] = {
                'champions_beaten': f"{len(stats['champions_beaten'])}/{requirements['champions_beaten']}",
                'beaten': stats['champions_beaten'],
            }
        elif stage == 'PAPER_TRADING':
            status['current_progress'] = {
                'days': f"{stats['paper_trading_days']}/{requirements['days']}",
            }
        
        return status
    
    def display_status(self):
        """Display formatted status"""
        status = self.get_status()
        
        print("\n" + "="*70)
        print(f"🎓 GRADUATION TRACKER - {status['stage']}")
        print("="*70)
        print(f"Stage: {status['description']}")
        print(f"\nProgress:")
        for key, value in status['current_progress'].items():
            print(f"  • {key}: {value}")
        
        if status['is_complete']:
            print(f"\n✅ STAGE COMPLETE! Ready to advance.")
        else:
            print(f"\n⏳ In progress...")
        
        print(f"\nAchievements: {len(self.progress['achievements'])}")
        for ach in self.progress['achievements'][-5:]:
            print(f"  🏅 {ach['name']}")
        
        print("="*70)


# ============================================================================
# POINT SYSTEM - COMPETITIVE SCORING
# ============================================================================

class PointSystem:
    """
    Competitive point system for AI vs Human.
    Points awarded for good trades, penalties for bad ones.
    """
    
    POINTS = {
        # Winning trades
        'small_win': 10,      # 0-3% profit
        'medium_win': 25,     # 3-8% profit
        'big_win': 50,        # 8%+ profit (like HOOD trade)
        
        # Losing trades
        'small_loss': -5,     # -3% to 0%
        'medium_loss': -15,   # -5% to -3%
        'big_loss': -30,      # worse than -5%
        
        # Special bonuses
        'dip_buy_success': 20,     # Bought a dip that recovered
        'profit_take_good': 15,    # Took profits at good time
        'quick_cut_loss': 10,      # Cut loss quickly
        
        # Penalties
        'held_loser_too_long': -20,  # Held a loser too long
        'missed_dip': -10,           # Missed obvious dip buy
        'sold_too_early': -10,       # Sold before 8% target
        
        # Streaks
        'win_streak_3': 25,
        'win_streak_5': 50,
        'win_streak_10': 100,
    }
    
    def __init__(self):
        self.ai_points = 0
        self.human_points = 0
        self.ai_streak = 0
        self.human_streak = 0
    
    def award_trade_points(self, trader, pnl_pct, special=None):
        """Award points for a trade"""
        points = 0
        
        # Base points from P&L
        if pnl_pct >= 8:
            points = self.POINTS['big_win']
        elif pnl_pct >= 3:
            points = self.POINTS['medium_win']
        elif pnl_pct >= 0:
            points = self.POINTS['small_win']
        elif pnl_pct >= -3:
            points = self.POINTS['small_loss']
        elif pnl_pct >= -5:
            points = self.POINTS['medium_loss']
        else:
            points = self.POINTS['big_loss']
        
        # Special bonuses/penalties
        if special:
            if special in self.POINTS:
                points += self.POINTS[special]
        
        # Update streak
        if trader == 'ai':
            if pnl_pct > 0:
                self.ai_streak += 1
                if self.ai_streak == 3:
                    points += self.POINTS['win_streak_3']
                elif self.ai_streak == 5:
                    points += self.POINTS['win_streak_5']
                elif self.ai_streak == 10:
                    points += self.POINTS['win_streak_10']
            else:
                self.ai_streak = 0
            self.ai_points += points
        else:
            if pnl_pct > 0:
                self.human_streak += 1
                if self.human_streak == 3:
                    points += self.POINTS['win_streak_3']
                elif self.human_streak == 5:
                    points += self.POINTS['win_streak_5']
                elif self.human_streak == 10:
                    points += self.POINTS['win_streak_10']
            else:
                self.human_streak = 0
            self.human_points += points
        
        return points
    
    def get_scoreboard(self):
        """Get point scoreboard"""
        return {
            'ai_points': self.ai_points,
            'human_points': self.human_points,
            'leader': 'AI' if self.ai_points > self.human_points else 'HUMAN',
            'lead_by': abs(self.ai_points - self.human_points),
            'ai_streak': self.ai_streak,
            'human_streak': self.human_streak,
        }


# ============================================================================
# RESEARCH ENGINE - LEARN FROM ALL DATA
# ============================================================================

class ResearchEngine:
    """
    Analyzes all training data to find what works.
    This is for YOUR research after the week of paper trading.
    """
    
    def __init__(self, log_dir='training_logs'):
        self.logger = TrainingLogger(log_dir)
    
    def analyze_win_patterns(self):
        """Find patterns in winning trades"""
        sessions = self.logger.get_all_sessions()
        
        wins = []
        losses = []
        
        for session in sessions:
            for episode in session.get('episodes', []):
                for trade in episode.get('trades', []):
                    if trade.get('pnl', 0) > 0:
                        wins.append(trade)
                    elif trade.get('pnl', 0) < 0:
                        losses.append(trade)
        
        analysis = {
            'total_wins': len(wins),
            'total_losses': len(losses),
            'win_rate': len(wins) / max(1, len(wins) + len(losses)),
        }
        
        if wins:
            analysis['avg_win'] = np.mean([t['pnl'] for t in wins])
            analysis['max_win'] = max(t['pnl'] for t in wins)
            analysis['winning_tickers'] = {}
            for t in wins:
                ticker = t.get('ticker', 'unknown')
                if ticker not in analysis['winning_tickers']:
                    analysis['winning_tickers'][ticker] = 0
                analysis['winning_tickers'][ticker] += 1
        
        if losses:
            analysis['avg_loss'] = np.mean([t['pnl'] for t in losses])
            analysis['max_loss'] = min(t['pnl'] for t in losses)
        
        return analysis
    
    def find_best_strategies(self):
        """Find the best performing strategies"""
        sessions = self.logger.get_all_sessions()
        
        strategies = {}
        
        for session in sessions:
            summary = session.get('summary', {})
            strategy_name = summary.get('strategy', 'unknown')
            
            if strategy_name not in strategies:
                strategies[strategy_name] = []
            
            strategies[strategy_name].append({
                'session_id': session['session_id'],
                'return': summary.get('total_return', 0),
                'win_rate': summary.get('win_rate', 0),
            })
        
        # Calculate averages
        best = []
        for name, results in strategies.items():
            avg_return = np.mean([r['return'] for r in results])
            avg_win_rate = np.mean([r['win_rate'] for r in results])
            best.append({
                'strategy': name,
                'avg_return': avg_return,
                'avg_win_rate': avg_win_rate,
                'sessions': len(results),
            })
        
        return sorted(best, key=lambda x: x['avg_return'], reverse=True)
    
    def generate_report(self):
        """Generate full research report"""
        stats = self.logger.get_session_stats()
        win_patterns = self.analyze_win_patterns()
        best_strategies = self.find_best_strategies()
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'session_stats': stats,
            'win_patterns': win_patterns,
            'best_strategies': best_strategies[:10],
            'recommendations': []
        }
        
        # Add recommendations
        if win_patterns.get('win_rate', 0) < 0.5:
            report['recommendations'].append(
                "Win rate below 50% - focus on entry timing"
            )
        
        if win_patterns.get('avg_loss', 0) < -5:
            report['recommendations'].append(
                "Average loss too high - implement stricter stop losses"
            )
        
        # Save report
        with open('research_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


# ============================================================================
# MAIN - GRADUATION PIPELINE
# ============================================================================

def main():
    print("="*70)
    print("🎓 AI GRADUATION SYSTEM")
    print("="*70)
    print("\nThis system tracks AI progress from novice to confident pro.")
    print("\nSTAGES:")
    for stage, info in GraduationTracker.STAGES.items():
        print(f"  {stage}: {info['description']}")
    
    # Initialize components
    logger = TrainingLogger()
    tracker = GraduationTracker()
    points = PointSystem()
    research = ResearchEngine()
    
    # Show current status
    tracker.display_status()
    
    # Show session stats
    print("\n📊 TRAINING HISTORY:")
    stats = logger.get_session_stats()
    print(f"  Total sessions: {stats['total_sessions']}")
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  By type: {stats['by_type']}")
    
    if stats['best_results']:
        print("\n🏆 BEST RESULTS:")
        for key, val in stats['best_results'].items():
            print(f"  {key}: {val['value']:.2f} (Session: {val['session_id']})")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Run championship_arena.py to fight champions")
    print("2. Run competition_dashboard.py to paper trade vs yourself")
    print("3. Run ALPHAGO_TRADER.ipynb in Colab for GPU training")
    print("4. Log all results for research")
    print("5. Graduate when AI beats 3/4 champions!")
    print("="*70)
    
    return logger, tracker, points, research


if __name__ == '__main__':
    logger, tracker, points, research = main()
