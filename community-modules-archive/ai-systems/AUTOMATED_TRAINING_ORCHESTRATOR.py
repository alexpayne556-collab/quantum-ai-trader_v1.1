"""
AUTOMATED_TRAINING_ORCHESTRATOR.py
===================================
Master orchestrator for automated training and self-improvement

Combines: Real data collection + Continuous learning + Active learning
Runs on schedule: Daily/Weekly/Monthly
"""

import schedule
import time
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutomatedTrainingOrchestrator:
    """
    Complete auto-training system
    
    Schedule:
    - Daily: Collect new market data
    - After each trade: Update model (continuous learning)
    - Weekly: Active learning on uncertain examples
    - Monthly: Full retraining
    """
    
    def __init__(self, models_dir: str = 'models', data_dir: str = 'training_data'):
        """Initialize orchestrator"""
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        try:
            from REAL_DATA_COLLECTOR import RealDataCollector
            from CONTINUOUS_LEARNING_SYSTEM import ContinuousLearningSystem
            from ACTIVE_LEARNING_SYSTEM import ActiveLearningSystem
            
            self.data_collector = RealDataCollector()
            
            # Initialize continuous learners for each model
            self.continuous_learners = {
                'pump_detector': ContinuousLearningSystem(
                    model_path=str(self.models_dir / 'pump_detector.pkl')
                ),
                'ofi_predictor': ContinuousLearningSystem(
                    model_path=str(self.models_dir / 'ofi_predictor.pkl')
                )
            }
            
            logger.info("✅ Orchestrator initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            raise
    
    def daily_data_collection(self):
        """Collect yesterday's market data"""
        logger.info(f"\n{'='*80}")
        logger.info(f"📦 DAILY DATA COLLECTION - {datetime.now()}")
        logger.info(f"{'='*80}\n")
        
        try:
            # Get penny stock universe
            universe = self.data_collector.get_penny_stock_universe()
            
            # Collect data from last 7 days
            X, y = self.data_collector.collect_pump_training_data(
                universe, 
                max_symbols=20  # Smaller batch for daily updates
            )
            
            if len(X) > 10:
                # Save incremental data
                timestamp = datetime.now().strftime('%Y%m%d')
                X.to_csv(self.data_dir / f'daily_features_{timestamp}.csv', index=False)
                
                logger.info(f"✅ Collected {len(X)} new training examples")
            else:
                logger.warning(f"⚠️ Only collected {len(X)} examples")
                
        except Exception as e:
            logger.error(f"❌ Daily collection failed: {e}")
    
    def update_after_trade(self, model_name: str, features, outcome: int):
        """Update model after each trade (continuous learning)"""
        
        if model_name not in self.continuous_learners:
            logger.warning(f"Unknown model: {model_name}")
            return
        
        try:
            self.continuous_learners[model_name].update_with_new_trade(features, outcome)
            
            # Log performance
            metrics = self.continuous_learners[model_name].get_performance_metrics()
            logger.info(f"📊 {model_name} accuracy: {metrics['accuracy']:.1%}")
            
        except Exception as e:
            logger.error(f"Trade update failed: {e}")
    
    def weekly_active_learning(self):
        """Active learning on uncertain predictions"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🎓 WEEKLY ACTIVE LEARNING - {datetime.now()}")
        logger.info(f"{'='*80}\n")
        
        try:
            from ACTIVE_LEARNING_SYSTEM import ActiveLearningSystem
            import joblib
            import numpy as np
            
            # Load pump detector
            pump_model = joblib.load(self.models_dir / 'pump_detector.pkl')
            
            # Get recent unlabeled data
            recent_data = self._get_recent_unlabeled_data()
            
            if recent_data is not None and len(recent_data) > 0:
                # Initialize active learning
                al_system = ActiveLearningSystem(
                    model=pump_model,
                    unlabeled_pool=recent_data
                )
                
                # Run active learning
                al_system.active_learning_loop(
                    n_iterations=5,
                    samples_per_iteration=10
                )
                
                logger.info(f"✅ Active learning complete")
            else:
                logger.warning("No unlabeled data available")
                
        except Exception as e:
            logger.error(f"❌ Active learning failed: {e}")
    
    def monthly_full_retrain(self):
        """Full retraining from scratch"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🔄 MONTHLY FULL RETRAINING - {datetime.now()}")
        logger.info(f"{'='*80}\n")
        
        try:
            # Collect fresh data from last 6 months
            universe = self.data_collector.get_penny_stock_universe()
            
            logger.info("Collecting 6 months of data...")
            X, y = self.data_collector.collect_pump_training_data(
                universe,
                max_symbols=100
            )
            
            if len(X) > 100:
                # Train pump detector
                self._train_pump_detector(X, y)
                
                # Train OFI predictor (if applicable)
                # self._train_ofi_predictor(X, y)
                
                logger.info(f"✅ Full retraining complete with {len(X)} examples")
            else:
                logger.warning(f"⚠️ Insufficient data: {len(X)} examples")
                
        except Exception as e:
            logger.error(f"❌ Full retraining failed: {e}")
    
    def _train_pump_detector(self, X, y):
        """Train pump detection model"""
        import lightgbm as lgb
        import joblib
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        logger.info(f"  Train accuracy: {train_acc:.1%}")
        logger.info(f"  Test accuracy: {test_acc:.1%}")
        
        # Save
        joblib.dump(model, self.models_dir / 'pump_detector.pkl')
        logger.info(f"  ✅ Saved pump_detector.pkl")
        
        # Update continuous learner cache
        self.continuous_learners['pump_detector'].load_initial_training_data(
            X_train.values if hasattr(X_train, 'values') else X_train,
            y_train
        )
    
    def _get_recent_unlabeled_data(self):
        """Get recent data that hasn't been labeled yet"""
        import pandas as pd
        import numpy as np
        
        try:
            # Load recent daily collections
            recent_files = sorted(self.data_dir.glob('daily_features_*.csv'))
            
            if len(recent_files) > 0:
                # Get most recent
                recent_df = pd.read_csv(recent_files[-1])
                return recent_df.values
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to load unlabeled data: {e}")
            return None
    
    def run_once_now(self):
        """Run all tasks once immediately (for testing)"""
        logger.info("\n" + "="*80)
        logger.info("🚀 RUNNING ALL TASKS NOW (TEST MODE)")
        logger.info("="*80)
        
        # Run each task
        self.daily_data_collection()
        time.sleep(2)
        
        self.weekly_active_learning()
        time.sleep(2)
        
        self.monthly_full_retrain()
        
        logger.info("\n✅ All tasks completed!")
    
    def start_scheduled_mode(self):
        """Start orchestrator in scheduled mode"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 AUTOMATED TRAINING ORCHESTRATOR STARTED")
        logger.info(f"{'='*80}\n")
        logger.info(f"Schedule:")
        logger.info(f"  • Daily data collection: 2:00 AM")
        logger.info(f"  • Weekly active learning: Sunday 3:00 AM")
        logger.info(f"  • Monthly full retrain: 1st of month 4:00 AM")
        logger.info(f"  • Continuous learning: After every trade")
        logger.info(f"\n{'='*80}\n")
        
        # Schedule tasks
        schedule.every().day.at("02:00").do(self.daily_data_collection)
        schedule.every().sunday.at("03:00").do(self.weekly_active_learning)
        schedule.every().month.at("01 04:00").do(self.monthly_full_retrain)
        
        # Run loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("\n🛑 Orchestrator stopped by user")


if __name__ == '__main__':
    import sys
    
    # Initialize orchestrator
    orchestrator = AutomatedTrainingOrchestrator(
        models_dir='/content/drive/MyDrive/QuantumAI/models',
        data_dir='/content/drive/MyDrive/QuantumAI/training_data'
    )
    
    # Check if running in test mode
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Run once for testing
        orchestrator.run_once_now()
    else:
        # Start scheduled mode
        orchestrator.start_scheduled_mode()

