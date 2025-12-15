"""
DAY 1: OHLCV DATA COLLECTOR WITH CHECKPOINTING
===============================================

Purpose: Collect 2 years of price/volume data for all 353 tickers
Runtime: 3-4 hours (yfinance batch processing + polite delays)
Features: Checkpointing every 50 tickers, error handling, retry logic

This is production code. Not a demo. Runs overnight if needed.

Author: Human-AI Collective
Date: December 15, 2025
"""

import yfinance as yf
import pandas as pd
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Import our database
from day1_foundation_database import TradingDatabase

# Setup logging
LOG_DIR = '/workspaces/quantum-ai-trader_v1.1/logs'
Path(LOG_DIR).mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{LOG_DIR}/ohlcv_collector_{datetime.now().strftime("%Y%m%d_%H%M")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Checkpoint file
CHECKPOINT_FILE = '/workspaces/quantum-ai-trader_v1.1/data/ohlcv_checkpoint.json'


class OHLCVCollector:
    """Production OHLCV collector with checkpointing and error handling."""
    
    def __init__(self, period='2y', batch_size=50, checkpoint_interval=50):
        self.period = period
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.db = TradingDatabase()
        self.db.connect()
        
        # Stats
        self.processed_count = 0
        self.success_count = 0
        self.failed_tickers = []
        self.start_time = None
        
    def load_checkpoint(self):
        """Load previous progress from checkpoint file."""
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
                logger.info(f"Loaded checkpoint: {len(checkpoint['processed'])} tickers already done")
                return set(checkpoint['processed']), checkpoint['failed']
        except FileNotFoundError:
            logger.info("No checkpoint found. Starting fresh.")
            return set(), []
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}. Starting fresh.")
            return set(), []
    
    def save_checkpoint(self, processed_set, failed_list):
        """Save current progress to checkpoint file."""
        try:
            checkpoint = {
                'timestamp': datetime.now().isoformat(),
                'processed': list(processed_set),
                'failed': failed_list,
                'success_count': self.success_count,
                'total_processed': self.processed_count
            }
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            logger.info(f"Checkpoint saved: {len(processed_set)} processed, {len(failed_list)} failed")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def collect_ticker_ohlcv(self, ticker, max_retries=3):
        """Collect OHLCV data for a single ticker with retry logic."""
        for attempt in range(max_retries):
            try:
                # Check if we already have recent data
                latest_date = self.db.get_latest_ohlcv_date(ticker)
                
                if latest_date:
                    latest_dt = datetime.strptime(latest_date, '%Y-%m-%d')
                    days_ago = (datetime.now() - latest_dt).days
                    
                    if days_ago < 7:
                        logger.info(f"[{ticker}] Data up-to-date (last: {latest_date}), skipping")
                        return True, 0
                
                # Download from yfinance
                logger.info(f"[{ticker}] Downloading {self.period} of data...")
                data = yf.download(ticker, period=self.period, progress=False, auto_adjust=False)
                
                if data.empty:
                    logger.warning(f"[{ticker}] No data returned from yfinance")
                    return False, 0
                
                if len(data) < 20:
                    logger.warning(f"[{ticker}] Insufficient data: only {len(data)} bars")
                    return False, 0
                
                # Insert into database
                rows_inserted = self.db.insert_ohlcv_batch(ticker, data)
                logger.info(f"[{ticker}] ✅ Inserted {rows_inserted} bars")
                
                return True, rows_inserted
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"[{ticker}] Failed after {max_retries} attempts: {e}")
                    return False, 0
                else:
                    wait_time = (2 ** attempt) + 1
                    logger.warning(f"[{ticker}] Attempt {attempt+1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
        
        return False, 0
    
    def run_collection(self):
        """Main collection loop with checkpointing."""
        logger.info("=" * 60)
        logger.info("OHLCV DATA COLLECTION STARTED")
        logger.info("=" * 60)
        
        self.start_time = time.time()
        
        # Get all active tickers
        all_tickers = self.db.get_active_tickers()
        logger.info(f"Total tickers to process: {len(all_tickers)}")
        
        # Load checkpoint
        processed_set, checkpoint_failed = self.load_checkpoint()
        self.failed_tickers = checkpoint_failed
        
        # Filter to only unprocessed tickers
        remaining_tickers = [t for t in all_tickers if t not in processed_set]
        logger.info(f"Remaining tickers to process: {len(remaining_tickers)}")
        
        if not remaining_tickers:
            logger.info("✅ All tickers already processed!")
            self.print_summary()
            return
        
        # Process in batches
        for i, ticker in enumerate(remaining_tickers, 1):
            self.processed_count += 1
            total_progress = len(processed_set) + i
            
            # Collect data
            success, bars = self.collect_ticker_ohlcv(ticker)
            
            if success:
                self.success_count += 1
                processed_set.add(ticker)
            else:
                self.failed_tickers.append(ticker)
            
            # Progress update every 10 tickers
            if i % 10 == 0:
                elapsed = time.time() - self.start_time
                rate = i / elapsed if elapsed > 0 else 0
                remaining = len(remaining_tickers) - i
                eta_seconds = remaining / rate if rate > 0 else 0
                
                logger.info(f"Progress: {total_progress}/{len(all_tickers)} ({total_progress/len(all_tickers)*100:.1f}%)")
                logger.info(f"Success rate: {self.success_count}/{self.processed_count} ({self.success_count/self.processed_count*100:.1f}%)")
                logger.info(f"ETA: {eta_seconds/60:.1f} minutes ({eta_seconds/3600:.2f} hours)")
            
            # Checkpoint every 50 tickers
            if i % self.checkpoint_interval == 0:
                self.save_checkpoint(processed_set, self.failed_tickers)
                logger.info(f"📁 Checkpoint saved at {total_progress} tickers")
            
            # Polite delay to avoid rate limiting
            time.sleep(0.5)
        
        # Final checkpoint
        self.save_checkpoint(processed_set, self.failed_tickers)
        
        # Summary
        self.print_summary()
        
        # Close database
        self.db.close()
    
    def print_summary(self):
        """Print final summary report."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        # Get database stats
        stats = self.db.get_database_stats()
        
        print("\n" + "=" * 60)
        print("OHLCV COLLECTION SUMMARY")
        print("=" * 60)
        print(f"Total tickers attempted:     {self.processed_count}")
        print(f"Successfully collected:      {self.success_count}")
        print(f"Failed:                      {len(self.failed_tickers)}")
        print(f"Success rate:                {self.success_count/self.processed_count*100:.1f}%" if self.processed_count > 0 else "N/A")
        print(f"Total runtime:               {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
        print(f"\nDatabase coverage:")
        print(f"  Tickers with data:         {stats['ohlcv_tickers']}")
        print(f"  Total OHLCV bars:          {stats['ohlcv_total_bars']:,}")
        print(f"  Date range:                {stats['ohlcv_earliest']} to {stats['ohlcv_latest']}")
        print("=" * 60)
        
        if len(self.failed_tickers) > 0:
            print(f"\n⚠️ Failed tickers ({len(self.failed_tickers)}):")
            print(f"   {self.failed_tickers[:20]}")
            if len(self.failed_tickers) > 20:
                print(f"   ... and {len(self.failed_tickers) - 20} more")
        
        # KILL SWITCH: Check data quality
        success_rate = self.success_count / self.processed_count if self.processed_count > 0 else 0
        print(f"\n🔍 VALIDATION CHECK:")
        print(f"   Success rate: {success_rate*100:.1f}%")
        
        if success_rate < 0.90:
            print(f"\n❌ KILL SWITCH TRIGGERED: Success rate {success_rate*100:.1f}% < 90%")
            print(f"   Data quality insufficient. Review failed tickers and retry.")
            print(f"   DO NOT proceed to next phase until >90% success rate achieved.")
        else:
            print(f"\n✅ VALIDATION PASSED: {success_rate*100:.1f}% success rate")
            print(f"   Data quality sufficient. Ready for next phase.")


def main():
    """Run OHLCV collection for all 353 tickers."""
    collector = OHLCVCollector(
        period='2y',           # 2 years of historical data
        batch_size=50,         # Process in batches of 50
        checkpoint_interval=50 # Save checkpoint every 50 tickers
    )
    
    collector.run_collection()


if __name__ == "__main__":
    print("🚀 Starting OHLCV data collection...")
    print("⏱️  Estimated runtime: 3-4 hours for 353 tickers")
    print("📁 Checkpoints saved every 50 tickers")
    print("🔄 Safe to interrupt and resume\n")
    
    # Auto-start (no prompt needed for background execution)
    main()
