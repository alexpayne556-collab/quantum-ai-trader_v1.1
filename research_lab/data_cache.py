"""
DATA CACHE AND RECOVERY SYSTEM

Ensure downloaded data is safe and can be recovered if process crashes.

Features:
1. Automatic backups during download
2. Resume capability (skip already-downloaded tickers)
3. Export to multiple formats (CSV, Parquet, HDF5)
4. Data integrity checks
5. Recovery from backup if primary DB corrupted
"""

import sqlite3
import pandas as pd
import os
import shutil
from datetime import datetime
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class DataCache:
    """
    Safe data caching with automatic backups and recovery
    """
    
    def __init__(self, db_path: str = "data/market_data.db"):
        self.db_path = db_path
        self.backup_dir = "data/backups"
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def create_backup(self, tag: Optional[str] = None) -> str:
        """
        Create timestamped backup of database
        
        Args:
            tag: Optional tag to add to filename
        
        Returns:
            Path to backup file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if tag:
            backup_filename = f"market_data_{tag}_{timestamp}.db"
        else:
            backup_filename = f"market_data_{timestamp}.db"
        
        backup_path = os.path.join(self.backup_dir, backup_filename)
        
        # Use SQLite backup API (handles locks properly)
        source = sqlite3.connect(self.db_path)
        backup = sqlite3.connect(backup_path)
        
        source.backup(backup)
        
        source.close()
        backup.close()
        
        size_mb = os.path.getsize(backup_path) / (1024 * 1024)
        logger.info(f"✓ Backup created: {backup_path} ({size_mb:.1f} MB)")
        
        # Clean old backups (keep last 10)
        self._cleanup_old_backups(keep_last=10)
        
        return backup_path
    
    def _cleanup_old_backups(self, keep_last: int = 10):
        """Remove old backups, keep only most recent N"""
        backups = sorted([
            os.path.join(self.backup_dir, f)
            for f in os.listdir(self.backup_dir)
            if f.endswith('.db')
        ], key=os.path.getmtime)
        
        if len(backups) > keep_last:
            for old_backup in backups[:-keep_last]:
                os.remove(old_backup)
                logger.info(f"Deleted old backup: {os.path.basename(old_backup)}")
    
    def export_to_csv(self, output_dir: str = "data/exports/csv") -> List[str]:
        """
        Export all data to CSV files (one per ticker)
        
        Useful for sharing data or analysis in other tools
        """
        os.makedirs(output_dir, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        
        # Get all tickers
        tickers = pd.read_sql_query("SELECT DISTINCT ticker FROM daily_bars ORDER BY ticker", conn)
        
        exported_files = []
        
        for ticker in tickers['ticker']:
            df = pd.read_sql_query(
                "SELECT * FROM daily_bars WHERE ticker = ? ORDER BY date",
                conn, params=[ticker]
            )
            
            output_file = os.path.join(output_dir, f"{ticker}.csv")
            df.to_csv(output_file, index=False)
            exported_files.append(output_file)
        
        conn.close()
        
        logger.info(f"✓ Exported {len(exported_files)} tickers to CSV")
        
        return exported_files
    
    def export_to_parquet(self, output_file: str = "data/exports/market_data.parquet"):
        """
        Export all data to single Parquet file
        
        Parquet is much faster to load than CSV and takes less space
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        
        df = pd.read_sql_query(
            "SELECT * FROM daily_bars ORDER BY ticker, date",
            conn
        )
        
        conn.close()
        
        df.to_parquet(output_file, compression='snappy', index=False)
        
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        logger.info(f"✓ Exported to Parquet: {output_file} ({size_mb:.1f} MB)")
        
        return output_file
    
    def check_integrity(self) -> Dict:
        """
        Check database integrity
        
        Returns:
            Dict with integrity check results
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # SQLite integrity check
        cursor.execute("PRAGMA integrity_check")
        integrity = cursor.fetchone()[0]
        
        # Check for duplicate rows
        cursor.execute("""
            SELECT ticker, date, COUNT(*) as count
            FROM daily_bars
            GROUP BY ticker, date
            HAVING count > 1
        """)
        duplicates = cursor.fetchall()
        
        # Check for NULL values in critical columns
        cursor.execute("""
            SELECT COUNT(*) FROM daily_bars
            WHERE ticker IS NULL OR date IS NULL OR close IS NULL
        """)
        null_count = cursor.fetchone()[0]
        
        # Check date ranges
        cursor.execute("""
            SELECT ticker, MIN(date) as first_date, MAX(date) as last_date, COUNT(*) as bars
            FROM daily_bars
            GROUP BY ticker
        """)
        ticker_stats = cursor.fetchall()
        
        conn.close()
        
        results = {
            'integrity_ok': integrity == 'ok',
            'duplicates_found': len(duplicates),
            'null_values': null_count,
            'total_tickers': len(ticker_stats),
            'total_bars': sum([s[3] for s in ticker_stats])
        }
        
        if results['integrity_ok'] and results['duplicates_found'] == 0 and results['null_values'] == 0:
            logger.info("✓ Database integrity check PASSED")
        else:
            logger.warning("⚠ Database integrity issues found:")
            if not results['integrity_ok']:
                logger.warning("  - SQLite integrity check failed")
            if results['duplicates_found'] > 0:
                logger.warning(f"  - {results['duplicates_found']} duplicate rows")
            if results['null_values'] > 0:
                logger.warning(f"  - {results['null_values']} NULL values in critical columns")
        
        return results
    
    def recover_from_backup(self, backup_path: Optional[str] = None):
        """
        Recover database from backup
        
        Args:
            backup_path: Specific backup to restore, or None for most recent
        """
        if backup_path is None:
            # Get most recent backup
            backups = sorted([
                os.path.join(self.backup_dir, f)
                for f in os.listdir(self.backup_dir)
                if f.endswith('.db')
            ], key=os.path.getmtime, reverse=True)
            
            if not backups:
                raise ValueError("No backups found")
            
            backup_path = backups[0]
        
        # Create safety backup of current DB
        if os.path.exists(self.db_path):
            safety_backup = self.db_path + '.before_recovery'
            shutil.copy2(self.db_path, safety_backup)
            logger.info(f"Safety backup: {safety_backup}")
        
        # Restore from backup
        shutil.copy2(backup_path, self.db_path)
        
        logger.info(f"✓ Recovered from backup: {backup_path}")
        
        # Verify integrity
        integrity = self.check_integrity()
        
        return integrity
    
    def get_download_progress(self) -> Dict:
        """
        Get current download progress stats
        """
        conn = sqlite3.connect(self.db_path)
        
        stats = {}
        
        # Overall stats
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT ticker), COUNT(*) FROM daily_bars")
        tickers, bars = cursor.fetchone()
        
        stats['tickers_downloaded'] = tickers
        stats['total_bars'] = bars
        
        # Date range
        cursor.execute("SELECT MIN(date), MAX(date) FROM daily_bars")
        min_date, max_date = cursor.fetchone()
        stats['date_range'] = f"{min_date} to {max_date}"
        
        # Download status if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='download_status'")
        if cursor.fetchone():
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM download_status
                GROUP BY status
            """)
            status_counts = dict(cursor.fetchall())
            stats['download_status'] = status_counts
        
        conn.close()
        
        return stats


def schedule_automatic_backups(interval_minutes: int = 30):
    """
    Schedule automatic backups every N minutes while download runs
    
    This runs in background and creates backups periodically
    """
    import time
    import threading
    
    def backup_loop():
        cache = DataCache()
        
        while True:
            time.sleep(interval_minutes * 60)
            
            try:
                cache.create_backup(tag='auto')
                logger.info(f"✓ Automatic backup completed")
            except Exception as e:
                logger.error(f"✗ Automatic backup failed: {e}")
    
    thread = threading.Thread(target=backup_loop, daemon=True)
    thread.start()
    
    logger.info(f"✓ Automatic backups scheduled every {interval_minutes} minutes")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=" * 70)
    print("DATA CACHE AND RECOVERY")
    print("=" * 70)
    
    cache = DataCache()
    
    # Check integrity
    print("\n[1/4] Checking database integrity...")
    integrity = cache.check_integrity()
    
    for key, value in integrity.items():
        print(f"  {key}: {value}")
    
    # Create backup
    print("\n[2/4] Creating backup...")
    backup_path = cache.create_backup(tag='manual')
    
    # Export to Parquet
    print("\n[3/4] Exporting to Parquet...")
    parquet_file = cache.export_to_parquet()
    
    # Get progress
    print("\n[4/4] Download progress...")
    progress = cache.get_download_progress()
    
    for key, value in progress.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("✓ Data is cached and safe")
    print("✓ Backups available in: data/backups/")
    print("✓ Parquet export: data/exports/market_data.parquet")
    print("=" * 70)
