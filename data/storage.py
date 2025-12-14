"""
Data Storage - Parquet file management

Handles efficient storage and retrieval of market data using Parquet format.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional, List

from .config import *

logger = logging.getLogger(__name__)


class DataStorage:
    """
    Manages parquet file storage for market data.
    
    Features:
    - Efficient parquet compression
    - Organized directory structure
    - Date-stamped files
    - Easy load/save operations
    """
    
    def __init__(self):
        """Initialize storage manager."""
        self.raw_dir = RAW_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR
        self.reference_dir = REFERENCE_DATA_DIR
        
        logger.debug("DataStorage initialized")
    
    def save_ticker_data(
        self,
        df: pd.DataFrame,
        ticker: str,
        data_type: str = 'raw',
        suffix: str = ''
    ) -> Path:
        """
        Save ticker data to parquet.
        
        Args:
            df: DataFrame to save
            ticker: Ticker symbol
            data_type: 'raw', 'processed', or 'reference'
            suffix: Optional suffix for filename
        
        Returns:
            Path to saved file
        """
        # Select directory
        if data_type == 'raw':
            base_dir = self.raw_dir
        elif data_type == 'processed':
            base_dir = self.processed_dir
        elif data_type == 'reference':
            base_dir = self.reference_dir
        else:
            raise ValueError(f"Invalid data_type: {data_type}")
        
        # Create ticker subdirectory
        ticker_dir = base_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        filename = f"{ticker}_{timestamp}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".parquet"
        
        filepath = ticker_dir / filename
        
        # Save to parquet
        df.to_parquet(
            filepath,
            compression=PARQUET_COMPRESSION,
            engine=PARQUET_ENGINE
        )
        
        logger.debug(f"Saved {ticker} to {filepath}")
        return filepath
    
    def load_ticker_data(
        self,
        ticker: str,
        data_type: str = 'raw',
        latest: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Load ticker data from parquet.
        
        Args:
            ticker: Ticker symbol
            data_type: 'raw', 'processed', or 'reference'
            latest: Load most recent file (True) or all files (False)
        
        Returns:
            DataFrame or None if not found
        """
        # Select directory
        if data_type == 'raw':
            base_dir = self.raw_dir
        elif data_type == 'processed':
            base_dir = self.processed_dir
        elif data_type == 'reference':
            base_dir = self.reference_dir
        else:
            raise ValueError(f"Invalid data_type: {data_type}")
        
        ticker_dir = base_dir / ticker
        
        if not ticker_dir.exists():
            logger.debug(f"No data found for {ticker} in {data_type}")
            return None
        
        # Get all parquet files
        parquet_files = sorted(ticker_dir.glob("*.parquet"))
        
        if not parquet_files:
            logger.debug(f"No parquet files found for {ticker}")
            return None
        
        if latest:
            # Load most recent file
            filepath = parquet_files[-1]
            df = pd.read_parquet(filepath, engine=PARQUET_ENGINE)
            logger.debug(f"Loaded {ticker} from {filepath.name}")
            return df
        else:
            # Load and combine all files
            dfs = []
            for filepath in parquet_files:
                df = pd.read_parquet(filepath, engine=PARQUET_ENGINE)
                dfs.append(df)
            
            combined = pd.concat(dfs)
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
            
            logger.debug(f"Loaded {len(parquet_files)} files for {ticker}")
            return combined
    
    def list_tickers(self, data_type: str = 'raw') -> List[str]:
        """
        List all tickers with data.
        
        Args:
            data_type: 'raw', 'processed', or 'reference'
        
        Returns:
            List of ticker symbols
        """
        if data_type == 'raw':
            base_dir = self.raw_dir
        elif data_type == 'processed':
            base_dir = self.processed_dir
        elif data_type == 'reference':
            base_dir = self.reference_dir
        else:
            raise ValueError(f"Invalid data_type: {data_type}")
        
        if not base_dir.exists():
            return []
        
        # Get all subdirectories (each is a ticker)
        tickers = [d.name for d in base_dir.iterdir() if d.is_dir()]
        return sorted(tickers)
    
    def cleanup_old_files(self, data_type: str = 'raw', keep_latest: int = 5):
        """
        Remove old parquet files, keeping only the most recent.
        
        Args:
            data_type: 'raw', 'processed', or 'reference'
            keep_latest: Number of recent files to keep per ticker
        """
        if data_type == 'raw':
            base_dir = self.raw_dir
        elif data_type == 'processed':
            base_dir = self.processed_dir
        elif data_type == 'reference':
            base_dir = self.reference_dir
        else:
            raise ValueError(f"Invalid data_type: {data_type}")
        
        removed_count = 0
        
        for ticker_dir in base_dir.iterdir():
            if not ticker_dir.is_dir():
                continue
            
            # Get all parquet files sorted by date
            parquet_files = sorted(ticker_dir.glob("*.parquet"))
            
            # Remove old files
            if len(parquet_files) > keep_latest:
                for filepath in parquet_files[:-keep_latest]:
                    filepath.unlink()
                    removed_count += 1
                    logger.debug(f"Removed old file: {filepath}")
        
        logger.info(f"Cleanup: Removed {removed_count} old files from {data_type}")
        return removed_count
    
    def get_storage_stats(self) -> Dict:
        """Get storage statistics."""
        stats = {}
        
        for data_type, base_dir in [
            ('raw', self.raw_dir),
            ('processed', self.processed_dir),
            ('reference', self.reference_dir)
        ]:
            if not base_dir.exists():
                stats[data_type] = {'tickers': 0, 'files': 0, 'size_mb': 0}
                continue
            
            ticker_count = len([d for d in base_dir.iterdir() if d.is_dir()])
            
            # Count files and calculate size
            file_count = 0
            total_size = 0
            for filepath in base_dir.rglob("*.parquet"):
                file_count += 1
                total_size += filepath.stat().st_size
            
            stats[data_type] = {
                'tickers': ticker_count,
                'files': file_count,
                'size_mb': total_size / (1024 * 1024)
            }
        
        return stats
