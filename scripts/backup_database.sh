#!/bin/bash
# Automated backup script for market data database
# Run this hourly or when needed to ensure data safety

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SOURCE="/workspaces/quantum-ai-trader_v1.1/data/market_data.db"
BACKUP_DIR="/workspaces/quantum-ai-trader_v1.1/data/backups"
BACKUP_FILE="${BACKUP_DIR}/market_data_${TIMESTAMP}.db"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Only backup if source exists
if [ -f "$SOURCE" ]; then
    # Use SQLite backup command for safe copy (handles locks)
    sqlite3 "$SOURCE" ".backup '${BACKUP_FILE}'"
    
    if [ $? -eq 0 ]; then
        SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
        echo "✓ Backup created: $BACKUP_FILE ($SIZE)"
        
        # Count backups
        NUM_BACKUPS=$(ls -1 "$BACKUP_DIR"/*.db 2>/dev/null | wc -l)
        echo "  Total backups: $NUM_BACKUPS"
        
        # Keep only last 10 backups (delete oldest)
        if [ $NUM_BACKUPS -gt 10 ]; then
            OLDEST=$(ls -t "$BACKUP_DIR"/*.db | tail -1)
            rm "$OLDEST"
            echo "  Deleted oldest backup: $(basename $OLDEST)"
        fi
    else
        echo "✗ Backup failed"
        exit 1
    fi
else
    echo "✗ Source database not found: $SOURCE"
    exit 1
fi
