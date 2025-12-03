"""
Upload Local Models to Google Drive (for Colab)

This script will be embedded in the Colab notebook to automatically
upload your locally-trained models to Google Drive.
"""

import os
import shutil
from pathlib import Path

def upload_local_models_to_drive():
    """
    Upload all locally trained models to Google Drive
    (This function runs inside Google Colab)
    """
    
    # Define paths
    local_base = Path('/content/quantum_trader_local')
    drive_base = Path('/content/drive/MyDrive/quantum_trader')
    
    # Files to upload
    files_to_upload = {
        'trained_models/pattern_stats.db': 'models/pattern_stats.db',
        'trained_models/training_logs.db': 'models/training_logs.db',
        'trained_models/quantile_models/': 'models/quantile_models/',
        'training_results/weight_optimization_recommendations.json': 'results/weight_optimization_recommendations.json',
        'training_results/quantile_forecaster_results.json': 'results/quantile_forecaster_results.json',
    }
    
    print("📤 Uploading local models to Google Drive...")
    print("="*80)
    
    for src_path, dst_path in files_to_upload.items():
        src = local_base / src_path
        dst = drive_base / dst_path
        
        if not src.exists():
            print(f"⚠️  Skipping {src_path} (not found)")
            continue
        
        # Create destination directory
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        if src.is_dir():
            # Copy directory
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"✅ Uploaded directory: {src_path}")
        else:
            # Copy file
            shutil.copy2(src, dst)
            size_mb = src.stat().st_size / (1024 * 1024)
            print(f"✅ Uploaded {src_path} ({size_mb:.2f} MB)")
    
    print("="*80)
    print("✅ All local models uploaded to Google Drive!")
    print(f"\n📁 Files available at: {drive_base}")

if __name__ == "__main__":
    upload_local_models_to_drive()
