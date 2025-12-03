"""
BACKUP TO GOOGLE DRIVE
======================
Zips the entire project and provides instructions for uploading to Google Drive.
"""

import os
import shutil
import datetime

def create_backup():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"quantum_ai_trader_backup_{timestamp}"
    source_dir = os.getcwd()
    output_filename = os.path.join(source_dir, backup_name)
    
    print(f"📦 Creating backup: {backup_name}.zip ...")
    
    # Create zip archive
    shutil.make_archive(output_filename, 'zip', source_dir)
    
    print(f"✅ Backup created: {output_filename}.zip")
    print("\n" + "="*60)
    print("INSTRUCTIONS TO SAVE TO GOOGLE DRIVE")
    print("="*60)
    print("1. If you are in Colab:")
    print("   from google.colab import drive")
    print("   drive.mount('/content/drive')")
    print(f"   !cp {backup_name}.zip /content/drive/MyDrive/")
    print("\n2. If you are in VS Code / Local:")
    print(f"   Manually upload {backup_name}.zip to your Google Drive.")
    print("="*60)

if __name__ == "__main__":
    create_backup()
