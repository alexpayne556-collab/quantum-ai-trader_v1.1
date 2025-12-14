#!/usr/bin/env python3
"""
Fix NumPy/SciPy/Scikit-learn version conflicts
Installs compatible versions for Colab/GPU training
"""
import subprocess
import sys

print("=" * 80)
print("🔧 FIXING DEPENDENCY CONFLICTS")
print("=" * 80)

# The error is: numpy 2.x incompatible with scipy/sklearn
# Solution: Downgrade numpy to 1.26.x (last stable 1.x release)

commands = [
    # Uninstall conflicting packages
    "pip uninstall -y numpy scipy scikit-learn",
    
    # Install compatible versions
    "pip install numpy==1.26.4",  # Last stable numpy 1.x
    "pip install scipy==1.11.4",   # Compatible with numpy 1.26
    "pip install scikit-learn==1.3.2",  # Compatible with numpy 1.26
    
    # Install ML packages
    "pip install xgboost==2.0.3",
    "pip install lightgbm==4.1.0",
    "pip install pandas==2.1.4",
    "pip install ta-lib-bin",  # Pre-compiled TA-Lib for Linux
    
    # Install other dependencies
    "pip install yfinance==0.2.33",
    "pip install requests==2.31.0",
    "pip install python-dotenv==1.0.0",
    "pip install matplotlib==3.8.2",
    "pip install seaborn==0.13.0",
]

for i, cmd in enumerate(commands, 1):
    print(f"\n[{i}/{len(commands)}] {cmd}")
    print("-" * 80)
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"⚠️  Command failed: {cmd}")
    else:
        print(f"✅ Success")

print("\n" + "=" * 80)
print("🧪 TESTING IMPORTS")
print("=" * 80)

test_imports = [
    "import numpy",
    "import scipy",
    "import sklearn",
    "import pandas",
    "import xgboost",
    "import lightgbm",
    "import talib",
    "import yfinance",
]

failed = []
for imp in test_imports:
    try:
        exec(imp)
        module = imp.split()[1]
        version = eval(f"{module}.__version__")
        print(f"✅ {module:15s} v{version}")
    except Exception as e:
        failed.append((imp, str(e)))
        print(f"❌ {imp:25s} FAILED: {str(e)[:50]}")

print("\n" + "=" * 80)
if not failed:
    print("🎉 ALL DEPENDENCIES INSTALLED SUCCESSFULLY!")
    print("✅ Ready for 8-hour training session")
else:
    print(f"⚠️  {len(failed)} imports failed:")
    for imp, err in failed:
        print(f"   • {imp}: {err}")
print("=" * 80)
