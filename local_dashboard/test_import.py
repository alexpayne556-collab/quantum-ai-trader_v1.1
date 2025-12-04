import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
try:
    import advanced_pattern_detector
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
