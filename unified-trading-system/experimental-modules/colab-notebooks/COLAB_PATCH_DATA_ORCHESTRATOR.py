"""
🔧 PATCH DATA_ORCHESTRATOR - Add Missing Methods
===============================================
Run this in Colab to add the missing methods to your data_orchestrator.py
"""

import sys
from pathlib import Path

print("="*80)
print("🔧 PATCHING DATA_ORCHESTRATOR")
print("="*80)

# Setup paths
BASE_DIR = Path('/content/drive/MyDrive/QuantumAI')
MODULES_DIR = BASE_DIR / 'backend' / 'modules'
data_orch_file = MODULES_DIR / 'data_orchestrator.py'

if not data_orch_file.exists():
    print(f"❌ File not found: {data_orch_file}")
    sys.exit(1)

print(f"\n📁 File: {data_orch_file}")
print(f"📊 Size: {data_orch_file.stat().st_size:,} bytes")

# Read file
print("\n📖 Reading file...")
with open(data_orch_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Check if already patched
if 'def get_returns(self, df: pd.DataFrame, period: int = 1) -> float:' in content:
    if 'class DataOrchestrator(DataOrchestrator_v84):' in content:
        # Check if methods are actually in the class
        class_start = content.find('class DataOrchestrator(DataOrchestrator_v84):')
        if class_start != -1:
            class_end = content.find('\nclass ', class_start + 1)
            if class_end == -1:
                class_end = len(content)
            
            class_content = content[class_start:class_end]
            
            if 'def get_returns' in class_content:
                print("\n✅ File already has get_returns method!")
                print("   But it's not being found - this might be a caching issue.")
                print("\n💡 SOLUTION: Restart runtime and re-run this script")
                sys.exit(0)

print("\n🔧 Adding missing methods...")

# Find where to insert the methods
# Look for the DataOrchestrator class
if 'class DataOrchestrator' not in content:
    print("❌ DataOrchestrator class not found!")
    print("   The file structure is different than expected.")
    sys.exit(1)

# Find the end of DataOrchestrator class (before next class or end of file)
class_start = content.find('class DataOrchestrator')
if class_start == -1:
    print("❌ Could not find DataOrchestrator class")
    sys.exit(1)

# Find the __init__ method
init_pos = content.find('def __init__(self):', class_start)
if init_pos == -1:
    print("❌ Could not find __init__ method")
    sys.exit(1)

# Find end of __init__ (next def or class)
init_end = content.find('\n    def ', init_pos + 1)
if init_end == -1:
    init_end = content.find('\nclass ', init_pos + 1)
if init_end == -1:
    init_end = len(content)

# Check if ScalarExtractor class exists
if 'class ScalarExtractor:' not in content:
    print("\n📝 Adding ScalarExtractor class...")
    
    # Find a good place to insert it (before DataOrchestrator)
    insert_pos = content.find('class DataOrchestrator')
    
    scalar_extractor_code = '''

# ============================================================
# ✅ SCALAR EXTRACTOR (Prevents Series comparison errors)
# ============================================================

class ScalarExtractor:
    """
    Utility class for safely extracting scalar values from pandas Series/DataFrames.
    Prevents "The truth value of a Series is ambiguous" errors.
    """
    
    @staticmethod
    def to_scalar(value) -> float:
        """Convert any value (Series, array, etc.) to float scalar"""
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, pd.Series):
            if len(value) == 0:
                return 0.0
            return float(value.iloc[0] if hasattr(value, 'iloc') else value.values[0])
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return 0.0
            return float(value.flat[0])
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return 0.0
            return float(value[0])
        try:
            return float(value)
        except:
            return 0.0
    
    @staticmethod
    def clean_dict(data: dict) -> dict:
        """Clean all values in dict to scalars"""
        result = {}
        for k, v in data.items():
            if isinstance(v, dict):
                result[k] = ScalarExtractor.clean_dict(v)
            elif isinstance(v, (list, tuple)):
                result[k] = [ScalarExtractor.to_scalar(x) if not isinstance(x, dict) else x for x in v]
            else:
                result[k] = ScalarExtractor.to_scalar(v)
        return result


'''
    
    content = content[:insert_pos] + scalar_extractor_code + content[insert_pos:]
    print("✅ ScalarExtractor class added")

# Now add methods to DataOrchestrator class
print("\n📝 Adding methods to DataOrchestrator class...")

# Find the end of DataOrchestrator class
class_end = content.find('\nclass ', class_start + 1)
if class_end == -1:
    class_end = len(content)

# Check if methods already exist in the class
class_content = content[class_start:class_end]

if 'def get_returns' not in class_content:
    # Find where to insert (after __init__ or at end of class)
    insert_pos = class_end
    
    # Try to find a better insertion point (after last method)
    last_method = class_content.rfind('\n    def ')
    if last_method != -1:
        # Find end of that method
        method_end = class_content.find('\n    def ', last_method + 1)
        if method_end == -1:
            method_end = len(class_content)
        # Find the closing of that method (next def, class, or end)
        insert_pos = class_start + method_end
    
    methods_code = '''
    
    # ============================================================
    # SCALAR UTILITY METHODS (Prevent Series comparison errors)
    # ============================================================
    
    def to_scalar(self, value):
        """Convert any value to scalar (Series, array, etc.)"""
        return ScalarExtractor.to_scalar(value)
    
    def get_price(self, df: pd.DataFrame, column: str = 'close', index: int = -1) -> float:
        """
        Get price as scalar from dataframe.
        Handles uppercase/lowercase column names.
        """
        try:
            # Try lowercase
            col = column.lower()
            if col not in df.columns:
                # Try uppercase
                col = column.upper()
            if col not in df.columns:
                # Try capitalize
                col = column.capitalize()
            
            value = df[col].iloc[index]
            return self.to_scalar(value)
        except:
            return 0.0
    
    def get_last_close(self, df: pd.DataFrame) -> float:
        """Get last close price as scalar"""
        return self.get_price(df, 'close', -1)
    
    def get_last_open(self, df: pd.DataFrame) -> float:
        """Get last open price as scalar"""
        return self.get_price(df, 'open', -1)
    
    def get_last_high(self, df: pd.DataFrame) -> float:
        """Get last high price as scalar"""
        return self.get_price(df, 'high', -1)
    
    def get_last_low(self, df: pd.DataFrame) -> float:
        """Get last low price as scalar"""
        return self.get_price(df, 'low', -1)
    
    def get_last_volume(self, df: pd.DataFrame) -> float:
        """Get last volume as scalar"""
        return self.get_price(df, 'volume', -1)
    
    def get_volume_ratio(self, df: pd.DataFrame, period: int = 20) -> float:
        """
        Get volume ratio (current / average) as scalar.
        
        Args:
            df: OHLCV dataframe
            period: Period for average calculation
            
        Returns:
            Volume ratio as scalar (1.0 = average volume)
        """
        if len(df) < period:
            return 1.0
        
        current_vol = self.get_last_volume(df)
        
        # Get average volume
        vol_col = 'volume'
        if vol_col not in df.columns:
            vol_col = 'Volume'
        
        avg_vol = df[vol_col].iloc[-period:-1].mean()
        avg_vol = self.to_scalar(avg_vol)
        
        if avg_vol > 0:
            return float(current_vol / avg_vol)
        return 1.0
    
    def get_returns(self, df: pd.DataFrame, period: int = 1) -> float:
        """
        Get returns over period as scalar.
        
        Args:
            df: OHLCV dataframe
            period: Number of periods back
            
        Returns:
            Return as decimal (0.05 = 5% gain)
        """
        if len(df) < period + 1:
            return 0.0
        
        current = self.get_last_close(df)
        previous = self.get_price(df, 'close', -(period + 1))
        
        if previous > 0:
            return float((current - previous) / previous)
        return 0.0
    
    def get_ma(self, df: pd.DataFrame, period: int = 20, column: str = 'close') -> float:
        """
        Get moving average as scalar.
        
        Args:
            df: OHLCV dataframe
            period: MA period
            column: Column to calculate MA on
            
        Returns:
            Moving average as scalar
        """
        if len(df) < period:
            return self.get_price(df, column, -1)
        
        col = column.lower()
        if col not in df.columns:
            col = column.upper()
        if col not in df.columns:
            col = column.capitalize()
        
        ma = df[col].rolling(period).mean().iloc[-1]
        return self.to_scalar(ma)
    
    def clean_module_output(self, output: dict) -> dict:
        """
        Clean module output to ensure all values are scalars.
        Use this after getting output from any module.
        """
        return ScalarExtractor.clean_dict(output)
'''
    
    content = content[:insert_pos] + methods_code + content[insert_pos:]
    print("✅ Methods added to DataOrchestrator class")

# Update __init__ to include scalar
if 'self.scalar = ScalarExtractor()' not in content:
    print("\n📝 Updating __init__ method...")
    
    # Find __init__
    init_start = content.find('def __init__(self):', class_start)
    if init_start != -1:
        # Find end of __init__ (next def or end of class)
        init_end = content.find('\n    def ', init_start + 1)
        if init_end == -1:
            init_end = content.find('\nclass ', init_start + 1)
        if init_end == -1:
            init_end = len(content)
        
        init_content = content[init_start:init_end]
        
        # Check if super().__init__() exists
        if 'super().__init__()' in init_content:
            # Add after super().__init__()
            super_pos = init_content.find('super().__init__()')
            insert_pos = init_start + super_pos + len('super().__init__()')
            
            # Find end of that line
            line_end = content.find('\n', insert_pos)
            
            scalar_init = '\n        self.scalar = ScalarExtractor()  # Add scalar extractor'
            content = content[:line_end] + scalar_init + content[line_end:]
            print("✅ Updated __init__ to include scalar")
        else:
            # Add at end of __init__
            content = content[:init_end] + '\n        self.scalar = ScalarExtractor()  # Add scalar extractor' + content[init_end:]
            print("✅ Added scalar to __init__")

# Write back
print("\n💾 Writing patched file...")
with open(data_orch_file, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"✅ File patched! New size: {data_orch_file.stat().st_size:,} bytes")

# Verify
print("\n🔍 Verifying patch...")
with open(data_orch_file, 'r') as f:
    new_content = f.read()

checks = {
    'ScalarExtractor class': 'class ScalarExtractor:' in new_content,
    'get_returns method': 'def get_returns(self, df: pd.DataFrame' in new_content,
    'get_ma method': 'def get_ma(self, df: pd.DataFrame' in new_content,
    'to_scalar method': 'def to_scalar(self, value)' in new_content,
    'scalar in __init__': 'self.scalar = ScalarExtractor()' in new_content,
}

print("\n📋 Verification:")
all_good = True
for check, passed in checks.items():
    if passed:
        print(f"  ✅ {check}")
    else:
        print(f"  ❌ {check}")
        all_good = False

if all_good:
    print("\n" + "="*80)
    print("✅ PATCH SUCCESSFUL!")
    print("="*80)
    print("\n🔄 NEXT STEPS:")
    print("   1. RESTART RUNTIME (Runtime → Restart runtime)")
    print("   2. Re-run your launcher:")
    print("      %run COLAB_LAUNCH_INSTITUTIONAL_SYSTEM.py")
    print("\n✅ The backtest should now work!")
else:
    print("\n⚠️  Some checks failed - the patch may be incomplete")
    print("   Try re-uploading the file from your local machine instead")

print("="*80)

