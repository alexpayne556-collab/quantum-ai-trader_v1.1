"""
COLAB PRO Deployment Script - Capital Ready
Deploys quantum AI trading system with all safety features to Google Colab
"""

import subprocess
import sys
import os
from pathlib import Path

class ColabDeployer:
    """Deploy capital-ready trading system to COLAB PRO"""
    
    def __init__(self):
        self.required_files = [
            'risk_manager.py',
            'market_regime_manager.py',
            'backtest_validator.py',
            'forecast_engine.py',
            'elliott_wave_detector.py',
            'trade_executor.py',
            'safe_indicators.py',
            'config.py',
            'logging_config.py'
        ]
        
        self.required_packages = [
            'yfinance',
            'pandas>=2.0.0',
            'numpy',
            'scikit-learn',
            'joblib',
            'python-dotenv',
            'plotly'
        ]
    
    def verify_environment(self) -> bool:
        """Verify COLAB environment and dependencies"""
        print("🔍 Verifying COLAB environment...")
        
        # Check if running in COLAB
        try:
            import google.colab
            print("✓ Running in Google Colab")
        except ImportError:
            print("⚠️  Not running in Google Colab - this script is designed for Colab Pro")
            return False
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️  No GPU detected - training will be slower")
        except ImportError:
            print("ℹ️  PyTorch not installed - GPU check skipped")
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install required packages"""
        print("\n📦 Installing dependencies...")
        
        for package in self.required_packages:
            try:
                print(f"   Installing {package}...")
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', '-q', package
                ])
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                return False
        
        print("✓ All dependencies installed")
        return True
    
    def verify_files(self) -> bool:
        """Verify all required files are present"""
        print("\n📋 Verifying required files...")
        
        missing_files = []
        for file in self.required_files:
            if not Path(file).exists():
                missing_files.append(file)
                print(f"❌ Missing: {file}")
            else:
                print(f"✓ Found: {file}")
        
        if missing_files:
            print(f"\n⚠️  {len(missing_files)} files missing!")
            print("   Upload these files to Colab before deployment")
            return False
        
        return True
    
    def mount_drive(self) -> bool:
        """Mount Google Drive for persistent storage"""
        print("\n💾 Mounting Google Drive...")
        
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)
            
            # Create directory structure
            base_path = Path('/content/drive/MyDrive/quantum_trader')
            (base_path / 'models').mkdir(parents=True, exist_ok=True)
            (base_path / 'data').mkdir(parents=True, exist_ok=True)
            (base_path / 'logs').mkdir(parents=True, exist_ok=True)
            (base_path / 'backtest_results').mkdir(parents=True, exist_ok=True)
            
            print(f"✓ Drive mounted at: {base_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to mount drive: {e}")
            return False
    
    def test_risk_manager(self) -> bool:
        """Test risk management system"""
        print("\n🛡️ Testing Risk Manager...")
        
        try:
            from risk_manager import RiskManager, MarketRegime
            
            # Initialize with test capital
            rm = RiskManager(initial_capital=10000.0)
            
            # Test CRASH regime
            can_trade = rm.can_trade(MarketRegime.CRASH)
            shares, risk = rm.calculate_position_size(100.0, 95.0, MarketRegime.CRASH)
            
            assert can_trade == False, "CRASH regime should block trading"
            assert shares == 0, "CRASH regime should return 0 shares"
            
            # Test BULL regime
            can_trade = rm.can_trade(MarketRegime.BULL)
            shares, risk = rm.calculate_position_size(100.0, 95.0, MarketRegime.BULL)
            
            assert can_trade == True, "BULL regime should allow trading"
            assert shares > 0, "BULL regime should calculate valid position size"
            
            print("✓ Risk Manager tests passed")
            return True
            
        except Exception as e:
            print(f"❌ Risk Manager test failed: {e}")
            return False
    
    def test_market_regime(self) -> bool:
        """Test market regime detection"""
        print("\n📊 Testing Market Regime Manager...")
        
        try:
            from market_regime_manager import MarketRegimeManager
            
            manager = MarketRegimeManager()
            regime_info = manager.get_current_regime()
            halt_info = manager.enforce_trading_halt()
            
            assert 'regime' in regime_info, "Missing regime in response"
            assert 'should_halt' in halt_info, "Missing halt status"
            
            print(f"✓ Current regime: {regime_info['regime']}")
            print(f"✓ Trading halted: {halt_info['should_halt']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Market Regime test failed: {e}")
            return False
    
    def test_backtest(self) -> bool:
        """Test backtest validator"""
        print("\n📈 Testing Backtest Validator...")
        
        try:
            from backtest_validator import BacktestValidator
            
            validator = BacktestValidator(initial_capital=10000.0)
            
            # Quick test with one symbol
            print("   Running quick backtest on SPY...")
            metrics = validator.run_full_backtest(['SPY'])
            
            print(f"✓ Backtest complete - {metrics['total_trades']} trades")
            print(f"   Win rate: {metrics['win_rate']:.1f}%")
            print(f"   Sharpe: {metrics['sharpe_ratio']:.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Backtest test failed: {e}")
            return False
    
    def run_deployment_checklist(self) -> dict:
        """Run complete deployment checklist"""
        print("=" * 70)
        print("🚀 COLAB PRO DEPLOYMENT CHECKLIST")
        print("=" * 70)
        
        results = {}
        
        # Step 1: Environment verification
        results['environment'] = self.verify_environment()
        if not results['environment']:
            print("\n❌ Environment verification failed - stopping deployment")
            return results
        
        # Step 2: Install dependencies
        results['dependencies'] = self.install_dependencies()
        if not results['dependencies']:
            print("\n❌ Dependency installation failed - stopping deployment")
            return results
        
        # Step 3: Mount Google Drive
        results['drive'] = self.mount_drive()
        
        # Step 4: Verify files
        results['files'] = self.verify_files()
        if not results['files']:
            print("\n❌ File verification failed - upload missing files")
            return results
        
        # Step 5: Test risk manager
        results['risk_manager'] = self.test_risk_manager()
        
        # Step 6: Test market regime
        results['market_regime'] = self.test_market_regime()
        
        # Step 7: Test backtest
        results['backtest'] = self.test_backtest()
        
        return results
    
    def print_deployment_summary(self, results: dict):
        """Print deployment summary"""
        print("\n" + "=" * 70)
        print("📊 DEPLOYMENT SUMMARY")
        print("=" * 70)
        
        total_checks = len(results)
        passed_checks = sum(1 for v in results.values() if v)
        
        print(f"\nChecks Passed: {passed_checks}/{total_checks}")
        print("\nDetailed Results:")
        
        for check, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {check.replace('_', ' ').title()}: {status}")
        
        if passed_checks == total_checks:
            print("\n" + "=" * 70)
            print("✅ ALL CHECKS PASSED - SYSTEM READY FOR CAPITAL DEPLOYMENT")
            print("=" * 70)
            print("\n🎯 Next Steps:")
            print("   1. Configure API keys in .env file")
            print("   2. Run paper trading for 1 week")
            print("   3. Verify Sharpe > 1.0, Win rate > 45%")
            print("   4. Start with $500 live capital")
            print("   5. Scale gradually based on performance")
        else:
            print("\n" + "=" * 70)
            print("⚠️  DEPLOYMENT NOT READY - FIX FAILED CHECKS")
            print("=" * 70)


def deploy_to_colab():
    """Main deployment function"""
    deployer = ColabDeployer()
    results = deployer.run_deployment_checklist()
    deployer.print_deployment_summary(results)
    return results


if __name__ == "__main__":
    # Run deployment
    results = deploy_to_colab()
    
    # Exit with appropriate code (only when run as script, not in notebook)
    if all(results.values()):
        print("\n✓ Deployment script completed successfully")
        # Don't call sys.exit in Jupyter notebooks
        try:
            # Check if running in IPython/Jupyter
            get_ipython()  # This exists in Jupyter
            pass  # Don't exit in notebook
        except NameError:
            sys.exit(0)  # Only exit in terminal
    else:
        print("\n❌ Deployment script completed with errors")
        try:
            get_ipython()
            pass
        except NameError:
            sys.exit(1)
