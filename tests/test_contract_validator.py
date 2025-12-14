"""
Happy-path tests for contract validator.
Tests that golden artifacts pass validation.
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pattern_discovery_lab.contract_validator import ContractValidator


# Paths to golden artifacts
GOLDEN_DIR = Path(__file__).parent / "golden_artifacts"
GOLDEN_STDOUT = GOLDEN_DIR / "stdout_canonical.txt"
GOLDEN_JSON = GOLDEN_DIR / "results_canonical.json"


class TestGoldenArtifacts:
    """Test that golden artifacts pass validation."""
    
    def test_golden_stdout_valid(self):
        """Golden stdout should pass grammar validation."""
        validator = ContractValidator(strict_mode=True)
        validator._validate_stdout_grammar(str(GOLDEN_STDOUT))
        assert len(validator.errors) == 0, f"Golden stdout has errors: {validator.errors}"
    
    def test_golden_json_valid(self):
        """Golden results.json should pass schema validation."""
        validator = ContractValidator(strict_mode=True)
        data = validator._validate_results_json(str(GOLDEN_JSON))
        assert data is not None, "Failed to load golden JSON"
        assert len(validator.errors) == 0, f"Golden JSON has errors: {validator.errors}"
    
    def test_golden_logic_consistent(self):
        """Golden artifacts should pass logic consistency checks."""
        validator = ContractValidator(strict_mode=True)
        data = validator._validate_results_json(str(GOLDEN_JSON))
        assert data is not None
        validator._validate_logic_consistency(data)
        assert len(validator.errors) == 0, f"Golden logic inconsistent: {validator.errors}"
    
    def test_golden_empty_stderr(self):
        """Empty stderr should pass validation."""
        validator = ContractValidator(strict_mode=True)
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            stderr_path = f.name
        
        try:
            validator._validate_stderr(stderr_path)
            assert len(validator.errors) == 0, f"Empty stderr failed: {validator.errors}"
        finally:
            os.unlink(stderr_path)
    
    def test_full_validation_pass(self):
        """Full validation of golden artifacts should pass."""
        # Create empty stderr
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            stderr_path = f.name
        
        try:
            validator = ContractValidator(strict_mode=True)
            success = validator.validate_all(
                stdout_path=str(GOLDEN_STDOUT),
                results_json_path=str(GOLDEN_JSON),
                stderr_path=stderr_path
            )
            
            if not success:
                print(validator.report())
            
            assert success, f"Golden artifacts failed validation: {validator.errors}"
        finally:
            os.unlink(stderr_path)
    
    def test_hash_validation_self_match(self):
        """Hashes should match when comparing golden to itself."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            stderr_path = f.name
        
        try:
            validator = ContractValidator(strict_mode=True)
            success = validator.validate_all(
                stdout_path=str(GOLDEN_STDOUT),
                results_json_path=str(GOLDEN_JSON),
                stderr_path=stderr_path,
                golden_stdout_path=str(GOLDEN_STDOUT),
                golden_json_path=str(GOLDEN_JSON),
                check_hashes=True
            )
            assert success, f"Golden self-hash check failed: {validator.errors}"
        finally:
            os.unlink(stderr_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
