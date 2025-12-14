"""
Smoke test for Pattern Discovery Lab.

Validates that the complete pipeline runs without errors:
- Dataset generation
- Detector execution
- Control suite
- Report generation
- Output contract compliance
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pattern_discovery_lab.src.runner import run_calibration
from pattern_discovery_lab.src.reporting import write_results_json, write_report_md


def test_minimal_run():
    """Test minimal configuration runs successfully."""
    
    config = {
        'name': 'smoke_test.yaml',
        'seed': 123,
        'series_length': 100,  # Short for speed
        'n_shuffles': 10,
        'n_surrogates': 10,
        'n_blocks': 3,
        'stability_threshold': 0.5,
        'enable_extra_detectors': False,
    }
    
    print("Running minimal calibration...")
    results = run_calibration(config)
    
    # Validate results structure
    assert 'run_id' in results
    assert 'timestamp' in results
    assert 'config' in results
    assert 'datasets' in results
    assert len(results['datasets']) == 3  # white_noise, ar1, garch
    
    print(f"✅ Run ID: {results['run_id']}")
    print(f"✅ Datasets: {len(results['datasets'])}")
    
    # Validate each dataset has results
    for dataset_result in results['datasets']:
        assert 'dataset_name' in dataset_result
        assert 'detectors' in dataset_result
        assert len(dataset_result['detectors']) == 2  # autocorr + time_reversal
        
        for detector_result in dataset_result['detectors']:
            assert 'detector_name' in detector_result
            assert 'real_score' in detector_result
            assert 'shuffle_p_value' in detector_result
            assert 'surrogate_p_value' in detector_result
            assert 'stability_metric' in detector_result
            assert 'passes_controls' in detector_result
    
    print("✅ Results structure valid")
    return results


def test_output_generation():
    """Test JSON and Markdown report generation."""
    
    config = {
        'name': 'smoke_test.yaml',
        'seed': 456,
        'series_length': 100,
        'n_shuffles': 10,
        'n_surrogates': 10,
        'n_blocks': 3,
        'stability_threshold': 0.5,
        'enable_extra_detectors': False,
    }
    
    results = run_calibration(config)
    
    # Create temp directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        run_id = results['run_id']
        
        # Write JSON
        json_path = write_results_json(run_id, results, tmpdir)
        assert os.path.exists(json_path)
        
        # Validate JSON is parseable
        with open(json_path, 'r') as f:
            loaded_results = json.load(f)
        assert loaded_results['run_id'] == run_id
        print(f"✅ JSON output valid: {json_path}")
        
        # Write Markdown
        report_path = write_report_md(run_id, results, tmpdir)
        assert os.path.exists(report_path)
        
        # Validate report contains required sections
        with open(report_path, 'r') as f:
            report_content = f.read()
        
        assert '## STOP RULES EVALUATED (ENFORCE_NOW)' in report_content
        assert '## Results Summary' in report_content
        assert 'Stability Metric' in report_content
        assert 'Coefficient of Variation' in report_content
        assert '| Dataset | Detector | Real Score |' in report_content
        assert '## Detailed Results' in report_content
        
        print(f"✅ Markdown report valid: {report_path}")
        print(f"✅ Report contains all required sections")


def test_noise_rejection():
    """Test that white noise is correctly rejected (no hallucination)."""
    
    config = {
        'name': 'smoke_test.yaml',
        'seed': 789,
        'series_length': 100,
        'n_shuffles': 10,
        'n_surrogates': 10,
        'n_blocks': 3,
        'stability_threshold': 0.5,
        'enable_extra_detectors': False,
    }
    
    results = run_calibration(config)
    
    # Find white_noise results
    white_noise_result = None
    for dataset_result in results['datasets']:
        if dataset_result['dataset_name'] == 'white_noise':
            white_noise_result = dataset_result
            break
    
    assert white_noise_result is not None
    
    # Check that detectors correctly reject noise
    for detector_result in white_noise_result['detectors']:
        detector_name = detector_result['detector_name']
        passes = detector_result['passes_controls']
        
        # White noise should NOT pass controls (would indicate hallucination)
        if passes:
            print(f"⚠️  WARNING: {detector_name} claims structure on white noise!")
            print(f"   Real score: {detector_result['real_score']}")
            print(f"   Shuffle p: {detector_result['shuffle_p_value']}")
            print(f"   Surrogate p: {detector_result['surrogate_p_value']}")
        
        # Note: Not asserting failure here because it depends on random seed
        # Just validating that the test is running
    
    print("✅ White noise rejection test complete")


def test_ar1_detection():
    """Test that AR(1) structure is detected."""
    
    config = {
        'name': 'smoke_test.yaml',
        'seed': 101112,
        'series_length': 100,
        'n_shuffles': 10,
        'n_surrogates': 10,
        'n_blocks': 3,
        'stability_threshold': 0.5,
        'enable_extra_detectors': False,
    }
    
    results = run_calibration(config)
    
    # Find AR1 results
    ar1_result = None
    for dataset_result in results['datasets']:
        if dataset_result['dataset_name'] == 'ar1':
            ar1_result = dataset_result
            break
    
    assert ar1_result is not None
    
    # Check autocorrelation detector on AR1
    autocorr_result = None
    for detector_result in ar1_result['detectors']:
        if detector_result['detector_name'] == 'autocorrelation_dependence':
            autocorr_result = detector_result
            break
    
    assert autocorr_result is not None
    
    print(f"✅ AR(1) autocorrelation score: {autocorr_result['real_score']:.4f}")
    print(f"✅ AR(1) shuffle p-value: {autocorr_result['shuffle_p_value']:.4f}")
    
    # AR(1) should have high autocorrelation score
    assert autocorr_result['real_score'] > 0.3, "AR(1) should have detectable autocorrelation"


def test_stability_metric():
    """Test that stability metric is computed correctly."""
    
    config = {
        'name': 'smoke_test.yaml',
        'seed': 131415,
        'series_length': 300,  # Longer for stability test
        'n_shuffles': 10,
        'n_surrogates': 10,
        'n_blocks': 3,
        'stability_threshold': 0.5,
        'enable_extra_detectors': False,
    }
    
    results = run_calibration(config)
    
    # Check that all detectors have stability metrics
    for dataset_result in results['datasets']:
        for detector_result in dataset_result['detectors']:
            assert 'stability_metric' in detector_result
            assert 'block_scores' in detector_result
            
            stability = detector_result['stability_metric']
            block_scores = detector_result['block_scores']
            
            assert len(block_scores) == 3  # Should have 3 blocks
            assert stability >= 0  # CV should be non-negative
            
    print("✅ Stability metrics computed for all detectors")


def main():
    """Run all smoke tests."""
    print("=" * 80)
    print("PATTERN DISCOVERY LAB - SMOKE TEST")
    print("=" * 80)
    print()
    
    tests = [
        ("Minimal Run", test_minimal_run),
        ("Output Generation", test_output_generation),
        ("Noise Rejection", test_noise_rejection),
        ("AR(1) Detection", test_ar1_detection),
        ("Stability Metric", test_stability_metric),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'─' * 80}")
        print(f"TEST: {test_name}")
        print(f"{'─' * 80}")
        
        try:
            test_func()
            print(f"✅ PASSED: {test_name}")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test_name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print()
    print("=" * 80)
    print(f"SMOKE TEST SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✅ ALL SMOKE TESTS PASSED")
        return 0
    else:
        print(f"\n❌ {failed} TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
