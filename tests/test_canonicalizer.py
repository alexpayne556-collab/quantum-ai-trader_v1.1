"""
Tests for stdout canonicalizer.

Tests that:
1. Changing ONLY the 3 volatile run-path lines does NOT change canonical hash
2. Changing ANY other stdout content DOES change canonical hash
3. Lines that don't match exact patterns remain unchanged (no over-matching)
"""

import pytest
from pathlib import Path

from pattern_discovery_lab.canonicalizer import canonicalize_stdout, compute_canonical_hash


# Load real golden stdout
GOLDEN_STDOUT_PATH = Path(__file__).parent.parent / "pattern_discovery_lab" / "golden" / "stdout_canonical.txt"


class TestCanonicalizerInvariant:
    """Tests that changing ONLY volatile lines does NOT change canonical hash."""
    
    def test_different_run_timestamps_same_hash(self):
        """Mutating only the run_id portions should NOT change canonical hash."""
        golden_content = GOLDEN_STDOUT_PATH.read_text()
        original_hash = compute_canonical_hash(golden_content)
        
        # Mutate the timestamp in Run Folder line
        import re
        mutated = re.sub(
            r'(Run Folder: .*/pattern_discovery_lab/runs/)\d{8}_\d{6}',
            r'\g<1>99999999_999999',
            golden_content
        )
        # Mutate the timestamp in Results written to line
        mutated = re.sub(
            r'(Results written to: .*/pattern_discovery_lab/runs/)\d{8}_\d{6}(/results\.json)',
            r'\g<1>99999999_999999\g<2>',
            mutated
        )
        # Mutate the timestamp in Debug info written to line
        mutated = re.sub(
            r'(Debug info written to: .*/pattern_discovery_lab/runs/)\d{8}_\d{6}(/results_debug\.json)',
            r'\g<1>99999999_999999\g<2>',
            mutated
        )
        
        mutated_hash = compute_canonical_hash(mutated)
        
        assert original_hash == mutated_hash, \
            f"Canonical hash should be unchanged when only timestamps mutated: {original_hash} != {mutated_hash}"
    
    def test_multiple_different_timestamps_same_hash(self):
        """Different timestamp values should all canonicalize to same hash."""
        golden_content = GOLDEN_STDOUT_PATH.read_text()
        original_hash = compute_canonical_hash(golden_content)
        
        # Create variants with different timestamps
        import re
        timestamps = ['20250101_000000', '20991231_235959', '12345678_123456']
        
        for ts in timestamps:
            mutated = re.sub(
                r'(Run Folder: .*/pattern_discovery_lab/runs/)\d{8}_\d{6}',
                rf'\g<1>{ts}',
                golden_content
            )
            mutated = re.sub(
                r'(Results written to: .*/pattern_discovery_lab/runs/)\d{8}_\d{6}(/results\.json)',
                rf'\g<1>{ts}\g<2>',
                mutated
            )
            mutated = re.sub(
                r'(Debug info written to: .*/pattern_discovery_lab/runs/)\d{8}_\d{6}(/results_debug\.json)',
                rf'\g<1>{ts}\g<2>',
                mutated
            )
            
            mutated_hash = compute_canonical_hash(mutated)
            assert original_hash == mutated_hash, f"Hash mismatch for timestamp {ts}"


class TestCanonicalizerStrict:
    """Tests that changing NON-volatile content DOES change canonical hash."""
    
    def test_changing_verdict_changes_hash(self):
        """Changing PASS to HALT must change canonical hash."""
        golden_content = GOLDEN_STDOUT_PATH.read_text()
        original_hash = compute_canonical_hash(golden_content)
        
        # Change a verdict
        mutated = golden_content.replace('Overall Status: PASS', 'Overall Status: HALT', 1)
        mutated_hash = compute_canonical_hash(mutated)
        
        assert original_hash != mutated_hash, \
            "Canonical hash MUST change when verdict is modified"
    
    def test_adding_blank_line_changes_hash(self):
        """Adding a blank line must change canonical hash."""
        golden_content = GOLDEN_STDOUT_PATH.read_text()
        original_hash = compute_canonical_hash(golden_content)
        
        # Add a blank line somewhere
        lines = golden_content.split('\n')
        lines.insert(10, '')
        mutated = '\n'.join(lines)
        mutated_hash = compute_canonical_hash(mutated)
        
        assert original_hash != mutated_hash, \
            "Canonical hash MUST change when blank line is added"
    
    def test_changing_pvalue_changes_hash(self):
        """Changing a p-value must change canonical hash."""
        golden_content = GOLDEN_STDOUT_PATH.read_text()
        original_hash = compute_canonical_hash(golden_content)
        
        # Change a p-value
        mutated = golden_content.replace('=0.8200', '=0.9999', 1)
        mutated_hash = compute_canonical_hash(mutated)
        
        assert original_hash != mutated_hash, \
            "Canonical hash MUST change when p-value is modified"
    
    def test_changing_gate_label_changes_hash(self):
        """Changing a gate label must change canonical hash."""
        golden_content = GOLDEN_STDOUT_PATH.read_text()
        original_hash = compute_canonical_hash(golden_content)
        
        # Change a gate label
        mutated = golden_content.replace('Gate A —', 'Gate X —', 1)
        mutated_hash = compute_canonical_hash(mutated)
        
        assert original_hash != mutated_hash, \
            "Canonical hash MUST change when gate label is modified"
    
    def test_removing_character_changes_hash(self):
        """Removing any character must change canonical hash."""
        golden_content = GOLDEN_STDOUT_PATH.read_text()
        original_hash = compute_canonical_hash(golden_content)
        
        # Remove a character from middle of file
        midpoint = len(golden_content) // 2
        mutated = golden_content[:midpoint] + golden_content[midpoint + 1:]
        mutated_hash = compute_canonical_hash(mutated)
        
        assert original_hash != mutated_hash, \
            "Canonical hash MUST change when character is removed"


class TestCanonicalizerNoOvermatch:
    """Tests that canonicalizer doesn't match lines it shouldn't."""
    
    def test_run_folder_without_pattern_discovery_lab_unchanged(self):
        """Line with 'Run Folder:' but different path must remain unchanged."""
        text = "Run Folder: /some/other/path/runs/20251213_123456"
        canonicalized = canonicalize_stdout(text)
        assert canonicalized == text, \
            "Line without /pattern_discovery_lab/runs/ should remain unchanged"
    
    def test_run_folder_missing_colon_unchanged(self):
        """Line 'Run Folder /missing/colon' must remain unchanged."""
        text = "Run Folder /workspaces/quantum-ai-trader_v1.1/pattern_discovery_lab/runs/20251213_123456"
        canonicalized = canonicalize_stdout(text)
        assert canonicalized == text, \
            "Line without colon after 'Run Folder' should remain unchanged"
    
    def test_partial_timestamp_unchanged(self):
        """Partial timestamps (not YYYYMMDD_HHMMSS) must remain unchanged."""
        text = "Run Folder: /workspaces/quantum-ai-trader_v1.1/pattern_discovery_lab/runs/12345"
        canonicalized = canonicalize_stdout(text)
        assert canonicalized == text, \
            "Partial timestamp should remain unchanged"
    
    def test_results_line_wrong_filename_unchanged(self):
        """Results line with wrong filename must remain unchanged."""
        text = "Results written to: /workspaces/quantum-ai-trader_v1.1/pattern_discovery_lab/runs/20251213_123456/other.json"
        canonicalized = canonicalize_stdout(text)
        assert canonicalized == text, \
            "Results line with wrong filename should remain unchanged"
    
    def test_random_text_with_runs_unchanged(self):
        """Random text containing '/runs/' should remain unchanged."""
        text = "Some text about /runs/20251213_123456 in the middle"
        canonicalized = canonicalize_stdout(text)
        assert canonicalized == text, \
            "Random text with /runs/ should remain unchanged"
    
    def test_preserves_trailing_whitespace(self):
        """Trailing whitespace on volatile lines must be preserved."""
        text = "Run Folder: /workspaces/quantum-ai-trader_v1.1/pattern_discovery_lab/runs/20251213_123456   \n"
        canonicalized = canonicalize_stdout(text)
        assert canonicalized == "Run Folder: /workspaces/quantum-ai-trader_v1.1/pattern_discovery_lab/runs/<RUN_ID>   \n", \
            "Trailing whitespace must be preserved"


class TestCanonicalizerCorrectReplacement:
    """Tests that canonicalizer replaces correctly."""
    
    def test_run_folder_line_canonicalized(self):
        """Run Folder line should be canonicalized correctly."""
        text = "Run Folder: /workspaces/quantum-ai-trader_v1.1/pattern_discovery_lab/runs/20251213_123456"
        expected = "Run Folder: /workspaces/quantum-ai-trader_v1.1/pattern_discovery_lab/runs/<RUN_ID>"
        canonicalized = canonicalize_stdout(text)
        assert canonicalized == expected
    
    def test_results_line_canonicalized(self):
        """Results written to line should be canonicalized correctly."""
        text = "Results written to: /workspaces/quantum-ai-trader_v1.1/pattern_discovery_lab/runs/20251213_123456/results.json"
        expected = "Results written to: /workspaces/quantum-ai-trader_v1.1/pattern_discovery_lab/runs/<RUN_ID>/results.json"
        canonicalized = canonicalize_stdout(text)
        assert canonicalized == expected
    
    def test_debug_line_canonicalized(self):
        """Debug info written to line should be canonicalized correctly."""
        text = "Debug info written to: /workspaces/quantum-ai-trader_v1.1/pattern_discovery_lab/runs/20251213_123456/results_debug.json"
        expected = "Debug info written to: /workspaces/quantum-ai-trader_v1.1/pattern_discovery_lab/runs/<RUN_ID>/results_debug.json"
        canonicalized = canonicalize_stdout(text)
        assert canonicalized == expected
    
    def test_full_stdout_only_3_lines_changed(self):
        """Only exactly 3 lines should be changed in full stdout."""
        golden_content = GOLDEN_STDOUT_PATH.read_text()
        canonicalized = canonicalize_stdout(golden_content)
        
        original_lines = golden_content.split('\n')
        canonical_lines = canonicalized.split('\n')
        
        assert len(original_lines) == len(canonical_lines), \
            "Number of lines should not change"
        
        changed_count = sum(1 for o, c in zip(original_lines, canonical_lines) if o != c)
        assert changed_count == 3, \
            f"Exactly 3 lines should be changed, but {changed_count} were changed"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
