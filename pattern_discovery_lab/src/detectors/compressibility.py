"""
Compressibility Detector - Lempel-Ziv Complexity

Measures algorithmic complexity via string compression.

Higher LZ complexity → less compressible → less structure (more random).
Lower LZ complexity → more compressible → more structure (more predictable).

Note: We invert the score so higher = more structure (consistent with other detectors).
"""

import numpy as np
from typing import Dict, Any
from ..detectors import BaseDetector


class LempelZivDetector(BaseDetector):
    """
    Lempel-Ziv complexity detector.
    
    Converts time-series to binary string (above/below median),
    then computes LZ complexity.
    
    structure_score = 1 - (LZ_complexity / max_possible_complexity)
    
    Higher score → more structure (more compressible).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: {
                'normalize': bool (default True, normalize to [0, 1])
            }
        """
        super().__init__(config)
        self.normalize = config.get('normalize', True)
    
    def to_binary_string(self, data: np.ndarray) -> str:
        """
        Convert time-series to binary string.
        
        Args:
            data: Time-series
        
        Returns:
            Binary string ('0' if below median, '1' if above)
        """
        median = np.median(data)
        binary = (data > median).astype(int)
        return ''.join(map(str, binary))
    
    def lempel_ziv_complexity(self, binary_string: str) -> int:
        """
        Compute Lempel-Ziv complexity.
        
        Counts number of distinct substrings in the LZ parsing.
        
        Args:
            binary_string: Binary string (e.g., '0110100...')
        
        Returns:
            LZ complexity (integer)
        """
        n = len(binary_string)
        i = 0
        complexity = 1
        
        while i < n:
            # Find longest prefix of S[i:] that occurs in S[0:i]
            max_len = 0
            for j in range(i + 1, n + 1):
                substr = binary_string[i:j]
                if binary_string[:i+1].find(substr) != -1:
                    max_len = j - i
                else:
                    break
            
            i += max(1, max_len)
            complexity += 1
        
        return complexity
    
    def detect(self, data: np.ndarray) -> float:
        """
        Compute LZ complexity structure score.
        
        Args:
            data: Time-series
        
        Returns:
            structure_score: 1 - (normalized LZ complexity)
                           Higher → more structure
        """
        binary_string = self.to_binary_string(data)
        lz_complexity = self.lempel_ziv_complexity(binary_string)
        
        if self.normalize:
            # Theoretical max complexity for random string of length n
            # is approximately n / log2(n)
            n = len(binary_string)
            max_complexity = n / np.log2(n) if n > 1 else 1
            normalized_complexity = lz_complexity / max_complexity
            
            # Invert: high complexity = low structure
            structure_score = 1.0 - min(normalized_complexity, 1.0)
        else:
            # Raw complexity (inverse)
            structure_score = 1.0 / (lz_complexity + 1)
        
        return structure_score
    
    def get_name(self) -> str:
        return "lempel_ziv_compressibility"
