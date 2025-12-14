# Pattern Discovery Lab - Run Report

**Run ID**: `20251213_070355`  
**Timestamp**: 2025-12-13T07:03:55.258355  
**Config**: default.yaml  

## STOP RULES

- ⚠️  INSTABILITY: time_reversal_asymmetry on white_noise shows high block variability
- ⚠️  INSTABILITY: time_reversal_asymmetry on ar1 shows high block variability
- ⚠️  INSTABILITY: time_reversal_asymmetry on garch shows high block variability

**Overall Status**: **PASS**

## Results Summary

| Dataset | Detector | Real Score | Shuffle P | Surrogate P | Stability | Status |
|---------|----------|------------|-----------|-------------|-----------|--------|
| white_noise | autocorrelation_dependence | 0.0149 | 0.7300 | 0.1000 | 0.3098 | FAIL |
| white_noise | time_reversal_asymmetry | 0.0732 | 0.3800 | 0.2800 | 0.5314 | FAIL |
| ar1 | autocorrelation_dependence | 0.9708 | 0.0000 | 0.3400 | 0.0974 | PASS |
| ar1 | time_reversal_asymmetry | 0.0638 | 0.3400 | 0.3600 | 0.5522 | FAIL |
| garch | autocorrelation_dependence | 0.0161 | 0.6600 | 0.0900 | 0.3216 | FAIL |
| garch | time_reversal_asymmetry | 0.0469 | 0.6000 | 0.4500 | 0.8176 | FAIL |

## Detailed Results

### Dataset: white_noise

- **Generator**: white_noise
- **Seed**: 42
- **Length**: 1000

#### Detector: autocorrelation_dependence

**Real Score**: 0.014946

**Controls**:
- Time-shuffle: p = 0.7300
- Phase-randomization: p = 0.1

**Stability (3 blocks)**: [0.056962183061116174, 0.03995995209656812, 0.025930748628241414]
- Stability metric (CV): 0.3098

**Status**: FAIL
**Failure reasons**: failed_time_shuffle

#### Detector: time_reversal_asymmetry

**Real Score**: 0.073152

**Controls**:
- Time-shuffle: p = 0.3800
- Phase-randomization: p = 0.28

**Stability (3 blocks)**: [0.19785373644216353, 0.06815636488264841, 0.07294791514943824]
- Stability metric (CV): 0.5314

**Status**: FAIL
**Failure reasons**: failed_time_shuffle

### Dataset: ar1

- **Generator**: ar1
- **Seed**: 42
- **Length**: 1000

#### Detector: autocorrelation_dependence

**Real Score**: 0.970788

**Controls**:
- Time-shuffle: p = 0.0000
- Phase-randomization: p = 0.34

**Stability (3 blocks)**: [0.9642010371293617, 0.878015387231386, 1.1101641705871046]
- Stability metric (CV): 0.0974

**Status**: PASS

#### Detector: time_reversal_asymmetry

**Real Score**: 0.063843

**Controls**:
- Time-shuffle: p = 0.3400
- Phase-randomization: p = 0.36

**Stability (3 blocks)**: [0.01700294686952084, 0.09728254504382422, 0.11107169381975948]
- Stability metric (CV): 0.5522

**Status**: FAIL
**Failure reasons**: failed_time_shuffle

### Dataset: garch

- **Generator**: garch
- **Seed**: 42
- **Length**: 1000

#### Detector: autocorrelation_dependence

**Real Score**: 0.016142

**Controls**:
- Time-shuffle: p = 0.6600
- Phase-randomization: p = 0.09

**Stability (3 blocks)**: [0.06453633368243832, 0.03943274261335787, 0.030481246435854696]
- Stability metric (CV): 0.3216

**Status**: FAIL
**Failure reasons**: failed_time_shuffle

#### Detector: time_reversal_asymmetry

**Real Score**: 0.046871

**Controls**:
- Time-shuffle: p = 0.6000
- Phase-randomization: p = 0.45

**Stability (3 blocks)**: [0.2654253575870591, 0.12614142791289862, 0.0025098121747533205]
- Stability metric (CV): 0.8176

**Status**: FAIL
**Failure reasons**: failed_time_shuffle
