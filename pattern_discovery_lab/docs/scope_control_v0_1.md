# Scope Control v0.1

## Default Configuration

**Principle**: Minimal viable detector set by default. Extended detectors are opt-in.

### Default Detectors (Always Enabled)

The following detectors run in every calibration by default:

1. **`autocorrelation_dependence`** - Linear dependence via ACF
2. **`time_reversal_asymmetry`** - Nonlinear structure via time-reversal test

**Rationale**: These two detectors cover orthogonal aspects of structure:
- Linear correlation (autocorrelation)
- Nonlinearity (time-reversal)

### Extended Detectors (Opt-In)

The following detectors require explicit enablement via config flag:

```yaml
enable_extra_detectors: true
```

When enabled, the following additional detectors will run:
- `lempel_ziv_compressibility` - Information-theoretic complexity
- `mutual_information` - Nonlinear dependence
- `changepoint` - Regime structure detection
- `stability` - Window-shift consistency

**Default**: `enable_extra_detectors: false`

## Configuration

### Enabling Extra Detectors

To run the full detector suite, update `configs/default.yaml`:

```yaml
# Scope control (v0.1)
enable_extra_detectors: true  # Enable all detectors
```

### Default Behavior

If `enable_extra_detectors` is not specified or set to `false`, only the two core detectors run:
- autocorrelation_dependence
- time_reversal_asymmetry

## Version History

- **v0.1** (2025-12-13): Initial scope lock with 2 default detectors, 4 opt-in detectors
