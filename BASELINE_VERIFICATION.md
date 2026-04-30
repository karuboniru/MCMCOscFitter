# Baseline Verification

`event_rateCU` and `chi2fittestCU` serve as the physics truth baseline.
Any change that visibly shifts their output is a bug.

## Baselines

Captured at commit `dcaea99`. Stored in `baselines/`.

### event_rateCU — Event Rates

```
Non-Oscillated: numu=7013.751 numubar=2598.670 nue=3623.456 nuebar=1171.382
Normal Hierarchy: numu=4821.124 numubar=1804.614 nue=3740.287 nuebar=1152.843
Inverted Hierarchy: numu=4836.316 numubar=1792.963 nue=3687.296 nuebar=1170.733
```

Tolerances: machine precision and CUDA non-determinism (≤0.1% per flavour).

### chi2fittestCU — χ² Fit Fval

```
Fval: 2.2602e+01
chi2 6.6153e-01 data: 1.6275e-01 pull: 4.9878e-01
```

Only `Fval` is a hard constraint. The `chi2`/`data`/`pull` breakdown varies
with random initialisation; tolerance for Fval is ≤0.3% (千分之几).

## Verification Workflow

```bash
# 1. Build
cmake --preset default
cmake --build build/ --target event_rateCU chi2fittestCU

# 2. Check event rates (all 12 values must match within FP tolerance)
./build/src/app/event_rateCU 2>/dev/null

# 3. Check Fval (ignore per-iteration llh log lines)
./build/src/app/chi2fittestCU 2>/dev/null | grep -E '^(Fval:|chi2 )'
```

The per-iteration `std::cout << llh << std::endl;` in `chi2fittestCU` is log
noise — only the final `Fval:` and `chi2 ...` lines matter.

## What Touches the Baselines

| Change area | Affects |
|-------------|---------|
| `ParBinned.cu`, `ParBinnedKernels.cu`, `ParBinned.cuh` | Both |
| `global_device_input_instance` (in `ParBinned.cu`) | Both |
| `ParProb3ppOscillation.cu` / `.cxx` / `_devspan.cu` | Both |
| `FitConfig` (binning, scale, rebin) | Both |
| `HondaFlux2D`, `GENIE_XSEC` (external data) | Both |
| `OscillationParameters` (PDG values, priors) | Fval only |
| `chi2fittestCU.cxx` (the app itself) | Fval only |
| `event_rateCU.cxx` (the app itself) | Event rates only |
| `BinnedInteraction` (CPU path) | Neither |
| `pybind/`, `test/`, `walker` | Neither |
