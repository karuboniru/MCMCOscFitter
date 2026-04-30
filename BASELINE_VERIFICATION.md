# Baseline Verification

`event_rateCU` and `chi2fittestCU` serve as the physics truth baseline.
Any change that visibly shifts their output is a bug.

## Build Preset

Always use **`cuda-gcc15-clang-fp32`** (`OSCILLATION_FP=float`) for verification.
nvcc must be on `PATH`:

```bash
export PATH="/usr/local/cuda/bin:$PATH"
cmake --preset cuda-gcc15-clang-fp32
```

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

Every self-contained change must pass all three gates before the next change begins.

```bash
# Prerequisite: nvcc on PATH
export PATH="/usr/local/cuda/bin:$PATH"

# Gate 1 — Build
cmake --preset cuda-gcc15-clang-fp32
cmake --build build/ -j$(nproc) 2>&1 | tail -5

# Gate 2 — Unit tests (all 22 must pass)
ctest --test-dir build/ --output-on-failure -j$(nproc) 2>&1 | tail -10

# Gate 3 — Physics baselines
# 3a. Event rates (12 values must match ≤0.1%)
./build/src/app/event_rateCU 2>/dev/null

# 3b. χ² fit Fval (must match ≤0.3%)
./build/src/app/chi2fittestCU 2>/dev/null | grep -E '^(Fval:|chi2 )'
```

The per-iteration `std::cout << llh << std::endl;` in `chi2fittestCU` is log
noise — only the final `Fval:` and `chi2 ...` lines matter.

### Per-Step Verification

Each self-contained change (e.g. adding raw-access virtuals, rewriting UpdatePrediction,
decoupling data from model) is verified by running Gates 1–3 immediately after the change.
Gate 2 must be run on the full test suite (not just the changed file's tests) to catch
transitive breakage.

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
