# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

All builds use CMake with Ninja and output to `build/`. Three presets are defined in `CMakeUserPresets.json`:

```bash
# Configure (choose one preset)
cmake --preset default       # CUDA with clang++ host compiler
cmake --preset cuda-gcc      # CUDA with gcc/g++ host compiler
cmake --preset cuda-llvm     # CUDA with clang++ as CUDA compiler

# Build all targets
cmake --build build/

# Build a specific target
cmake --build build/ --target testfitbinned

# CPU-only build (without a preset)
cmake -B build -G Ninja -DENABLE_CUDA=OFF
cmake --build build/
```

Build type is `RelWithDebInfo` with IPO and `-march=native`. ASAN/UBSAN flags are commented out in `CMakeLists.txt` and can be re-enabled for debugging.

## Executables

After building, binaries are in `build/`. Key executables:

| Binary | Purpose |
|--------|---------|
| `testsample` | Sample events from oscillation model |
| `testfit` | MCMC on unbinned data |
| `testfitbinned` | MCMC on binned histograms (CPU) |
| `chi2fit` | χ² minimization (CPU, ROOT Minuit2) |
| `event_rate` | Calculate NC/CC event rates (CPU) |
| `event_rateCU` | GPU-accelerated event rates |
| `testfitbinnedCU` | GPU-accelerated MCMC fitting |
| `chi2fitCU` | GPU-accelerated χ² minimization |
| `event_rate_xcheck` | Cross-check event rate calculation |

Executables rely on data files at the repo root (`data/` directory). The `DATA_PATH` macro is defined at compile time as the repo root.

## Testing

Catch2 v3 is fetched automatically via CMake FetchContent. Tests are registered with CTest.

```bash
# Build all test executables
cmake --build build/ --target test_pure test_physics test_binned

# Run all tests
ctest --test-dir build/

# Run a single test by name (Catch2 tag or test-case name)
build/test/test_pure "linspace"
build/test/test_physics "TH2D_chi2: data == prediction"

# Verbose output
ctest --test-dir build/ --output-on-failure
```

Three test executables:

| Executable | Coverage | External deps |
|-----------|---------|---------------|
| `test_pure` | `binning_tool.hpp`, `constants.h` | none |
| `test_physics` | `OscillationParameters`, `walker`, `TH2D_chi2` | ROOT |
| `test_binned` | `BinnedInteraction` (injectable constructor + mock propagator) | ROOT |

`test_binned` uses `IdentityPropagator` (in `test_binned_interaction.cxx`) — a mock that returns no-oscillation probability matrices. The `BinnedInteraction` injectable constructor accepts pre-built flux/xsec histograms and a `shared_ptr<IHistogramPropagator>`, so the full GENIE/Honda physics stack is not required.

## Architecture

### Physics Pipeline

```
OscillationParameters          ← PMNS parameters (θ₁₂, θ₁₃, θ₂₃, Δm², δCP, hierarchy)
    ↓
Prob3ppOscillation             ← wraps Prob3++ for oscillation probability histograms (CPU)
ParProb3ppOscillation          ← parallelized variant (CUDA .cu or OpenMP .cxx, same header)
    ↓
BinnedInteraction              ← binned likelihood using 2D histograms (E vs cos θ, 4 flavors)
SimpleInteraction              ← unbinned event-by-event likelihood
ParBinnedInterface (pImpl)     ← wraps CUDA ParBinned kernels
    ↓
ModelAndData<Model, Data>      ← combines model likelihood + data likelihood
    ↓
walker                         ← Metropolis-Hastings MCMC sampler
```

### Key Design Points

- **CPU/CUDA duality**: `ParProb3ppOscillation` has two compilation units — `ParProb3ppOscillation.cu` for CUDA builds and `ParProb3ppOscillation.cxx` for CPU builds. The CMake `ENABLE_CUDA` flag selects which is compiled.
- **pImpl for CUDA isolation**: `ParBinnedInterface.h` uses pImpl so CUDA code in `ParBinned.cu`/`ParBinnedKernels.cu` is not exposed to non-CUDA translation units.
- **StateI interface**: `src/state/StateI.h` is the abstract base class for state objects used by the walker — implements `proposeStep()` and `GetLogLikelihood()`.
- **Separate TUs for compile speed**: CUDA kernels are split across multiple `.cu` files to allow parallel compilation.
- **Atomic rebinning**: The CUDA rebinning uses `atomicAdd` for lock-free histogram filling.

### External Dependencies

- **ROOT** (CERN) — histograms, RDataFrame, Minuit2
- **Eigen3** — linear algebra
- **OpenMP** — CPU parallelism (required even for CUDA builds via CUDAProb3)
- **Prob3plusplus** (`external/`) — PMNS oscillation calculations through Earth
- **CUDAProb3** (`external/`) — header-only GPU/OpenMP oscillation propagation
- **hondaflux / hondaflux2d** (`external/`) — Honda atmospheric neutrino flux models
- **xsec_genie_tune** (`external/`) — GENIE neutrino cross-section spline interface

All external libraries are git submodules under `external/`.

### Data Files

Physics input data lives in `data/` at the repo root:
- `honda-2d.*.txt` / `honda-3d.txt` — Honda atmospheric flux tables
- `density.txt` — Earth density profile (for oscillation through Earth)
- `total_xsec.root` — GENIE cross-section splines (~160 MB, not in git)
