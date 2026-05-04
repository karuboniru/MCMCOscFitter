# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

All builds use CMake with Ninja and output to `build/`. Three presets are defined in `CMakeUserPresets.json`:

```bash
# Configure (choose one preset)
cmake --preset default              # CUDA with clang++ host compiler
cmake --preset cuda-gcc             # CUDA with gcc/g++ host compiler
cmake --preset cuda-llvm            # CUDA with clang++ as CUDA compiler
cmake --preset cuda-gcc15-clang     # CUDA with nvcc (g++-15 host) + clang++ compile driver
cmake --preset cuda-gcc15-clang-fp32  # above + OSCILLATION_FP=float (baseline verification)
cmake --preset cuda-gcc15-clang-fp64  # above + OSCILLATION_FP=double

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
| `testptbinned` | Parallel tempering on binned histograms (CPU) |
| `chi2fit` | χ² minimization (CPU, ROOT Minuit2) |
| `chi2fittestCU` | CUDA χ² fit baseline verification |
| `chi2fitCU` | GPU-accelerated χ² minimization |
| `chi2fitplotCU` | GPU-accelerated χ² fit with plots |
| `event_rate` | Calculate NC/CC event rates (CPU) |
| `event_rateCU` | GPU-accelerated event rates |
| `event_rate_xcheck` | Cross-check event rate calculation |
| `testfitbinnedCU` | GPU-accelerated MCMC fitting |
| `plotter` | ROOT RDataFrame plotting utility |

Executables rely on data files at the repo root (`data/` directory). The `DATA_PATH` macro is defined at compile time as the repo root.

## Python bindings

The `pybind/` directory builds `mcmcoscfitter` — a Python extension module.
pybind11 is fetched automatically via FetchContent.

```bash
cmake --build build/ --target mcmcoscfitter
# .so lands in build/pybind/
export PYTHONPATH=$PWD/build/pybind
python3 -c "import mcmcoscfitter as mof; print(mof.scale_factor_6y)"
```

**Key Python API:**

```python
import mcmcoscfitter as mof
import numpy as np, copy

Ebins    = mof.logspace(0.1, 20.0, 51)      # 50 E-bins
costhbins = mof.linspace(-1.0, 1.0, 21)     # 20 costh-bins

prop = mof.ParProb3ppOscillation(mof.to_center(Ebins),
                                  mof.to_center(costhbins))

# Provide flux [nE×nCosth] and xsec [nE] as numpy arrays
histos = mof.BinnedHistograms(
    flux_numu=..., flux_numubar=..., flux_nue=..., flux_nuebar=...,
    xsec_numu=..., xsec_numubar=..., xsec_nue=..., xsec_nuebar=...,
    Ebins=Ebins, costhbins=costhbins)

model   = mof.BinnedInteraction(Ebins, costhbins, prop, histos)
data    = model.generate_data()   # DataHist with .numu/.numubar/.nue/.nuebar arrays

mof.set_seed(42)
current = model
for _ in range(10000):
    nxt = copy.deepcopy(current)    # BinnedInteraction supports __deepcopy__
    nxt.propose_step()
    if mof.mcmc_accept(current, nxt, data):
        current = nxt
    # read: current.DM32sq, .T23, .T13, .DM21sq, .T12, .DeltaCP, .is_NH
```

Custom propagators can be implemented in Python by subclassing `mof.PropagatorBase`
and overriding `get_prob_hists(Ebins, costhbins, params) -> np.ndarray (2,2,2,nE,nCosth)`
and `get_prob_hists_3f(...)  -> np.ndarray (2,3,3,nE,nCosth)`.

**Linking notes:** The pybind module links against `BinnedInteractionInject` (no
`HondaFlux2D`/`GENIE_XSEC` globals), so it imports cleanly without physics data
files. The production `BinnedInteraction(Ebins, costhbins, scale, ...)` constructor
(which reads Honda flux + GENIE xsec) is only available from C++ executables.

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
- **CUDAProb3** (`external/`) — GPU-accelerated Barger oscillation propagation (static library + CUDA kernels)
- **hondaflux** (`external/`) — Honda atmospheric neutrino flux (1D interpolation)
- **hondaflux / hondaflux2d** (`external/`) — Honda atmospheric neutrino flux models
- **xsec_genie_tune** (`external/`) — GENIE neutrino cross-section spline interface
- **toyfit_tools** (`external/`) — Toy MC fitting utilities and neutrino state

All external libraries are git submodules under `external/`.

### Data Files

Physics input data lives in `data/` at the repo root:
- `honda-2d.*.txt` / `honda-3d.txt` — Honda atmospheric flux tables
- `density.txt` — Earth density profile (for oscillation through Earth)
- `total_xsec.root` — GENIE cross-section splines (~160 MB, not in git)
