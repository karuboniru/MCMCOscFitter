# MCMCOscFitter

MCMC and gradient-based fitting of neutrino oscillation parameters
for JUNO, using CUDA-accelerated Earth-matter propagation and binned
Poisson likelihood.

## Overview

The project provides three complementary fitting pipelines:

| Pipeline | Backend | Algorithm | Interface |
|----------|---------|-----------|-----------|
| `chi2fit` / `chi2fitCU` | CUDA + Minuit2 | MIGRAD (quasi-Newton) | C++ executable |
| `testfitbinned` / `testfitbinnedCU` | CUDA + ROOT | Metropolis-Hastings MCMC | C++ executable |
| `jax_barger/` | JAX (GPU) | L-BFGS-B, HMC, NumPyro NUTS | Python |

All three share the same physics core: Barger et al. matter oscillation
through Earth using the PREM density model evaluated over 2D histograms
in (E, cosθ).

## Directory Map

```
MCMCOscFitter/
├── src/                        # C++ source
│   ├── app/                    # Executables (chi2fit, event_rate, MCMC, ...)
│   ├── common/                 # Shared utilities (binning, constants, pod_hist, physics_input, fit_config, interpolation, myhist, timer, tools)
│   ├── concepts/               # C++20 concept constraints (mcmc_concepts.h)
│   ├── data/                   # SimpleDataHist / SimpleDataPoint (binned / point-like data)
│   ├── llh/                    # Likelihood models
│   │   ├── BinnedInteraction*  # CPU binned likelihood (injectable + production ctors)
│   │   ├── ParBinned*          # GPU binned likelihood (pImpl, CUDA kernels)
│   │   └── SimpleInteraction*  # Unbinned event-by-event likelihood
│   ├── state/                  # Oscillation parameter objects
│   │   ├── OscillationParameters*  # 6 PMNS params + pull terms + hierarchy
│   │   ├── IHistogramPropagator.h  # Abstract propagator interface
│   │   ├── Prob3ppOscillation*     # CPU propagator (Prob3++)
│   │   └── ParProb3ppOscillation*  # GPU propagator (CUDAProb3)
│   └── walker/                 # MCMC infrastructure
│       ├── walker*             # Metropolis-Hastings sampler
│       ├── ParallelTempering.h # Replica exchange
│       ├── MCMCWorker.h        # Worker with LLH cache + two-state ownership
│       └── temperature_ladder.h
│
├── external/                   # Git submodules
│   ├── CUDAProb3/              # GPU Barger propagator (header + CUDA kernels)
│   ├── Prob3plusplus/          # CPU Barger propagator (C)
│   ├── hondaflux/              # Honda atmospheric flux (1D interpolation)
│   ├── hondaflux2d/            # Honda atmospheric neutrino flux (2D interpolation)
│   ├── xsec_genie_tune/        # GENIE neutrino cross-sections (ROOT splines)
│   └── toyfit_tools/           # Toy MC fitting utilities and neutrino state
│
├── pybind/                     # Python bindings (pybind11)
│   ├── bindings.cxx            # Core API: ParProb3ppOscillation, BinnedInteraction, MCMC
│   └── data_export.cxx         # Honda flux / GENIE xsec / PREM → numpy arrays
│
├── jax_barger/                 # JAX-based differentiable propagator + fitting
│   ├── jax_barger/             # Python package
│   │   ├── config.py           # Physical constants (G_F, R_earth, ...)
│   │   ├── pmns.py             # PMNS matrix + mass differences
│   │   ├── earth.py            # PREM model + Earth path geometry
│   │   ├── matter.py           # Matter-effect cubic eigenvalue solver
│   │   ├── barger.py           # Core propagation (vmapped over E, cosθ)
│   │   ├── event_rate.py       # Flux × prob × xsec folding + Poisson χ²
│   │   ├── mcmc.py             # HMC sampler + Laplace evidence + MAP finder
│   │   └── pymc_model.py       # Unified fitting API + NumPyro NUTS (z-space)
│   ├── check_pymc_env.py       # Verify PyMC + NumPyro + JAX environment
│   ├── run_pymc.py             # Unified Bayesian + frequentist driver (--sampler hmc/numpyro)
│   ├── run_hierarchy_pymc.py   # NH vs IH via ArviZ model comparison
│   ├── test_pymc_model.py      # Cross-check: logp, gradient, MAP, NUTS vs HMC
│   ├── validate.py             # Forward validation against C++ ParProb3ppOscillation
│   ├── compare_fit.py          # JAX vs C++ fitting comparison (optimisation algorithms)
│   ├── compare_fit_fine.py     # Fine-binning + rebinning hierarchy test
│   ├── run_mcmc.py             # (Legacy) Single-model HMC driver (--fast/--fine, --fp32)
│   ├── run_hierarchy_mcmc.py   # (Legacy) NH vs IH Bayes factor via HMC + Laplace
│   ├── plot_corner.py          # Corner-plot generator (png/pdf/eps, with metadata)
│   ├── plot_arviz.py           # ArviZ diagnostic plots (trace, pair, forest, rank, ridge)
│   ├── pyproject.toml          # uv package config
│   └── uv.lock
│
├── data/                       # Physics input files
│   ├── honda-2d.solmin.txt     # Honda atmospheric flux (solar minimum)
│   ├── honda-2d.solmax.txt     # Honda atmospheric flux (solar maximum)
│   ├── honda-3d.txt            # Honda 3D flux
│   ├── density.txt             # PREM Earth density profile (5 layers)
│   └── total_xsec.root         # GENIE cross-section splines (~160 MB)
│
├── test/                       # Catch2 unit tests
│   ├── test_binning.cxx        # Binning utilities
│   ├── test_constants.cxx      # Physical constants
│   ├── test_osc_params.cxx     # Oscillation parameter pull terms
│   ├── test_binned_interaction.cxx  # BinnedInteraction (injectable + mock propagator)
│   ├── test_chi2.cxx           # Poisson χ² statistic
│   ├── test_walker.cxx         # MCMC walker
│   └── test_parallel_tempering.cxx  # Parallel tempering
│
├── cross_check/                # Standalone cross-checks (event rate, probability, ROOT-to-hist)
├── baselines/                  # Reference output for regression testing (golden files + check.sh)
├── CMakeLists.txt              # Top-level build
├── CLAUDE.md                   # Build / test / architecture reference (for AI tools)
└── BASELINE_VERIFICATION.md    # Regression-test workflow
```

## Physics Pipeline

```
                      ┌─────────────────────────┐
                      │ OscillationParameters   │
                      │ θ₁₂, θ₁₃, θ₂₃, δCP,     │
                      │ Δm²₂₁, Δm²₃₂, hierarchy │
                      └────────────┬────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
┌──────────────────────┐ ┌─────────────────┐ ┌────────────────────┐
│ ParProb3ppOscillation│ │IHistogramPropag.│ │ jax_barger.barger  │
│ (CUDA GPU kernel)    │ │ (Python subclass)│ │ (JAX vmapped)      │
│ CUDAProb3 Barger     │ │                 │ │ Pure-JAX Barger    │
└──────────┬───────────┘ └────────┬────────┘ └─────────┬──────────┘
           │                      │                     │
           ▼                      ▼                     ▼
   raw P[3×3][E][cosθ]    PodHist2D P histograms    JAX array P
           │                      │                     │
           └──────────────────────┼─────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │  flux × P × xsec → events │
                    │  (Honda 2D + GENIE splines)│
                    └────────────┬─────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
          ┌──────────────────┐    ┌──────────────────────┐
          │ BinnedInteraction│    │ jax_barger.event_rate│
          │ (CPU OpenMP)      │    │ (JAX GPU)             │
          │ Poisson χ² + pull │    │ Poisson χ² + pull     │
          └────────┬─────────┘    └──────────┬───────────┘
                   │                          │
                   ▼                          ▼
           ┌──────────────────┐    ┌──────────────────────┐
           │ Minuit2 MIGRAD   │    │ L-BFGS-B / HMC /     │
           │ or MCMC walker   │    │ NumPyro NUTS          │
           └──────────────────┘    │ (scipy + jax_barger)  │
                                   └──────────────────────┘
```

## Key Components

### Oscillation Propagation

- **`ParProb3ppOscillation`** (`src/state/`): GPU-accelerated oscillation probability computation using CUDAProb3. Evaluates 3×3 PMNS propagation through the PREM Earth model at all (E, cosθ) points simultaneously. Supports raw-result access for the fast path in `BinnedInteraction::UpdatePrediction()`.

- **`IHistogramPropagator`** (`src/state/`): Abstract base class for propagators. Python subclasses (via `PropagatorBase` trampoline) can override `get_prob_hists()` / `get_prob_hists_3f()` to inject custom oscillation physics.

- **`jax_barger.barger`**: Pure-JAX implementation of the Barger propagator, vmapped over E and cosθ grids. Validated against CUDAProb3 to within float32 precision (rms ΔP = 5.6×10⁻⁶). Provides analytical gradients through all physics operations.

### Likelihood Models

- **`BinnedInteraction`** (`src/llh/`): Binned Poisson likelihood using 2D histograms (E × cosθ). Two constructors:
  - **Injectable**: accepts pre-built flux/xsec histograms + a propagator (used from Python)
  - **Production**: reads Honda flux + GENIE xsec from data files (used from C++ executables)

- **`ParBinned`** (`src/llh/`): GPU-native variant. Oscillation probabilities stay on device; event-rate rebinning uses `atomicAdd` CUDA kernels. No D2H transfer until likelihood evaluation.

- **`SimpleInteraction`** (`src/llh/`): Unbinned event-by-event likelihood for low-statistics samples.

### MCMC Infrastructure

- **`walker`** (`src/walker/`): Metropolis-Hastings sampler with Configurable acceptance temperature.
- **`ParallelTempering`** (`src/walker/`): Replica exchange (parallel tempering) with configurable temperature ladder.
- **`MCMCWorker`** (`src/walker/`): Worker state ownership pattern — caches the LLH across propose/accept to avoid redundant recomputation.

### Data Exchange (pybind)

- **`mcmcoscfitter`** Python module (built from `pybind/`):
  - `ParProb3ppOscillation` — GPU propagator accessible from Python
  - `BinnedInteraction` — injectable constructor for MCMC in Python
  - `BinnedHistograms` — numpy → PodHist2D conversion
  - `load_honda_flux_2d()`, `load_genie_xsec()`, `load_physics_input()` — export flux/xsec as numpy arrays
  - `load_prem_density()` — PREM radii, density, Yₑ
  - MCMC utilities: `mcmc_accept()`, `mcmc_accept_swap()`, `set_seed()`

### JAX Fitting (`jax_barger/`)

Key findings documented in `jax_barger/README.md`:

- χ² values match C++ identically at all parameter points
- `sin²θ` parameterisation causes `∂θ/∂(sin²θ) → ∞` at maximal mixing → use **θ directly** for fitting
- Raw-parameter Hessian condition number ~7×10¹⁰ → **z-space rescaling** drops it to ~10³
- Fine binning + rebinning is **required** for correct physics: direct center-point evaluation overestimates hierarchy discrimination by ~37×

**Unified fitting workflow** (`run_pymc.py`): starts from JAX physics, runs MAP via L-BFGS-B (frequentist), then MCMC sampling via HMCSampler (fast, well-tuned) or NumPyro NUTS (adaptive trajectory, exact posterior via z-space rescaling), with ArviZ diagnostics and netCDF output.

**Quick usage:**
```bash
cd jax_barger
PYTHONPATH=../build/pybind:.. .venv/bin/python run_pymc.py --fast --skip-hmc       # MAP only
PYTHONPATH=../build/pybind:.. .venv/bin/python run_pymc.py --fast --sampler hmc     # HMC
PYTHONPATH=../build/pybind:.. .venv/bin/python run_pymc.py --fast --sampler numpyro # NumPyro NUTS
```

## Build & Run

```bash
# CPU build
cmake -B build -G Ninja -DENABLE_CUDA=OFF
cmake --build build/

# CUDA build — 6 presets defined in CMakeUserPresets.json
export PATH="/usr/local/cuda/bin:$PATH"

# General-purpose CUDA builds
cmake --preset default              # nvcc + clang++ host compiler
cmake --preset cuda-gcc             # nvcc + gcc/g++ host compiler
cmake --preset cuda-llvm            # clang++ as CUDA compiler

# Verification builds (required for baseline regression tests)
cmake --preset cuda-gcc15-clang     # nvcc (g++-15 host) + clang++ compile driver
cmake --preset cuda-gcc15-clang-fp32  # above + OSCILLATION_FP=float
cmake --preset cuda-gcc15-clang-fp64  # above + OSCILLATION_FP=double

cmake --build build/

# Run tests
ctest --test-dir build/

# Python bindings
cmake --build build/ --target mcmcoscfitter
PYTHONPATH=build/pybind python -c "import mcmcoscfitter as mof; print(mof.scale_factor_6y)"

# JAX fitting (requires pybind module + uv environment)
cd jax_barger
uv sync
PYTHONPATH=../build/pybind:.. .venv/bin/python check_pymc_env.py
PYTHONPATH=../build/pybind:.. .venv/bin/python run_pymc.py --fast --skip-hmc
PYTHONPATH=../build/pybind:.. .venv/bin/python validate.py
```

## Data Files

Physics input data lives in `data/` at the repo root:

| File | Description |
|------|-------------|
| `honda-2d.solmin.txt` | Honda atmospheric flux, 2D (solar minimum, default) |
| `honda-2d.solmax.txt` | Honda atmospheric flux, 2D (solar maximum) |
| `honda-3d.txt` | Honda atmospheric flux, 3D |
| `density.txt` | PREM Earth density profile (radius, density, Yₑ) |
| `total_xsec.root` | GENIE cross-section splines (~160 MB, not tracked in git) |

`DATA_PATH` is defined at compile time as the repo root.

## External Dependencies

- **ROOT** (CERN) — histograms, TF1/TF2 integration, Minuit2
- **Eigen3** — linear algebra
- **OpenMP** — CPU parallelism
- **CUDA** (optional) — GPU acceleration
- **pybind11** — Python bindings (fetched via CMake FetchContent)
- **Catch2** — unit tests (fetched via CMake FetchContent)
- **JAX + scipy** — Python-side fitting and validation (via `uv`)
