# JAX Barger Propagator

JAX-based differentiable neutrino oscillation propagator implementing the Barger
et al. (PRD 22.11, 1980) matter oscillation formalism through Earth.
Replicates the CUDAProb3 physics with automatic differentiation support.

## Structure

```
jax_barger/
├── jax_barger/              # Python package
│   ├── __init__.py
│   ├── config.py            # Physical constants + DTYPE precision control
│   ├── pmns.py              # PMNS matrix builder + mass differences
│   ├── earth.py             # PREM density model + path geometry
│   ├── matter.py            # Matter-effect cubic eigenvalue solver
│   ├── barger.py            # Core propagation engine (vectorized over E, cosθ)
│   ├── event_rate.py        # Event-rate folding: P × flux × xsec
│   └── mcmc.py              # HMC sampler + Laplace evidence + MAP finder
├── validate.py              # Forward validation against C++ ParProb3ppOscillation
├── compare_fit.py           # JAX vs C++ fitting comparison (L-BFGS-B, NM, MIGRAD)
├── compare_fit_fine.py      # Fine-binning + rebinning hierarchy discrimination
├── run_mcmc.py              # Single-model HMC driver (--fast/--fine, --fp32)
├── run_hierarchy_mcmc.py    # NH vs IH Bayes factor via HMC + Laplace
├── plot_corner.py           # Corner-plot generator (png/pdf/eps, with metadata)
├── pyproject.toml           # uv package config
└── README.md
```

## Quick Start

```bash
cd jax_barger

# Forward validation (requires built pybind module: mcmcoscfitter)
PYTHONPATH=../build/pybind:.. .venv/bin/python validate.py

# Fast HMC test (algorithm debug, non-physical χ²)
PYTHONPATH=../build/pybind:.. .venv/bin/python run_mcmc.py --fast --warmup 100 --samples 100

# Production HMC (fine grid, physically correct)
JAX_BARGER_FLOAT32=1 PYTHONPATH=../build/pybind:.. .venv/bin/python \
    run_mcmc.py --fine --warmup 200 --samples 2000

# NH vs IH hierarchy comparison (Bayes factor)
JAX_BARGER_FLOAT32=1 PYTHONPATH=../build/pybind:.. .venv/bin/python \
    run_hierarchy_mcmc.py --fine --warmup 200 --samples 2000 --chains 1
```

```python
from jax_barger import oscillation_probabilities
from jax_barger.earth import default_prem
from jax_barger.event_rate import event_rate
import jax.numpy as jnp

P = oscillation_probabilities(
    E_grid, cos_grid, theta12, theta13, theta23,
    deltacp, dm21sq, dm32sq, radii, density, Ye)
# P shape: (2, 3, 3, nE, nCos) — (ν/ν̄, from, to, E, cosθ)

events = event_rate(P, flux_dict, xsec_dict)
```

## Validation Results

JAX Barger propagator validated against C++ ParProb3ppOscillation (CUDAProb3):

| Test (30 E × 15 cosθ, 4 parameter sets) | max |ΔP| | rms |ΔP| | Result |
|-------------------------------------------|-----------|-----------|--------|
| Oscillation probabilities                 | 7.8e-05   | 5.6e-06   | ✓      |
| Event rates (numu)                        | 4.3e-03   | 4.9e-04   | ✓      |
| Event rates (nue)                         | 2.8e-03   | 3.1e-04   | ✓      |

Differences are limited by CUDAProb3's float32 precision; JAX uses float64.

## Key Findings from Fitting Analysis

### 1. χ² Agreement: JAX ≡ C++

The chi2 function computed by JAX matches C++ identically at all parameter
points tested:

```
Point          JAX χ²      C++ χ²       Δ
Truth           -0.0000     -0.0000     0.00
Biased start   13216.38    13216.38     0.00
Mid-point      1639.68     1639.68     0.00
```

### 2. Gradient Singularity: sin²θ Parameterization

The parameterization `θ = arcsin(√sin²θ)` has a divergent derivative at
sin²θ = 1 (θ = π/2, maximal mixing):

```
∂θ/∂(sin²θ) = 1 / (2·sinθ·cosθ) → ∞  at cosθ = 0
```

This causes `jax.grad` to return `inf` when L-BFGS-B hits the T23 upper
bound at sin²θ₂₃ = 1.0.

**Fix**: Use θ as the fitting parameter directly, and convert to sin²θ
only when computing the Gaussian pull penalty.

### 3. Hessian Ill-Conditioning

The raw-parameter Hessian has a condition number of **~7×10¹⁰**, dominated
by the DM2/Dm2 directions (tight PDG priors: σ_DM2 ~ 2.8×10⁻⁵ eV² vs
σ_θ₂₃ ~ 0.018 rad).

**Fix**: Work in σ-units (z-space): zᵢ = (pᵢ - truthᵢ) / σᵢ. This drops
the condition number to **~10³** because the pull penalty contributes
exact 2·I to the Hessian — acting as natural regularization.

### 4. χ² Landscape Barrier

The χ² surface has a physical barrier between the far biased start
(χ² ~ 13216) and the global minimum (χ² = 0). Along a straight line
from a typical stuck point to the truth, χ² first INCREASES before
decreasing:

```
Distance from stuck point → truth:
  t=0.0: χ²=1638  (stuck)
  t=0.5: χ²=2132  (peak — barrier!)
  t=1.0: χ²=0     (truth)
```

This barrier affects all local optimization methods (Nelder-Mead,
L-BFGS-B) equally. The production chi2fit avoids it via 12 random starts.

### 5. Fitting Performance Summary

| Method | Param | Start | χ² | Evals | Time | Evals/s |
|--------|-------|-------|----|-------|------|--------|
| C++ NM | sin²θ | 1σ | 0.00 | 852 | 0.3s | 2840 |
| JAX NM | θ | 1σ | 0.00 | 771 | 4.2s | 184 |
| JAX L-BFGS-B | θ | 1σ | 9.2 | 153 | 8.7s | — |
| JAX L-BFGS-B | z-space | 1σ | **0.00** | **44** | 7.7s | **5.7** |
| JAX L-BFGS-B | z-space | far | 5062 | 48 | 7.5s | — |
| C++ NM | sin²θ | far | 1650 | 882 | 1.0s | 882 |

### 6. Fine Binning + Rebinning — Critical for Physics

**Direct evaluation at analysis-bin centers produces biased χ² values.**
Oscillation probabilities vary on energy scales much finer than the
analysis bin width (~2.7 GeV/bin for 10 E-bins across [0.1, 20] GeV).
A single center-point value does not represent the bin-averaged
oscillation probability. This is especially severe for hierarchy
discrimination, where the wrong hierarchy's center-point values
coincidentally differ more from truth than the bin-averaged ones.

**Comparison: NH data vs IH prediction (pure Poisson χ², no pull terms):**

| Method | Points | Poisson χ² | Note |
|--------|--------|------------|------|
| Direct 10×12 centers | 120 | **893.6** | ~37× overestimate |
| 200E×120cosθ + rebin 20×10 | 24,000 | **24.1** | Physically correct |

The fine-binning approach matches the C++ production workflow (400E×480cosθ
→ rebin 40×40 → 10×12 analysis in `chi2fit`/`chi2fitCU`).

**Hierarchy discrimination (200×120 fine + rebin, 6-year exposure, z-space L-BFGS-B):**

| Hypothesis | Best χ² | Δχ² | Significance |
|------------|---------|-----|-------------|
| NH (correct) | 0.00 | — | — |
| IH (wrong) | 23.97 | 24.0 | ~4.9 σ |

All multi-start runs converge to the same IH minimum (χ² ≈ 23.97),
indicating a well-defined global structure. The NH fit converges to
χ² = 0 in 12 L-BFGS-B evaluations (~17s on GPU).

See `compare_fit_fine.py` for the implementation.

### 7. GPU Performance

Per-evaluation timing on RTX 3060 (10 E × 12 cosθ grid):

| Backend | ms/eval | Notes |
|---------|---------|-------|
| JAX JIT (GPU) | 2.3 | After JIT warmup (2-3s compilation) |
| C++ CUDA + CPU χ² | 0.7-1.4 | CUDA kernel + OpenMP χ² sum |

JAX is ~3× slower per evaluation due to XLA dispatch overhead, but enables
analytical gradients and end-to-end differentiability.

## Dependencies

- `jax[cuda12]` (GPU acceleration)
- `numpy`
- `scipy` (for fitting/optimization)
- `matplotlib` (for corner plots)
- `mcmcoscfitter` (C++ pybind module, for data export and validation)

## 8. HMC Sampler

Adaptive Hamiltonian Monte Carlo in θ-space with prior-based mass matrix and
dual-averaging step-size tuning (`mcmc.py`).  Key features:

- **θ-space sampling** avoids the `∂θ/∂(sin²θ) → ∞` gradient singularity.
- **Prior-based mass matrix** `M_ii = 1/σ²_θ` correctly handles the 11-decade
  range of posterior eigenvalues without sample-based (noisy) adaptation.
- **Multi-chain jax.lax.scan** production sampling with JIT-compiled chains.

| Component | File | Purpose |
|-----------|------|---------|
| `HMCSampler` | `mcmc.py` | Warmup + multi-chain production sampling |
| `build_neg_log_posterior` | `mcmc.py` | Construct θ-space neg-log-posterior with pull priors |
| `find_map` | `mcmc.py` | L-BFGS-B MAP search with analytical gradients |
| `laplace_log_evidence` | `mcmc.py` | Finite-difference Laplace evidence estimator |
| `run_mcmc.py` | — | Single-model HMC driver (`--fast`/`--fine`) |
| `run_hierarchy_mcmc.py` | — | NH vs IH comparison with Bayesian evidence |

### Performance (fine grid, RTX 3060, fp32)

| Phase | Time | Notes |
|-------|------|-------|
| JIT compile | ~12 s | One-time XLA compilation |
| func + grad eval | 73 ms | GPU, 200E×120cosθ |
| HMC proposal (10 leapfrog) | ~1.5 s | 20 grad + 2 func evals |
| Warmup 200 steps | ~44 min | Step-size adaptation |
| Sampling 2000 samples | ~50 min | jax.lax.scan, JIT compiled |

### Hierarchy Bayes Factor Results

NH Asimov data, fine grid (200E×120cosθ → 10×12), full PDG pull priors:

| Method | 2 ln BF | BF | Evidence |
|--------|---------|-----|----------|
| Laplace (fp64, prior mean) | 24.02 | 1.64×10⁵ | Decisive |
| Laplace (fp32, prior mean) | 22.63 | 8.20×10⁴ | Decisive |
| Laplace (fp32, posterior mean) | 26.64 | 6.11×10⁵ | Decisive |

| Pull configuration | 2 ln BF (fine) | BF |
|---|---|---|
| Full (6 pulls) | 24.02 | 1.64×10⁵ |
| Dm2 + θ₁₃ + θ₁₂ only | 18.17 | 8.83×10³ |

The Hessian determinant correction is < 5% of the Δ-χ² term.  The Bayes factor
remains decisive even when only the three most tightly constrained parameters
carry prior information.

## 9. Float32 Precision

Set `JAX_BARGER_FLOAT32=1` environment variable (or `--fp32` CLI flag) before
importing to enable fp32 throughout the entire computation chain.  This:

- **Halves GPU VRAM usage** — essential for fine-grid (200×120) on consumer GPUs
- **Speeds up evaluation ~2–3×** on RTX 3060 (73 ms/eval fp32 vs ~200 ms fp64)
- **Matches C++ CUDAProb3 internal precision** (CUDAProb3 uses float32)

```bash
# fp32 production run
JAX_BARGER_FLOAT32=1 python run_mcmc.py --fine --warmup 200 --samples 2000

# fp64 (default, backward compatible)
python run_mcmc.py --fast ...
```

The config module exports `DTYPE` and `DTYPE_NP` so all downstream code uses
the current precision consistently:

```python
from jax_barger.config import DTYPE, DTYPE_NP
# DTYPE is jnp.float32 or jnp.float64 depending on JAX_BARGER_FLOAT32
```

## 10. Corner Plot

`plot_corner.py` generates publication-ready pair plots from MCMC chain files:

```bash
python plot_corner.py \
  --nh hmc_chains_nh_fine2k.npz --ih hmc_chains_ih_fine2k.npz \
  --basename posterior_corner \
  --pull "full: DM2 Dm2 s2θ23 s2θ13 δCP s2θ12" \
  --grid "fine 200E×120cosθ → 10×12" \
  --precision fp32
```

Outputs `posterior_corner.{png,pdf,eps}`.  The upper-right corner carries a
metadata panel listing active pull terms, grid configuration, and a caveat
about observational errors.

## See Also

- `validate.py` — forward validation against C++ Prob3++ oscillation probabilities
- `compare_fit.py` — optimization comparison (JAX L-BFGS-B vs C++ Nelder-Mead)
- `compare_fit_fine.py` — fine-binning + rebinning hierarchy discrimination
- `run_mcmc.py` — single-model HMC driver
- `run_hierarchy_mcmc.py` — NH vs IH Bayesian model comparison
- `plot_corner.py` — posterior corner-plot generator
- `mcmc.py` — core HMC sampler, Laplace evidence, MAP finder
- `pybind/data_export.cxx` — Honda flux / GENIE xsec / PREM data export
- `external/CUDAProb3/` — reference CUDA implementation
