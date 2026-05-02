# JAX Barger Propagator

JAX-based differentiable neutrino oscillation propagator implementing the Barger
et al. (PRD 22.11, 1980) matter oscillation formalism through Earth.
Replicates the CUDAProb3 physics with automatic differentiation support.

## Structure

```
jax_barger/
├── jax_barger/           # Python package
│   ├── __init__.py
│   ├── config.py         # Physical constants (G_F, Earth radius, ...)
│   ├── pmns.py           # PMNS matrix builder + mass differences
│   ├── earth.py          # PREM density model + path geometry
│   ├── matter.py         # Matter-effect cubic eigenvalue solver
│   ├── barger.py         # Core propagation engine (vectorized over E, cosθ)
│   └── event_rate.py     # Event-rate folding: P × flux × xsec
├── validate.py           # Forward validation against C++ ParProb3ppOscillation
├── compare_fit.py        # JAX vs C++ fitting comparison (Nelder-Mead, L-BFGS-B)
├── pyproject.toml        # uv package config
└── README.md
```

## Quick Start

```bash
cd jax_barger
uv venv && uv pip install "jax[cuda12]" numpy scipy
# Run validation (requires built pybind module: mcmcoscfitter)
PYTHONPATH=../build/pybind:.. .venv/bin/python validate.py
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

### 6. GPU Performance

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
- `mcmcoscfitter` (C++ pybind module, for data export and validation)

## See Also

- `validate.py` — forward validation script
- `compare_fit.py` — fitting comparison script
- `pybind/data_export.cxx` — Honda flux / GENIE xsec / PREM data export
- `external/CUDAProb3/` — reference CUDA implementation
