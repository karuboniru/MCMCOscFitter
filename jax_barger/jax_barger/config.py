"""Physical constants for the Barger neutrino oscillation propagator.

Matches the values used in CUDAProb3 (external/CUDAProb3/src/math/constants.cuh).

Precision control
-----------------
Default (float64):    matches C++ double, validated to ~1e-4 against CUDAProb3.
float32:              set environment variable ``JAX_BARGER_FLOAT32=1`` before
                       importing.  Reduces GPU VRAM ~2×, speeds up evaluation
                       ~2-3× on consumer GPUs (RTX 3060).  Numerical accuracy
                       limited to float32 (~7 decimal digits), which matches
                       the C++ CUDAProb3 internal precision exactly.

VRAM-conservation workarounds
-----------------------------
On consumer GPUs (e.g. RTX 3060 12GB), two workarounds are active by default:

1. ``jax.checkpoint`` on the per-point oscillation propagator — trades
   forward recomputation for reduced AD intermediate storage.
2. Batched mean-U evaluation (100-sample slices) after chain generation.

On professional GPUs (A100 80GB, H100 80GB) these restrict throughput.
Disable by setting ``JAX_BARGER_NO_VRAM_WORKAROUND=1`` before import.

Examples::
    $ python run_mcmc.py --fast ...                         # fp64, workarounds ON
    $ JAX_BARGER_FLOAT32=1 python run_mcmc.py --fast ...    # fp32, workarounds ON
    $ JAX_BARGER_NO_VRAM_WORKAROUND=1 python run_mcmc.py --fp32 --fast ...  # no workarounds
"""

import os
import jax
import numpy as np

# ── Float precision ──
# NOTE: importing this module sets ``jax_enable_x64 = True`` globally,
# affecting all subsequent JAX operations in the process.  Set
# ``JAX_BARGER_FLOAT32=1`` before import if float32 is desired instead.

_FLOAT32 = os.environ.get('JAX_BARGER_FLOAT32', '').strip().lower() in (
    '1', 'true', 'yes', 'on')

if _FLOAT32:
    pass
else:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

DTYPE = jnp.float32 if _FLOAT32 else jnp.float64
DTYPE_NP = np.float32 if _FLOAT32 else np.float64
FLOAT_BITS = 32 if _FLOAT32 else 64

# ── VRAM-conservation workarounds (disable on >24 GB GPUs) ──

VRAM_SAFE = not os.environ.get('JAX_BARGER_NO_VRAM_WORKAROUND', '').strip().lower() in (
    '1', 'true', 'yes', 'on')

# ── Physical constants ──

tworttwoGf = 1.52588e-4   # 2√2 G_F [eV² · cm³ / (mol · GeV)]
LoEfac     = 2.534        # 1/(2ħc) [GeV / (eV² · km)]
R_earth    = 6371.0       # Earth radius [km]
h_prod     = 15.0         # atmospheric neutrino production height [km]
epsilon    = 5.0e-9       # small shift to break mass degeneracies
MAX_LAYERS = 10           # max possible: atm + 4 shells in + deepest + 4 shells out
