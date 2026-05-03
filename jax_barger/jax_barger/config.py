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

Examples::
    $ python run_mcmc.py --fast ...                # float64 (default)
    $ JAX_BARGER_FLOAT32=1 python run_mcmc.py --fast ...  # float32
"""

import os
import jax
import numpy as np

_FLOAT32 = os.environ.get('JAX_BARGER_FLOAT32', '').strip().lower() in (
    '1', 'true', 'yes', 'on')

if _FLOAT32:
    # keep jax_enable_x64 at default (False) → float32
    pass
else:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

# Exported dtype constants — use these instead of hardcoded jnp.float64 / np.float64.
DTYPE = jnp.float32 if _FLOAT32 else jnp.float64
DTYPE_NP = np.float32 if _FLOAT32 else np.float64
FLOAT_BITS = 32 if _FLOAT32 else 64

tworttwoGf = 1.52588e-4   # 2√2 G_F [eV² · cm³ / (mol · GeV)]
LoEfac     = 2.534        # 1/(2ħc) [GeV / (eV² · km)]
R_earth    = 6371.0       # Earth radius [km]
h_prod     = 15.0         # atmospheric neutrino production height [km]
epsilon    = 5.0e-9       # small shift to break mass degeneracies
MAX_LAYERS = 10           # max possible: atm + 4 shells in + deepest + 4 shells out
