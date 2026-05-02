"""Physical constants for the Barger neutrino oscillation propagator.

Matches the values used in CUDAProb3 (external/CUDAProb3/src/math/constants.cuh).
"""

import jax

# Enable float64 for precision matching with C++ (which uses double).
# Set to True by default; set environment variable JAX_BARGER_FLOAT32=1 to
# use float32 (useful for GPU memory-constrained environments).
_FLOAT32 = False  # Use float64 for full precision matching with C++
if not _FLOAT32:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

tworttwoGf = 1.52588e-4   # 2√2 G_F [eV² · cm³ / (mol · GeV)]
LoEfac     = 2.534        # 1/(2ħc) [GeV / (eV² · km)]
R_earth    = 6371.0       # Earth radius [km]
h_prod     = 15.0         # atmospheric neutrino production height [km]
epsilon    = 5.0e-9       # small shift to break mass degeneracies
MAX_LAYERS = 10           # max possible: atm + 4 shells in + deepest + 4 shells out
