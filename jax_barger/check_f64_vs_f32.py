"""Compare JAX f64 vs JAX f32 oscillation probabilities (same code, different dtype)."""
import sys, os, time
import numpy as np

REPO = '/var/home/yan/codes/MCMCOscFitter'
sys.path.insert(0, os.path.join(REPO, 'build', 'pybind'))

import jax, jax.numpy as jnp

# ── Params (use fp64 values, then cast) ──
DM2    = -0.002522
Dm2    =  0.000075
T23_s2 =  0.547195
T13_s2 =  0.021880
T12_s2 =  0.307328
DCP    =  3.673413

th12 = np.float64(np.arcsin(np.sqrt(T12_s2)))
th13 = np.float64(np.arcsin(np.sqrt(T13_s2)))
th23 = np.float64(np.arcsin(np.sqrt(T23_s2)))

# Grid: 30E × 15C
E_g = np.logspace(np.log10(0.1), np.log10(20.0), 30, dtype=np.float64)
C_g = np.linspace(-1.0, 1.0, 15, dtype=np.float64)

from jax_barger.earth import default_prem
radii, density, Ye = default_prem()
r = np.array(radii, dtype=np.float64)
d = np.array(density, dtype=np.float64)
y = np.array(Ye, dtype=np.float64)

# We can't easily switch dtype of the whole jax_barger module.
# Instead, run with x64 enabled (f64), then restart with x64 disabled (f32).
# Or: manually cast inputs and use the f32 path via the config module.

# Simpler: just run the script twice, once with JAX_ENABLE_X64=1 and once without.
# Let's do it inline by controlling x64 per-evaluation.

# Import jax_barger after setting x64
# Actually the oscillation_probabilities function from barger.py uses the DTYPE
# from config.py which is set at import time. So we need two separate processes.

# Let's just write output for both in one script by importing twice is tricky.
# Better: write a small helper that we call twice via subprocess.

print("Running JAX f64...")
import subprocess
r64 = subprocess.run([
    sys.executable, '-c', f'''
import os, sys
sys.path.insert(0, "{REPO}/build/pybind")
sys.path.insert(0, "{REPO}")
os.environ.pop("JAX_BARGER_FLOAT32", None)
os.environ.pop("JAX_ENABLE_X64", None)

import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax_barger.barger import oscillation_probabilities
from jax_barger.earth import default_prem

r, d, y = default_prem()
P = np.array(oscillation_probabilities(
    jnp.array({E_g.tolist()}), jnp.array({C_g.tolist()}),
    {th12}, {th13}, {th23}, {DCP}, {Dm2}, {DM2}, r, d, y))
print("P_f64 =", P.tobytes().hex())
'''], capture_output=True, text=True, timeout=120,
    env={{**os.environ, 'PYTHONPATH': f'{REPO}/build/pybind:{REPO}'}})

print("Running JAX f32...")
r32 = subprocess.run([
    sys.executable, '-c', f'''
import os, sys
sys.path.insert(0, "{REPO}/build/pybind")
sys.path.insert(0, "{REPO}")
os.environ["JAX_BARGER_FLOAT32"] = "1"

import jax.numpy as jnp
import numpy as np
from jax_barger.barger import oscillation_probabilities
from jax_barger.earth import default_prem

r, d, y = default_prem()
P = np.array(oscillation_probabilities(
    jnp.array({E_g.tolist()}), jnp.array({C_g.tolist()}),
    {th12}, {th13}, {th23}, {DCP}, {Dm2}, {DM2}, r, d, y))
print("P_f32 =", P.tobytes().hex())
'''], capture_output=True, text=True, timeout=120,
    env={{**os.environ, 'PYTHONPATH': f'{REPO}/build/pybind:{REPO}'}})

for line in r64.stdout.splitlines():
    if line.startswith('P_f64'):
        P_f64 = np.frombuffer(bytes.fromhex(line.split('= ')[1]), dtype=np.float64).reshape(2, 3, 3, 30, 15)
for line in r32.stdout.splitlines():
    if line.startswith('P_f32'):
        P_f32 = np.frombuffer(bytes.fromhex(line.split('= ')[1]), dtype=np.float32).reshape(2, 3, 3, 30, 15).astype(np.float64)

# Also warn about x64 issue
if r64.returncode != 0:
    print("STDERR f64:", r64.stderr[-500:])
if r32.returncode != 0:
    print("STDERR f32:", r32.stderr[-500:])
