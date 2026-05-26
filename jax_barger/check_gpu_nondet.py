"""Check GPU float non-determinism: run JAX + C++ twice each, compare self-consistency."""
import sys, os, time
import numpy as np

REPO = '/var/home/yan/codes/MCMCOscFitter'
sys.path.insert(0, os.path.join(REPO, 'build', 'pybind'))
sys.path.insert(0, REPO)

import mcmcoscfitter as mof
import jax, jax.numpy as jnp
from jax_barger.barger import oscillation_probabilities
from jax_barger.earth import default_prem
from jax_barger.event_rate import event_rate, poisson_chi2, rebin_2d

# ── Grid: 30 E × 15 cosθ (small, fast to run twice) ──
E_grid = np.logspace(np.log10(0.1), np.log10(20.0), 30, dtype=np.float64)
C_grid = np.linspace(-1.0, 1.0, 15, dtype=np.float64)

# ── Params at C++ best-fit from chi2fittestCU (IH fit) ──
DM2    = -0.002522
Dm2    =  0.000075
T23_s2 =  0.547195
T13_s2 =  0.021880
T12_s2 =  0.307328
DCP    =  3.673413

th12 = np.arcsin(np.sqrt(T12_s2))
th13 = np.arcsin(np.sqrt(T13_s2))
th23 = np.arcsin(np.sqrt(T23_s2))

radii, density, Ye = default_prem()
prem = {'radii': radii, 'density': density, 'Ye': Ye}

# ════════════════════════════════════════════════════════════════
# JAX: run twice, compare self
# ════════════════════════════════════════════════════════════════
@jax.jit
def jax_prob(E, C):
    return oscillation_probabilities(E, C, th12, th13, th23, DCP, Dm2, DM2,
                                     prem['radii'], prem['density'], prem['Ye'])

Ej = jnp.array(E_grid); Cj = jnp.array(C_grid)
_ = jax_prob(Ej, Cj)  # warmup

t0 = time.time()
P1 = np.array(jax_prob(Ej, Cj))
P2 = np.array(jax_prob(Ej, Cj))
dt_jax = time.time() - t0

jax_self_diff = np.abs(P1 - P2)
print(f"JAX self-consistency (f64, 30E×15C, {dt_jax:.1f}s):")
print(f"  max_abs={np.max(jax_self_diff):.2e}")
print(f"  rms    ={np.sqrt(np.mean(jax_self_diff**2)):.2e}")
print(f"  n_uneq ={np.sum(jax_self_diff > 0)} / {P1.size}")

# ════════════════════════════════════════════════════════════════
# C++: run twice, compare self
# ════════════════════════════════════════════════════════════════
E_c = E_grid.astype(np.float32)
C_c = C_grid.astype(np.float32)

# Build bin edges from centers (as validate.py does)
n = len(E_grid)
E_edges = np.zeros(n + 1, dtype=np.float64)
E_edges[0] = E_grid[0] - (E_grid[1]-E_grid[0])/2
if E_edges[0] < 1e-3: E_edges[0] = 1e-3
E_edges[-1] = E_grid[-1] + (E_grid[-1]-E_grid[-2])/2
for i in range(1, n): E_edges[i] = (E_grid[i-1]+E_grid[i])/2

n = len(C_grid)
C_edges = np.zeros(n + 1, dtype=np.float64)
C_edges[0] = C_grid[0] - (C_grid[1]-C_grid[0])/2
if C_edges[0] < -1.0: C_edges[0] = -1.0
C_edges[-1] = C_grid[-1] + (C_grid[-1]-C_grid[-2])/2
if C_edges[-1] > 1.0: C_edges[-1] = 1.0
for i in range(1, n): C_edges[i] = (C_grid[i-1]+C_grid[i])/2

p_mof = mof.OscillationParameters()
p_mof.set_param(mof.Param(DM2, Dm2, T23_s2, T13_s2, T12_s2, DCP))
prop = mof.ParProb3ppOscillation(E_c, C_c)

t0 = time.time()
P_cpp1 = np.array(prop.get_prob_hists_3f(E_edges.tolist(), C_edges.tolist(), p_mof))
P_cpp2 = np.array(prop.get_prob_hists_3f(E_edges.tolist(), C_edges.tolist(), p_mof))
dt_cpp = time.time() - t0

cpp_self_diff = np.abs(P_cpp1 - P_cpp2)
print(f"\nC++ self-consistency (f32 CUDA, 30E×15C, {dt_cpp:.1f}s):")
print(f"  max_abs={np.max(cpp_self_diff):.2e}")
print(f"  rms    ={np.sqrt(np.mean(cpp_self_diff**2)):.2e}")
print(f"  n_uneq ={np.sum(cpp_self_diff > 0)} / {P_cpp1.size}")

# ════════════════════════════════════════════════════════════════
# Cross-comparison: JAX vs C++
# ════════════════════════════════════════════════════════════════
cross_diff = np.abs(P1 - P_cpp1)
mask = np.abs(P_cpp1) > 1e-6
rel_cross = cross_diff[mask] / np.abs(P_cpp1[mask])

print(f"\nCross JAX(64) vs C++(32):")
print(f"  max_abs={np.max(cross_diff):.2e}")
print(f"  rms    ={np.sqrt(np.mean(cross_diff**2)):.2e}")
print(f"  max_rel={np.max(rel_cross):.2e}" if len(rel_cross)>0 else "  max_rel=N/A")

# ── Summary ──
print(f"\n{'='*50}")
print(f"GPU float non-determinism contribution:")
print(f"  JAX (f64) self-diff rms: {np.sqrt(np.mean(jax_self_diff**2)):.1e}")
print(f"  C++ (f32) self-diff rms: {np.sqrt(np.mean(cpp_self_diff**2)):.1e}")
print(f"  Cross    diff      rms: {np.sqrt(np.mean(cross_diff**2)):.1e}")
print(f"→ GPU nondeterminism < cross-diff by factor "
      f"{np.sqrt(np.mean(cross_diff**2))/max(np.sqrt(np.mean(jax_self_diff**2)), 1e-30):.0f}x")
