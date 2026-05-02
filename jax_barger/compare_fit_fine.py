"""Fine-binning + rebinning JAX fit, matching the C++ chi2fit approach.

The C++ production code (chi2fit) evaluates oscillation probabilities on a fine
grid (400 E × 480 cosθ), then rebins to analysis bins (10 E × 12 cosθ).
This script replicates that workflow in JAX with a reduced but reasonable fine
grid, then fits NH Asimov data under both NH and IH hypotheses.

Usage:
    cd /var/home/yan/codes/MCMCOscFitter/jax_barger
    PYTHONPATH=../build/pybind:.. .venv/bin/python compare_fit_fine.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build', 'pybind'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
import mcmcoscfitter as mof
from jax_barger.barger import oscillation_probabilities, oscillation_prob_layer
from jax_barger.earth import default_prem, precompute_path_data
from jax_barger.event_rate import event_rate, poisson_chi2, rebin_2d
from jax_barger.pmns import build_pmns, build_dm, compute_mass_order

# ================================================================
# Fine binning — reduced from 400×480 to keep JAX evaluation fast
# ================================================================
N_E_FINE   = 200      # C++ uses 400
N_COS_FINE = 120      # C++ uses 480
E_REBIN    = 20       # → 200/20 = 10 analysis E bins
C_REBIN    = 10       # → 120/10 = 12 analysis cosθ bins

E_edges_fine = mof.logspace(0.1, 20.0, N_E_FINE + 1)
C_edges_fine = mof.linspace(-1.0, 1.0, N_COS_FINE + 1)
E_c_fine = np.array(mof.to_center(E_edges_fine))
C_c_fine = np.array(mof.to_center(C_edges_fine))

scale = float(mof.scale_factor_6y)
radii, density, Ye = default_prem()

print(f"Fine grid: {N_E_FINE} E × {N_COS_FINE} cosθ = {N_E_FINE * N_COS_FINE} points")
print(f"Analysis:  {N_E_FINE // E_REBIN} E × {N_COS_FINE // C_REBIN} cosθ")

# Load flux/xsec at FINE binning
pi = mof.load_physics_input(E_edges_fine, C_edges_fine, scale)
flux = {k: jnp.array(pi[f'flux_{k}']) for k in ['numu', 'numubar', 'nue', 'nuebar']}
xsec = {k: jnp.array(pi[f'xsec_{k}']) for k in ['numu', 'numubar', 'nue', 'nuebar']}

# Precompute path data at fine cosθ grid
dist_path, rhoe_path = precompute_path_data(jnp.array(C_c_fine), radii, density, Ye)

# Physics constants
NH_TRUTH = {'DM2': 2.455e-3, 'Dm2': 7.53e-5, 'th23': 0.558, 'th13': 2.19e-2, 'th12': 0.307,
             'DCP': 1.19 * np.pi}
NH_SIGMA = {'DM2': 0.028e-3, 'Dm2': 0.18e-5, 'th23': 0.018, 'th13': 0.07e-2, 'th12': 0.013,
             'DCP': 0.22 * np.pi}
IH_TRUTH = {'DM2': -2.529e-3, 'Dm2': 7.53e-5, 'th23': 0.553, 'th13': 2.19e-2, 'th12': 0.307,
             'DCP': 1.19 * np.pi}
IH_SIGMA = {'DM2': 0.029e-3, 'Dm2': 0.18e-5, 'th23': 0.020, 'th13': 0.07e-2, 'th12': 0.013,
             'DCP': 0.22 * np.pi}
NAMES = ['DM2', 'Dm2', 'th23', 'th13', 'DCP', 'th12']

E_c_jnp = jnp.array(E_c_fine)
_vm = jax.vmap(jax.vmap(oscillation_prob_layer,
                         in_axes=(0, None, None, None, None, None, None, None)),
               in_axes=(None, 0, 0, None, None, None, None, None))


# ================================================================
# Generate NH Asimov data (fine grid → rebin → analysis data)
# ================================================================
def make_analysis_data(params):
    """Compute Asimov event rates on analysis grid from θ-space parameters."""
    P = oscillation_probabilities(
        jnp.array(E_c_fine), jnp.array(C_c_fine),
        params[5], params[3], params[2], params[4], params[1], params[0],
        radii, density, Ye)
    ev = event_rate(np.array(P), flux, xsec)
    return {k: rebin_2d(ev[k], E_REBIN, C_REBIN) for k in ['numu', 'numubar', 'nue', 'nuebar']}


nh_truth_th = jnp.array([NH_TRUTH['DM2'], NH_TRUTH['Dm2'],
                          np.arcsin(np.sqrt(NH_TRUTH['th23'])),
                          np.arcsin(np.sqrt(NH_TRUTH['th13'])),
                          NH_TRUTH['DCP'],
                          np.arcsin(np.sqrt(NH_TRUTH['th12']))])
data_analysis = make_analysis_data(nh_truth_th)
data_s = {k: jnp.array(v) for k, v in data_analysis.items()}
print(f"NH Asimov data: numu sum = {float(data_s['numu'].sum()):.1f}")


# ================================================================
# Chi2 in z-space, with fine binning + rebinning
# ================================================================
def make_chi2_z(truth_s, sigma_s):
    """Build chi2(z) function: z in sigma-units, fine-grid eval + rebin."""
    tv_sin2 = jnp.array([truth_s[n] for n in NAMES])
    sv_sin2 = jnp.array([sigma_s[n] for n in NAMES])

    def chi2_fn(z):
        s2 = tv_sin2 + z * sv_sin2
        p = jnp.array([
            s2[0], s2[1],
            jnp.arcsin(jnp.sqrt(jnp.clip(s2[2], 0.0, 1.0))),
            jnp.arcsin(jnp.sqrt(jnp.clip(s2[3], 0.0, 1.0))),
            s2[4],
            jnp.arcsin(jnp.sqrt(jnp.clip(s2[5], 0.0, 1.0)))
        ])
        Ur, Ui = build_pmns(p[5], p[3], p[2], p[4])
        dm = build_dm(p[1], p[0])
        order = compute_mass_order(dm)

        # Fine-grid oscillation probabilities → event rates on fine grid
        Pn = _vm(E_c_jnp, dist_path, rhoe_path, 0, Ur, Ui, dm, order)
        Pa = _vm(E_c_jnp, dist_path, rhoe_path, 1, Ur, -Ui, dm, order)
        P = jnp.transpose(jnp.stack([Pn, Pa], 0), (0, 4, 3, 2, 1))
        ev = event_rate(P, flux, xsec)

        # Rebin to analysis bins
        ev_analysis = {
            ch: rebin_2d(ev[ch], E_REBIN, C_REBIN)
            for ch in ['numu', 'numubar', 'nue', 'nuebar']
        }

        # Poisson chi2 on analysis bins
        c = sum(poisson_chi2(data_s[ch], ev_analysis[ch])
                for ch in ['numu', 'numubar', 'nue', 'nuebar'])

        # Pull penalty in z-space: just z^T · z
        return c + jnp.sum(z ** 2)

    return chi2_fn


chi2_nh = make_chi2_z(NH_TRUTH, NH_SIGMA)
chi2_ih = make_chi2_z(IH_TRUTH, IH_SIGMA)

chi2_nh_jit = jax.jit(chi2_nh)
chi2_nh_vg = jax.jit(jax.value_and_grad(chi2_nh))
chi2_ih_jit = jax.jit(chi2_ih)
chi2_ih_vg = jax.jit(jax.value_and_grad(chi2_ih))

# ================================================================
# Warmup
# ================================================================
print("\nJIT warmup...")
_ = chi2_nh_jit(jnp.zeros(6))
_ = chi2_ih_jit(jnp.zeros(6))
print("Done.\n")

# Sanity checks
print(f"NH data, NH hypo @ NH truth:    {float(chi2_nh_jit(jnp.zeros(6))):.2f}")
print(f"NH data, IH hypo @ IH truth:    {float(chi2_ih_jit(jnp.zeros(6))):.2f}")

# ================================================================
# NH fit (reference) — z-space L-BFGS-B
# ================================================================
z_start_nh = np.array([2.0, -2.0, 2.0, 2.0, 2.0, 2.0])
print(f"\n=== NH fit to NH data ===")
print(f"Start chi2: {float(chi2_nh_jit(jnp.array(z_start_nh))):.2f}")

count = [0]
def objg_nh(z):
    count[0] += 1
    zz = jnp.array(z, dtype=jnp.float64)
    f, g = chi2_nh_vg(zz)
    gn = float(jnp.linalg.norm(g))
    print(f"\r  NH eval {count[0]}: chi2={float(f):.2f} |g|={gn:.1e}  ", end="", flush=True)
    return float(f), np.array(g, dtype=np.float64)

t0 = time.time()
r_nh = minimize(objg_nh, z_start_nh, method='L-BFGS-B', jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
dt_nh = time.time() - t0
print()
print(f"NH best: chi2={r_nh.fun:.4f} in {count[0]} evals, {dt_nh:.1f}s")

# ================================================================
# IH fit — z-space L-BFGS-B, multi-start
# ================================================================
z_starts_ih = [
    np.array([-2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([2.0, -2.0, -2.0, -2.0, -2.0, -2.0]),
    np.array([-5.0, 0.0, 3.0, -3.0, 0.0, 4.0]),
    np.array([5.0, 0.0, -3.0, 3.0, 0.0, -4.0]),
]

best_ih_chi2 = float('inf')
best_ih_z = None

for i, z0 in enumerate(z_starts_ih):
    c = [0]
    def objg_ih(z):
        c[0] += 1
        zz = jnp.array(z, dtype=jnp.float64)
        f, g = chi2_ih_vg(zz)
        return float(f), np.array(g, dtype=np.float64)

    t0 = time.time()
    r = minimize(objg_ih, z0, method='L-BFGS-B', jac=True,
                 options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
    dt = time.time() - t0
    print(f"  IH start {i + 1}: chi2={r.fun:.2f} in {c[0]} evals, {dt:.1f}s")
    if r.fun < best_ih_chi2:
        best_ih_chi2 = r.fun
        best_ih_z = r.x

# ================================================================
# Summary
# ================================================================
print(f"\n{'=' * 60}")
print(f"Fine binning: {N_E_FINE}E × {N_COS_FINE}cosθ → rebin {E_REBIN}×{C_REBIN} → analysis")
print(f"Exposure: 6 years (scale_factor_6y)")
print(f"NH best chi2: {r_nh.fun:.2f}")
print(f"IH best chi2: {best_ih_chi2:.2f}")
print(f"Δχ²(IH−NH) = {best_ih_chi2 - r_nh.fun:.2f}")
print(f"Significance ≈ √Δχ² = {np.sqrt(max(0, best_ih_chi2 - r_nh.fun)):.1f} σ")
print(f"\n(For comparison: original 10×12 direct eval gave Δχ² = 571 ≈ 24σ)")
print(f"Fine binning captures bin-averaged oscillation, direct center-point")
print(f"evaluation misses sub-bin structure → biased χ² values.")
