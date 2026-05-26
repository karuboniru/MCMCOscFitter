"""JAX chi2 matching C++ chi2fittestCU: 400×480 fine grid → 40×40 rebin → 10×12.

Runs in fp32 (OSCILLATION_FP=float) to match C++ default precision.
Compares IH hypothesis chi2 against NH Asimov data.
"""

import sys, os, time
os.environ['JAX_BARGER_FLOAT32'] = '1'

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(REPO, 'build', 'pybind'))
sys.path.insert(0, REPO)

import numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
import mcmcoscfitter as mof

from jax_barger.barger import oscillation_probabilities, oscillation_prob_layer
from jax_barger.earth import default_prem, precompute_path_data
from jax_barger.event_rate import event_rate, poisson_chi2, rebin_2d
from jax_barger.pmns import build_pmns, build_dm, compute_mass_order

print(f"JAX dtype: {jax.numpy.ones(1).dtype}")
print(f"x64 enabled: {jax.config.read('jax_enable_x64')}")

# ── Match C++ FitConfig exactly: 400 E × 480 cosθ, rebin 40×40 → 10×12 ──
N_E_FINE   = 400
N_COS_FINE = 480
E_REBIN    = 40
C_REBIN    = 40

scale = float(mof.scale_factor_6y)

E_edges_fine = mof.logspace(0.1, 20.0, N_E_FINE + 1)
C_edges_fine = mof.linspace(-1.0, 1.0, N_COS_FINE + 1)
E_c_fine = np.array(mof.to_center(E_edges_fine), dtype=np.float32)
C_c_fine = np.array(mof.to_center(C_edges_fine), dtype=np.float32)

print(f"Fine grid: {N_E_FINE} E × {N_COS_FINE} cosθ = {N_E_FINE * N_COS_FINE} points")
print(f"Analysis:  {N_E_FINE // E_REBIN} E × {N_COS_FINE // C_REBIN} cosθ")

# Load flux/xsec at fine binning
pi = mof.load_physics_input(E_edges_fine, C_edges_fine, scale)
flux = {k: jnp.array(pi[f'flux_{k}'], dtype=jnp.float32) for k in ['numu', 'numubar', 'nue', 'nuebar']}
xsec = {k: jnp.array(pi[f'xsec_{k}'], dtype=jnp.float32) for k in ['numu', 'numubar', 'nue', 'nuebar']}

radii, density, Ye = default_prem()
dist_path, rhoe_path = precompute_path_data(jnp.array(C_c_fine, dtype=jnp.float32), radii, density, Ye)

# ── NH truth parameters (sin² space, matching OscillationParameters.h) ──
NH_TRUTH = {'DM2': 2.455e-3, 'Dm2': 7.53e-5, 'T23': 0.558, 'T13': 2.19e-2, 'T12': 0.307, 'DCP': 1.19 * np.pi}
NH_SIGMA = {'DM2': 0.028e-3, 'Dm2': 0.18e-5, 'T23': 0.018, 'T13': 0.07e-2, 'T12': 0.013, 'DCP': 0.22 * np.pi}
IH_TRUTH = {'DM2': -2.529e-3, 'Dm2': 7.53e-5, 'T23': 0.553, 'T13': 2.19e-2, 'T12': 0.307, 'DCP': 1.19 * np.pi}
IH_SIGMA = {'DM2': 0.029e-3, 'Dm2': 0.18e-5, 'T23': 0.020, 'T13': 0.07e-2, 'T12': 0.013, 'DCP': 0.22 * np.pi}

NAMES = ['DM2', 'Dm2', 'T23', 'T13', 'DCP', 'T12']

# Vec for physics: [th12, th13, th23, dcp, dm21sq, dm32sq]
# So mapping from sin²-space param order is:
#   sin2[0]=DM2 -> th32sq
#   sin2[1]=Dm2 -> dm21sq
#   sin2[2]=T23 -> th23
#   sin2[3]=T13 -> th13
#   sin2[4]=DCP -> dcp
#   sin2[5]=T12 -> th12
# Passed to oscillation_probabilities(E, C, th12, th13, th23, dcp, dm21sq, dm32sq, ...)

E_c_jnp = jnp.array(E_c_fine, dtype=jnp.float32)
_vm = jax.vmap(jax.vmap(oscillation_prob_layer,
                         in_axes=(0, None, None, None, None, None, None, None)),
               in_axes=(None, 0, 0, None, None, None, None, None))


def compute_chi2(params_sin2, truth_s, sigma_s, data_s):
    """params_sin2: [DM2, Dm2, T23, T13, DCP, T12] in sin² space with float32.

    Returns Poisson chi2 + pull penalty.
    """
    tv = jnp.array([truth_s[n] for n in NAMES], dtype=jnp.float32)
    sv = jnp.array([sigma_s[n] for n in NAMES], dtype=jnp.float32)

    # Convert angles from sin² to radians
    th12 = jnp.arcsin(jnp.sqrt(jnp.clip(params_sin2[5], 0.0, 1.0)))
    th13 = jnp.arcsin(jnp.sqrt(jnp.clip(params_sin2[3], 0.0, 1.0)))
    th23 = jnp.arcsin(jnp.sqrt(jnp.clip(params_sin2[2], 0.0, 1.0)))
    dcp  = params_sin2[4]
    dm21 = params_sin2[1]
    dm32 = params_sin2[0]

    Ur, Ui = build_pmns(th12, th13, th23, dcp)
    dm = build_dm(dm21, dm32)
    order = compute_mass_order(dm)

    Pn = _vm(E_c_jnp, dist_path, rhoe_path, 0, Ur, Ui, dm, order)
    Pa = _vm(E_c_jnp, dist_path, rhoe_path, 1, Ur, -Ui, dm, order)
    P = jnp.transpose(jnp.stack([Pn, Pa], 0), (0, 4, 3, 2, 1))
    ev = event_rate(P, flux, xsec)

    ev_analysis = {
        ch: rebin_2d(ev[ch], E_REBIN, C_REBIN)
        for ch in ['numu', 'numubar', 'nue', 'nuebar']
    }

    chi2_p = sum(poisson_chi2(data_s[ch], ev_analysis[ch])
                 for ch in ['numu', 'numubar', 'nue', 'nuebar'])

    # Pull penalty (cyclic for DCP)
    diff = params_sin2 - tv
    d_dcp = jnp.arctan2(jnp.sin(diff[4]), jnp.cos(diff[4]))
    diff = diff.at[4].set(d_dcp)
    chi2_pull = jnp.sum((diff / sv) ** 2)

    return chi2_p + chi2_pull


# ── Generate NH Asimov data with JAX (fp32) ──
nh_th_jax = jnp.array([NH_TRUTH[n] for n in NAMES], dtype=jnp.float32)

def compute_asimov_data(params_sin2):
    th12 = jnp.arcsin(jnp.sqrt(jnp.clip(params_sin2[5], 0.0, 1.0)))
    th13 = jnp.arcsin(jnp.sqrt(jnp.clip(params_sin2[3], 0.0, 1.0)))
    th23 = jnp.arcsin(jnp.sqrt(jnp.clip(params_sin2[2], 0.0, 1.0)))
    dcp  = params_sin2[4]
    dm21 = params_sin2[1]
    dm32 = params_sin2[0]

    Ur, Ui = build_pmns(th12, th13, th23, dcp)
    dm = build_dm(dm21, dm32)
    order = compute_mass_order(dm)

    Pn = _vm(E_c_jnp, dist_path, rhoe_path, 0, Ur, Ui, dm, order)
    Pa = _vm(E_c_jnp, dist_path, rhoe_path, 1, Ur, -Ui, dm, order)
    P = jnp.transpose(jnp.stack([Pn, Pa], 0), (0, 4, 3, 2, 1))
    ev = event_rate(P, flux, xsec)
    return {ch: jnp.array(rebin_2d(ev[ch], E_REBIN, C_REBIN)) for ch in ['numu', 'numubar', 'nue', 'nuebar']}

t0 = time.time()
data_s = compute_asimov_data(nh_th_jax)
print(f"NH Asimov data: numu sum = {float(jnp.sum(data_s['numu'])):.1f}  ({time.time()-t0:.1f}s)")

# ── JIT compile chi2 functions ──
@jax.jit
def chi2_nh_jit(p):
    return compute_chi2(p, NH_TRUTH, NH_SIGMA, data_s)

@jax.jit
def chi2_ih_jit(p):
    return compute_chi2(p, IH_TRUTH, IH_SIGMA, data_s)

nh_vg = jax.jit(jax.value_and_grad(lambda p: compute_chi2(p, NH_TRUTH, NH_SIGMA, data_s)))
ih_vg = jax.jit(jax.value_and_grad(lambda p: compute_chi2(p, IH_TRUTH, IH_SIGMA, data_s)))

print("\nJIT warmup...")
_ = chi2_nh_jit(nh_th_jax)
_ = chi2_ih_jit(jnp.array([IH_TRUTH[n] for n in NAMES], dtype=jnp.float32))
print("Done.\n")

# ── Chi2 @ truth points ──
print(f"NH data, NH hypo @ NH truth:    {float(chi2_nh_jit(nh_th_jax)):.2f}")
ih_th_jax = jnp.array([IH_TRUTH[n] for n in NAMES], dtype=jnp.float32)
print(f"NH data, IH hypo @ IH truth:    {float(chi2_ih_jit(ih_th_jax)):.2f}")

# ── Evaluate at C++ best-fit parameters ──
cpp_best = jnp.array([-0.002522, 0.000075, 0.547195, 0.021880, 3.673413, 0.307328], dtype=jnp.float32)
print(f"NH data, IH hypo @ C++ best-fit: {float(chi2_ih_jit(cpp_best)):.4f}")

# Counter for pull-only check
def pull_only(p, truth_s, sigma_s):
    tv = jnp.array([truth_s[n] for n in NAMES], dtype=jnp.float32)
    sv = jnp.array([sigma_s[n] for n in NAMES], dtype=jnp.float32)
    diff = p - tv
    d_dcp = jnp.arctan2(jnp.sin(diff[4]), jnp.cos(diff[4]))
    diff = diff.at[4].set(d_dcp)
    return jnp.sum((diff / sv) ** 2)

print(f"  pull-only @ C++ best-fit:      {float(pull_only(cpp_best, IH_TRUTH, IH_SIGMA)):.4f}")

# ── IH fit (z-space L-BFGS-B, matching compare_fit_fine approach) ──
print(f"\n=== IH fit to NH data (fp32, 400×480 fine grid) ===")
truth_ih_v = jnp.array([IH_TRUTH[n] for n in NAMES], dtype=jnp.float32)
sigma_ih_v = jnp.array([IH_SIGMA[n] for n in NAMES], dtype=jnp.float32)

def make_chi2_ih_z(z):
    p = truth_ih_v + z * sigma_ih_v
    return compute_chi2(p, IH_TRUTH, IH_SIGMA, data_s)

chi2_ih_z_vg = jax.jit(jax.value_and_grad(make_chi2_ih_z))

# Warmup
_ = chi2_ih_z_vg(jnp.zeros(6, dtype=jnp.float32))

z_starts = [
    np.array([-2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    np.array([2.0, -2.0, -2.0, -2.0, -2.0, -2.0], dtype=np.float32),
    np.array([-5.0, 0.0, 3.0, -3.0, 0.0, 4.0], dtype=np.float32),
    np.array([5.0, 0.0, -3.0, 3.0, 0.0, -4.0], dtype=np.float32),
]

best_ih_chi2 = float('inf')
best_ih_z = None

for i, z0 in enumerate(z_starts):
    c = [0]
    def objg(z):
        c[0] += 1
        zz = jnp.array(z, dtype=jnp.float32)
        f, g = chi2_ih_z_vg(zz)
        return float(f), np.array(g, dtype=np.float32)

    t0 = time.time()
    r = minimize(objg, z0, method='L-BFGS-B', jac=True,
                 options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8})
    dt = time.time() - t0
    print(f"  IH start {i+1}: chi2={r.fun:.4f} in {c[0]} evals, {dt:.1f}s")
    if r.fun < best_ih_chi2:
        best_ih_chi2 = r.fun
        best_ih_z = r.x

# ── Also NH fit to NH data (for reference) ──
print(f"\n=== NH fit to NH data (reference) ===")
truth_nh_v = jnp.array([NH_TRUTH[n] for n in NAMES], dtype=jnp.float32)
sigma_nh_v = jnp.array([NH_SIGMA[n] for n in NAMES], dtype=jnp.float32)

def make_chi2_nh_z(z):
    p = truth_nh_v + z * sigma_nh_v
    return compute_chi2(p, NH_TRUTH, NH_SIGMA, data_s)

chi2_nh_z_vg = jax.jit(jax.value_and_grad(make_chi2_nh_z))
_ = chi2_nh_z_vg(jnp.zeros(6, dtype=jnp.float32))

z0_nh = np.array([2.0, -2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
c_nh = [0]
def objg_nh(z):
    c_nh[0] += 1
    zz = jnp.array(z, dtype=jnp.float32)
    f, g = chi2_nh_z_vg(zz)
    print(f"\r  NH eval {c_nh[0]}: chi2={float(f):.4f} |g|={float(jnp.linalg.norm(g)):.2e}  ", end="", flush=True)
    return float(f), np.array(g, dtype=np.float32)

t0 = time.time()
r_nh = minimize(objg_nh, z0_nh, method='L-BFGS-B', jac=True,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
dt_nh = time.time() - t0
print()

# ── Summary ──
print(f"\n{'='*60}")
print(f"JAX fp32, 400E × 480cosθ → rebin 40×40 → 10×12, 6y exposure")
print(f"{'='*60}")
print(f"NH fit min chi2: {r_nh.fun:.4f}")
print(f"IH fit min chi2: {best_ih_chi2:.4f}")
print(f"Δχ²(IH−NH):      {best_ih_chi2 - r_nh.fun:.4f}")
print(f"Significance:     {np.sqrt(max(0, best_ih_chi2 - r_nh.fun)):.1f} σ")
print()
print(f"C++ chi2fittestCU  IH min chi2: 23.49")
print(f"C++ chi2fittestCU  Δχ²(IH−NH):  {23.49 - 0:.2f}")
