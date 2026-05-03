"""Bayesian hierarchy comparison: NH vs IH fit to NH Asimov data.

  1. Generate NH Asimov data on fine grid (200E × 120cosθ) → rebin to 10×12.
  2. For each hierarchy hypothesis (NH, IH):
     a. Find MAP via L-BFGS-B with analytical gradients.
     b. Compute Laplace log-evidence at the MAP.
     c. Run adaptive HMC warmup + multi-chain production sampling.
     d. Collect posterior summaries, ESS, R̂, correlation matrix.
  3. Report Bayes factor:  2 ln BF = 2 (ln Z_NH − ln Z_IH).

Usage:
    cd /var/home/yan/codes/MCMCOscFitter/jax_barger
    PYTHONPATH=../build/pybind:.. .venv/bin/python run_hierarchy_mcmc.py --warmup 200 --samples 500 --chains 4
    PYTHONPATH=../build/pybind:.. .venv/bin/python run_hierarchy_mcmc.py --fast --fp32  # fast + float32
"""

import sys, os, time, math, argparse

# ── Handle --fp32 before any JAX imports ──
if '--fp32' in sys.argv:
    os.environ['JAX_BARGER_FLOAT32'] = '1'
    sys.argv.remove('--fp32')

import numpy as np
import jax, jax.numpy as jnp

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(REPO_ROOT, 'build', 'pybind'))
sys.path.insert(0, REPO_ROOT)

import mcmcoscfitter as mof
from jax_barger.earth import default_prem, precompute_path_data
from jax_barger.barger import oscillation_probabilities
from jax_barger.event_rate import event_rate, poisson_chi2, rebin_2d
from jax_barger.mcmc import (build_neg_log_posterior, HMCSampler, _PNAMES,
                             laplace_log_evidence, find_map, format_correlation)


_CHANNELS = ['numu', 'numubar', 'nue', 'nuebar']


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description='Hierarchy comparison: HMC + Bayes factor')
parser.add_argument('--warmup', type=int, default=200, help='Warmup steps per hierarchy')
parser.add_argument('--samples', type=int, default=500, help='Production samples per chain')
parser.add_argument('--chains', type=int, default=4, help='Number of chains')
parser.add_argument('--leapfrog', type=int, default=15, help='Leapfrog steps per proposal')
parser.add_argument('--eps0', type=float, default=0.05, help='Initial step size')
parser.add_argument('--fast', action='store_true',
                    help='Use 10×12 grid for quick testing (default: fine 200×120)')
parser.add_argument('--fp32', action='store_true',
                    help='Use float32 precision (2x less VRAM, ~2-3x faster on GPU)')
args = parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Fine-grid setup (200E × 120cosθ → rebin 20×10 → 10×12 analysis)
# ═══════════════════════════════════════════════════════════════════════════════

if args.fast:
    N_E_FINE, N_COS_FINE, E_REBIN, C_REBIN = 10, 12, 1, 1
    mode_label = "FAST (10×12, no rebinning)"
else:
    N_E_FINE, N_COS_FINE, E_REBIN, C_REBIN = 200, 120, 20, 10
    mode_label = f"FINE ({N_E_FINE}×{N_COS_FINE} → {E_REBIN}×{C_REBIN})"

E_edges = mof.logspace(0.1, 20.0, N_E_FINE + 1)
C_edges = mof.linspace(-1.0, 1.0, N_COS_FINE + 1)
E_c = np.array(mof.to_center(E_edges))
C_c = np.array(mof.to_center(C_edges))
scale = float(mof.scale_factor_6y)
radii, density, Ye = default_prem()

print(f"{'=' * 72}")
print(f"Hierarchy MCMC: NH data, NH vs IH hypothesis")
print(f"{'=' * 72}")
print(f"Grid:       {mode_label}")
print(f"Points:     {N_E_FINE * N_COS_FINE}")
print(f"{'─' * 72}")
print(f"Warmup:  {args.warmup} steps   Samples: {args.samples} × {args.chains} chains")
print(f"Leapfrog: {args.leapfrog} steps/proposal   eps₀ = {args.eps0}")


# ── Load flux/xsec + precompute path data ──
pi = mof.load_physics_input(E_edges, C_edges, scale)
flux = {k: jnp.array(pi[f'flux_{k}']) for k in _CHANNELS}
xsec = {k: jnp.array(pi[f'xsec_{k}']) for k in _CHANNELS}
dist_path, rhoe_path = precompute_path_data(jnp.array(C_c), radii, density, Ye)


# ═══════════════════════════════════════════════════════════════════════════════
# Prior definitions — sin²θ convention (matching compare_fit_fine.py)
# ═══════════════════════════════════════════════════════════════════════════════

NH_TRUTH = {'DM2': 2.455e-3, 'Dm2': 7.53e-5, 'T23': 0.558, 'T13': 2.19e-2,
            'T12': 0.307, 'DCP': 1.19 * np.pi}
NH_SIGMA = {'DM2': 0.028e-3, 'Dm2': 0.18e-5, 'T23': 0.018, 'T13': 0.07e-2,
            'T12': 0.013, 'DCP': 0.22 * np.pi}

IH_TRUTH = {'DM2': -2.529e-3, 'Dm2': 7.53e-5, 'T23': 0.553, 'T13': 2.19e-2,
            'T12': 0.307, 'DCP': 1.19 * np.pi}
IH_SIGMA = {'DM2': 0.029e-3, 'Dm2': 0.18e-5, 'T23': 0.020, 'T13': 0.07e-2,
            'T12': 0.013, 'DCP': 0.22 * np.pi}


# ═══════════════════════════════════════════════════════════════════════════════
# Generate NH Asimov data (fine grid → rebin)
# ═══════════════════════════════════════════════════════════════════════════════

def sin2_to_theta(sin2):
    return np.arcsin(np.sqrt(np.clip(sin2, 0.0, 1.0)))

_nh_theta = jnp.array([
    NH_TRUTH['DM2'], NH_TRUTH['Dm2'],
    sin2_to_theta(NH_TRUTH['T23']), sin2_to_theta(NH_TRUTH['T13']),
    NH_TRUTH['DCP'], sin2_to_theta(NH_TRUTH['T12']),
])

print(f"\n{'─' * 72}")
print("Generating NH Asimov data...")
P_nom = oscillation_probabilities(
    jnp.array(E_c), jnp.array(C_c),
    float(_nh_theta[5]), float(_nh_theta[3]), float(_nh_theta[2]),
    float(_nh_theta[4]), float(_nh_theta[1]), float(_nh_theta[0]),
    radii, density, Ye)
ev_nom = event_rate(np.array(P_nom), flux, xsec)
data = {k: jnp.array(rebin_2d(ev_nom[k], E_REBIN, C_REBIN)) for k in _CHANNELS}
for k in _CHANNELS:
    print(f"  {k:>8s} sum = {float(data[k].sum()):.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: compute initial mass matrix diagonal from prior (same as run_mcmc.py)
# ═══════════════════════════════════════════════════════════════════════════════

def prior_mass_diag_h(prior_mean, prior_sigma):
    """Compute initial mass-matrix diagonal M_ii = 1/σ² in θ-space."""
    th23_rad = sin2_to_theta(prior_mean['T23'])
    th13_rad = sin2_to_theta(prior_mean['T13'])
    th12_rad = sin2_to_theta(prior_mean['T12'])
    dth23 = 1.0 / np.sin(2.0 * th23_rad) if np.sin(2.0 * th23_rad) > 0 else 1.0
    dth13 = 1.0 / np.sin(2.0 * th13_rad) if np.sin(2.0 * th13_rad) > 0 else 1.0
    dth12 = 1.0 / np.sin(2.0 * th12_rad) if np.sin(2.0 * th12_rad) > 0 else 1.0
    var = np.array([
        prior_sigma['DM2'] ** 2,
        prior_sigma['Dm2'] ** 2,
        (prior_sigma['T23'] * dth23) ** 2,
        (prior_sigma['T13'] * dth13) ** 2,
        prior_sigma['DCP'] ** 2,
        (prior_sigma['T12'] * dth12) ** 2,
    ])
    return 1.0 / np.maximum(var, 1e-30)


# ═══════════════════════════════════════════════════════════════════════════════
# Run a single hierarchy fit
# ═══════════════════════════════════════════════════════════════════════════════

def run_hierarchy(label, prior_mean, prior_sigma):
    """Fit one hierarchy hypothesis to the (same) NH Asimov data.

    Returns dict with keys: map, ln_z, ln_z_details, chains, diagnostics.
    """
    print(f"\n{'=' * 72}")
    print(f"  {label} HYPOTHESIS")
    print(f"{'=' * 72}")

    # --- Build neg-log posterior ---
    nlp_raw = build_neg_log_posterior(
        jnp.array(E_c), jnp.array(C_c),
        dist_path, rhoe_path, flux, xsec, data,
        prior_mean, prior_sigma, E_REBIN, C_REBIN)

    # Quick JIT warmup for the neg-log-prob
    _nlp_jit = jax.jit(nlp_raw)
    z0 = jnp.array([prior_mean[n] if n in ['DM2', 'Dm2', 'DCP']
                    else sin2_to_theta(prior_mean[n])
                    for n in _PNAMES])
    _ = _nlp_jit(z0)

    # --- Find MAP ---
    print(f"\n  Finding MAP (L-BFGS-B)...")
    t0 = time.time()
    theta_map, map_res = find_map(nlp_raw, np.array(z0),
                                  bounds=MAP_BOUNDS, maxiter=500)
    dt_map = time.time() - t0
    map_nlp = float(_nlp_jit(jnp.array(theta_map)))
    print(f"    MAP  -logP = {map_nlp:.4f}")
    print(f"    MAP  |grad| = {float(np.linalg.norm(np.array(jax.grad(nlp_raw)(jnp.array(theta_map))))):.2e}")
    print(f"    Evals: {map_res.nfev}  time: {dt_map:.1f}s")

    # --- Laplace evidence ---
    print(f"\n  Computing Laplace evidence...")
    t0 = time.time()
    import gc; gc.collect()
    ln_z, ln_z_det = laplace_log_evidence(nlp_raw, jnp.array(theta_map))
    dt_lz = time.time() - t0
    print(f"    ln Z  = {ln_z:6.1f}")
    print(f"    -log L(θ*) = {ln_z_det['neg_log_map']:.2f}")
    print(f"    ln|H| = {ln_z_det['log_det_hessian']:.2f}")
    print(f"    time: {dt_lz:.1f}s")

    # --- HMC sampling ---
    init_mass = prior_mass_diag_h(prior_mean, prior_sigma)

    # Initial position: MAP with small perturbation
    np.random.seed(hash(label) % 2**31)
    pert = np.random.randn(6) * 0.5
    z_init = np.array(theta_map) + pert * np.array([
        prior_sigma['DM2'], prior_sigma['Dm2'],
        0.05, 0.01, 0.1, 0.05,
    ])
    z_init[2] = np.clip(z_init[2], 0.01, np.pi / 2 - 0.01)
    z_init[3] = np.clip(z_init[3], 0.001, np.pi / 2 - 0.001)
    z_init[5] = np.clip(z_init[5], 0.01, np.pi / 2 - 0.01)
    z_init[4] = np.arctan2(np.sin(z_init[4]), np.cos(z_init[4]))

    sampler = HMCSampler(nlp_raw, eps_0=args.eps0,
                         n_leapfrog=args.leapfrog,
                         target_accept=0.651,
                         initial_mass_diag=init_mass)

    print(f"\n  HMC Warmup ({args.warmup} steps)...")
    t0 = time.time()
    import gc; gc.collect()
    sampler.warmup(n_steps=args.warmup, z_init=z_init,
                   adapt_step=True, adapt_mass=False)
    dt_wu = time.time() - t0
    print(f"    Warmup time: {dt_wu:.1f}s")

    print(f"\n  HMC Sampling ({args.samples} × {args.chains} chains)...")
    t0 = time.time()
    chains = sampler.sample(n_samples=args.samples, n_chains=args.chains)
    dt_sm = time.time() - t0
    print(f"    Sampling time: {dt_sm:.1f}s")

    sampler.diagnostics()

    corr = sampler.correlation_matrix()

    return {
        'label': label,
        'map': theta_map,
        'map_nlp': map_nlp,
        'ln_z': ln_z,
        'ln_z_det': ln_z_det,
        'chains': chains,
        'diagnostics': sampler.diagnostics_,
        'correlation': corr,
        'time_map': dt_map,
        'time_lnz': dt_lz,
        'time_warmup': dt_wu,
        'time_sample': dt_sm,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAP bounds in θ-space
# ═══════════════════════════════════════════════════════════════════════════════

MAP_BOUNDS = [
    (-1.0, 1.0),            # DM2  (allows both NH and IH)
    (1e-7, 1e-3),           # Dm2  (always positive)
    (0.1, np.pi / 2 - 0.01),  # θ₂₃
    (0.01, np.pi / 2 - 0.01),  # θ₁₃
    (-np.pi, np.pi),        # δCP
    (0.1, np.pi / 2 - 0.01),  # θ₁₂
]


# ═══════════════════════════════════════════════════════════════════════════════
# Run both fits
# ═══════════════════════════════════════════════════════════════════════════════

t_total = time.time()

res_nh = run_hierarchy("NH", NH_TRUTH, NH_SIGMA)
res_ih = run_hierarchy("IH", IH_TRUTH, IH_SIGMA)

dt_total = time.time() - t_total


# ═══════════════════════════════════════════════════════════════════════════════
# Bayes factor
# ═══════════════════════════════════════════════════════════════════════════════

ln_bf = res_nh['ln_z'] - res_ih['ln_z']
bf_2ln = 2.0 * ln_bf

print(f"\n{'=' * 72}")
print(f"BAYESIAN MODEL COMPARISON")
print(f"{'=' * 72}")
print(f"  NH log-evidence  ln Z(NH) = {res_nh['ln_z']:.4f}")
print(f"  IH log-evidence  ln Z(IH) = {res_ih['ln_z']:.4f}")
print(f"  {'─' * 42}")
print(f"  Log Bayes factor  ln BF    = {ln_bf:.4f}")
print(f"  2 ln BF                    = {bf_2ln:.2f}")
print(f"  {'─' * 42}")

# Jeffreys interpretation
if bf_2ln < 2:
    scale = "Barely worth mentioning (BF < 3)"
elif bf_2ln < 6:
    scale = "Substantial (3 < BF < 20)"
elif bf_2ln < 10:
    scale = "Strong (20 < BF < 150)"
else:
    scale = "Decisive (BF > 150)"
print(f"  Interpretation: {scale}")
print(f"  (Data favor NH over IH by factor exp({ln_bf:.1f}) = {np.exp(ln_bf):.2e})")


# ═══════════════════════════════════════════════════════════════════════════════
# Posterior comparison
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 72}")
print(f"POSTERIOR COMPARISON (θ-space)")
print(f"{'=' * 72}")

for res, hname in [(res_nh, 'NH'), (res_ih, 'IH')]:
    d = res['diagnostics']
    print(f"\n  {hname} model posteriors:")
    print(f"  {'Param':<8} {'Mean':>18} {'± Std':>14}  {'68% CI':>28}  {'R̂':>6}  {'ESS':>8}")
    print(f"  {'─' * 8} {'─' * 18} {'─' * 14}  {'─' * 28}  {'─' * 6}  {'─' * 8}")
    for name in _PNAMES:
        di = d[name]
        ci68 = f"[{di['ci68'][0]:.4e}, {di['ci68'][1]:.4e}]"
        print(f"  {name:<8} {di['mean']:18.8e} {di['std']:14.4e}  "
              f"{ci68:>28}  {di['rhat']:6.3f}  {di['ess']:8.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Correlation matrices
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 72}")
print(f"CORRELATION MATRICES")
print(f"{'=' * 72}")
print("\n  NH model:")
print(format_correlation(res_nh['correlation'], _PNAMES))
print("\n  IH model:")
print(format_correlation(res_ih['correlation'], _PNAMES))


# ═══════════════════════════════════════════════════════════════════════════════
# Timing summary
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 72}")
print(f"TIMING")
print(f"{'=' * 72}")
for res in [res_nh, res_ih]:
    total = res['time_map'] + res['time_lnz'] + res['time_warmup'] + res['time_sample']
    print(f"  {res['label']}: MAP={res['time_map']:.0f}s  "
          f"Laplace={res['time_lnz']:.0f}s  "
          f"Warmup={res['time_warmup']:.0f}s  "
          f"Sample={res['time_sample']:.0f}s  "
          f"= {total:.0f}s")
print(f"  Total wall time: {dt_total:.0f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════════════

for res in [res_nh, res_ih]:
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f'hmc_chains_{res["label"].lower()}.npz')
    np.savez(out, chains=res['chains'],
             diagnostics=np.array([res['diagnostics']]),
             map=res['map'], ln_z=res['ln_z'], correlation=res['correlation'])
    print(f"  Saved chains to {out}")

print(f"\n{'=' * 72}")
print(f"Done. Bayes factor analysis complete.")
print(f"{'=' * 72}")
