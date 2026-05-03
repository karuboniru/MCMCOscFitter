"""Run gradient-accelerated HMC sampling for neutrino oscillation parameters.

Physics context: Nh Asimov data, 6-parameter θ-space sampling with Gaussian
pull priors from PDG.  Two grid modes:

  --fast   10 E × 12 cosθ direct evaluation (bin centers, no rebinning)
           Fast (≈2-3 ms/eval GPU), for quick tests and tuning.
  --fine   200 E × 120 cosθ fine grid → rebin to 10×12 analysis bins
           Physically correct, ≈50-100 ms/eval GPU, for production.

Usage:
    cd /var/home/yan/codes/MCMCOscFitter/jax_barger
    PYTHONPATH=../build/pybind:.. .venv/bin/python run_mcmc.py --fast --warmup 200 --samples 500 --chains 2
    PYTHONPATH=../build/pybind:.. .venv/bin/python run_mcmc.py --fast --fp32  # float32 mode

Environment:
    JAX_BARGER_FLOAT32=1   Use float32 (2× less VRAM, ~2-3× faster on GPU)
"""

import sys, os, time, argparse

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
from jax_barger.mcmc import build_neg_log_posterior, HMCSampler, _PNAMES


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description='HMC MCMC for neutrino oscillations')
_group = parser.add_mutually_exclusive_group(required=True)
_group.add_argument('--fast', action='store_true', help='Fast mode: 10x12 direct bin centers')
_group.add_argument('--fine', action='store_true', help='Fine mode: 200x120 + rebin to 10x12')
parser.add_argument('--warmup', type=int, default=200, help='Warmup steps')
parser.add_argument('--samples', type=int, default=1000, help='Production samples per chain')
parser.add_argument('--chains', type=int, default=2, help='Number of chains')
parser.add_argument('--leapfrog', type=int, default=20, help='Leapfrog steps per proposal')
parser.add_argument('--eps0', type=float, default=0.05, help='Initial step size')
parser.add_argument('--adapt-mass', action='store_true',
                    help='Enable sample-based mass matrix adaptation (experimental)')
parser.add_argument('--fp32', action='store_true',
                    help='Use float32 precision (2x less VRAM, ~2-3x faster on GPU)')
args = parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Grid setup
# ═══════════════════════════════════════════════════════════════════════════════

radii, density, Ye = default_prem()
scale = float(mof.scale_factor_6y)

if args.fast:
    N_E, N_COS, E_REBIN, C_REBIN = 10, 12, 1, 1
    E_edges = mof.logspace(0.1, 20.0, N_E + 1)
    C_edges = mof.linspace(-1.0, 1.0, N_COS + 1)
    E_c = np.array(mof.to_center(E_edges))
    C_c = np.array(mof.to_center(C_edges))
    mode_label = "FAST (10×12 bin centers, no rebinning)"
else:
    N_E, N_COS, E_REBIN, C_REBIN = 200, 120, 20, 10
    E_edges = mof.logspace(0.1, 20.0, N_E + 1)
    C_edges = mof.linspace(-1.0, 1.0, N_COS + 1)
    E_c = np.array(mof.to_center(E_edges))
    C_c = np.array(mof.to_center(C_edges))
    mode_label = f"FINE ({N_E}×{N_COS} → rebin {E_REBIN}×{C_REBIN} → {N_E // E_REBIN}×{N_COS // C_REBIN} analysis)"

print(f"Mode:        {mode_label}")
print(f"Grid:        {N_E} E × {N_COS} cosθ = {N_E * N_COS} points")
print(f"Chains:      {args.chains} chains × {args.samples} samples each")
print(f"Warmup:      {args.warmup} steps")
print(f"Leapfrog:    {args.leapfrog} steps/proposal")
print(f"Scale:       {scale:.2e}")

# ── Load flux/xsec ──
pi = mof.load_physics_input(E_edges, C_edges, scale)
flux = {k: jnp.array(pi[f'flux_{k}']) for k in ['numu', 'numubar', 'nue', 'nuebar']}
xsec = {k: jnp.array(pi[f'xsec_{k}']) for k in ['numu', 'numubar', 'nue', 'nuebar']}

# ── Precompute path data ──
dist_path, rhoe_path = precompute_path_data(jnp.array(C_c), radii, density, Ye)


# ═══════════════════════════════════════════════════════════════════════════════
# Priors — PDG central values (sin²θ convention)
# ═══════════════════════════════════════════════════════════════════════════════

PRIOR_MEAN = {
    'DM2':  2.455e-3,
    'Dm2':  7.53e-5,
    'T23':  0.558,
    'T13':  2.19e-2,
    'T12':  0.307,
    'DCP':  1.19 * np.pi,
}
PRIOR_SIGMA = {
    'DM2': 0.028e-3,
    'Dm2': 0.18e-5,
    'T23': 0.018,
    'T13': 0.07e-2,
    'T12': 0.013,
    'DCP': 0.22 * np.pi,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Generate Asimov data at prior mean
# ═══════════════════════════════════════════════════════════════════════════════

def sin2_to_theta(sin2):
    return np.arcsin(np.sqrt(np.clip(sin2, 0.0, 1.0)))

_asimov_th = jnp.array([
    PRIOR_MEAN['DM2'],
    PRIOR_MEAN['Dm2'],
    sin2_to_theta(PRIOR_MEAN['T23']),
    sin2_to_theta(PRIOR_MEAN['T13']),
    PRIOR_MEAN['DCP'],
    sin2_to_theta(PRIOR_MEAN['T12']),
])

P_nom = oscillation_probabilities(
    jnp.array(E_c), jnp.array(C_c),
    float(_asimov_th[5]), float(_asimov_th[3]), float(_asimov_th[2]),
    float(_asimov_th[4]), float(_asimov_th[1]), float(_asimov_th[0]),
    radii, density, Ye)

ev_nom = event_rate(np.array(P_nom), flux, xsec)
data = {k: jnp.array(rebin_2d(ev_nom[k], E_REBIN, C_REBIN))
        for k in ['numu', 'numubar', 'nue', 'nuebar']}

for k in data:
    print(f"  {k:>8s} sum = {float(data[k].sum()):.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Build neg-log-posterior + sanity
# ═══════════════════════════════════════════════════════════════════════════════

print("\nBuilding and JIT-compiling neg-log-posterior...")
neg_log_prob_raw = build_neg_log_posterior(
    jnp.array(E_c), jnp.array(C_c),
    dist_path, rhoe_path,
    flux, xsec, data,
    PRIOR_MEAN, PRIOR_SIGMA,
    E_REBIN, C_REBIN)

# Quick JIT warmup
neg_log_prob_jit = jax.jit(neg_log_prob_raw)
grad_jit = jax.jit(jax.grad(neg_log_prob_raw))
_ = neg_log_prob_jit(_asimov_th)
_ = grad_jit(_asimov_th)

nll_truth = float(neg_log_prob_jit(_asimov_th))
print(f"  -log P at prior mean: {nll_truth:.4f}  (≈0 for Asimov)")
g_norm = float(np.linalg.norm(np.array(grad_jit(_asimov_th))))
print(f"  |grad| at prior mean: {g_norm:.2e}  (≈0 for Asimov)")


# ═══════════════════════════════════════════════════════════════════════════════
# HMC Sampling
# ═══════════════════════════════════════════════════════════════════════════════

# Initial position: prior mean + 1σ perturbation
np.random.seed(42)
_perturb = np.random.randn(6) * 1.0
z_init = np.array(_asimov_th) + _perturb * np.array([
    PRIOR_SIGMA['DM2'],
    PRIOR_SIGMA['Dm2'],
    0.05,  # θ₂₃ perturbation ~3°
    0.01,  # θ₁₃ perturbation ~0.6°
    0.1,   # δCP perturbation
    0.05,  # θ₁₂ perturbation ~3°
])
z_init[2] = np.clip(z_init[2], 0.01, np.pi / 2 - 0.01)
z_init[3] = np.clip(z_init[3], 0.001, np.pi / 2 - 0.001)
z_init[5] = np.clip(z_init[5], 0.01, np.pi / 2 - 0.01)
z_init[4] = np.arctan2(np.sin(z_init[4]), np.cos(z_init[4]))

print(f"\n{'=' * 72}")
print(f"HMC Sampling")
print(f"{'=' * 72}")

# Compute initial mass matrix diagonal from prior sigmas.
# M_ii = 1 / σ²_θ  so that tight directions get larger mass
# and move more slowly in leapfrog steps.
_deriv_th23 = 1.0 / np.sin(2.0 * float(_asimov_th[2]))
_deriv_th13 = 1.0 / np.sin(2.0 * float(_asimov_th[3]))
_deriv_th12 = 1.0 / np.sin(2.0 * float(_asimov_th[5]))
_prior_var_theta = np.array([
    PRIOR_SIGMA['DM2'] ** 2,
    PRIOR_SIGMA['Dm2'] ** 2,
    (PRIOR_SIGMA['T23'] * _deriv_th23) ** 2,
    (PRIOR_SIGMA['T13'] * _deriv_th13) ** 2,
    PRIOR_SIGMA['DCP'] ** 2,
    (PRIOR_SIGMA['T12'] * _deriv_th12) ** 2,
])
_initial_mass_diag = 1.0 / np.maximum(_prior_var_theta, 1e-30)
print(f"Prior var θ:  {_prior_var_theta}")
print(f"Init mass dg: {_initial_mass_diag}")
print(f"Init θ:   DM2={z_init[0]:.4e}  Dm2={z_init[1]:.4e}")
print(f"           θ₂₃={z_init[2]:.4f}  θ₁₃={z_init[3]:.4f}  "
      f"δCP={z_init[4]:.4f}  θ₁₂={z_init[5]:.4f}")
print(f"Init -logP: {float(neg_log_prob_jit(jnp.array(z_init))):.3f}")

sampler = HMCSampler(neg_log_prob_raw,
                     eps_0=args.eps0, n_leapfrog=args.leapfrog,
                     target_accept=0.651,
                     initial_mass_diag=_initial_mass_diag)

t_start = time.time()

print(f"\n--- Warmup ({args.warmup} steps) ---")
sampler.warmup(n_steps=args.warmup, z_init=z_init,
               adapt_step=True,
               adapt_mass=args.adapt_mass)

dt_warmup = time.time() - t_start
print(f"Warmup time: {dt_warmup:.1f}s")

print(f"\n--- Production ({args.samples} samples × {args.chains} chains) ---")
t_sample = time.time()
chains = sampler.sample(n_samples=args.samples, n_chains=args.chains)
dt_sample = time.time() - t_sample
print(f"Sampling time: {dt_sample:.1f}s")
print(f"Total time:    {dt_warmup + dt_sample:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════════════════════

sampler.diagnostics()

# Compare with truth
print(f"\n{'=' * 72}")
print("Bias vs prior mean (Asimov truth)")
print(f"{'=' * 72}")
print(f"{'Param':<8} {'Truth':>12} {'Posterior':>14} {'+/-':>10} {'Bias':>12}")
print(f"{'─' * 8} {'─' * 12} {'─' * 14} {'─' * 10} {'─' * 12}")
for name in _PNAMES:
    if name in ['T23', 'T13', 'T12']:
        truth_val = sin2_to_theta(PRIOR_MEAN[name])
    else:
        truth_val = PRIOR_MEAN[name]
    d = sampler.diagnostics_[name]
    bias = d['mean'] - truth_val
    print(f"{name:<8} {truth_val:12.4e} {d['mean']:14.4e} "
          f"{d['std']:10.4e} {bias:+12.4e}")

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hmc_chains.npz')
sampler.save(outpath)
