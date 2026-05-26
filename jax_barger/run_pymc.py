"""Unified Bayesian + frequentist fit using JAX + ArviZ.

Workflow:
  1. Build JAX negative log-posterior via build_jax_log_prob
  2. Frequentist: MAP estimate via L-BFGS-B (JAX gradients)
  3. Bayesian: MCMC sampling (choice of sampler)
  4. ArviZ diagnostics + posterior summary

Samplers:
  hmc      Custom HMCSampler (jax_barger.mcmc) — well-tuned, fast, default
  numpyro  NumPyro NUTS with z-space rescaling — exact posterior, no manual tuning

Usage:
    PYTHONPATH=../build/pybind:.. .venv/bin/python run_pymc.py --fine --sampler hmc --samples 2000 --warmup 800 --chains 4
    PYTHONPATH=../build/pybind:.. .venv/bin/python run_pymc.py --fast --sampler numpyro --samples 500 --warmup 200 --chains 2
    PYTHONPATH=../build/pybind:.. .venv/bin/python run_pymc.py --fast --fp32 --skip-hmc
"""

import sys, os, time, argparse, math
import numpy as np

if '--fp32' in sys.argv:
    os.environ['JAX_BARGER_FLOAT32'] = '1'
    sys.argv.remove('--fp32')
if '--no-vram-workaround' in sys.argv:
    os.environ['JAX_BARGER_NO_VRAM_WORKAROUND'] = '1'
    sys.argv.remove('--no-vram-workaround')

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(REPO_ROOT, 'build', 'pybind'))
sys.path.insert(0, REPO_ROOT)

import mcmcoscfitter as mof
import jax, jax.numpy as jnp
import arviz as az

from jax_barger.earth import default_prem, precompute_path_data
from jax_barger.barger import oscillation_probabilities
from jax_barger.event_rate import event_rate, rebin_2d
from jax_barger.pymc_model import build_jax_log_prob, fit_map, fit_numpyro_nuts_exact
from jax_barger.mcmc import HMCSampler, _PNAMES


parser = argparse.ArgumentParser(description='Unified fit with JAX + ArviZ')
_group = parser.add_mutually_exclusive_group(required=True)
_group.add_argument('--fast', action='store_true', help='Fast mode: 10x12 bin centres')
_group.add_argument('--fine', action='store_true', help='Fine mode: 200x120 + rebin to 10x12')
parser.add_argument('--sampler', choices=['hmc', 'numpyro', 'both'], default='hmc',
                    help='MCMC sampler (default: hmc)')
parser.add_argument('--samples', type=int, default=2000, help='Production samples per chain')
parser.add_argument('--warmup', type=int, default=800, help='Warmup steps')
parser.add_argument('--chains', type=int, default=4, help='Number of chains')
parser.add_argument('--leapfrog', type=int, default=30, help='HMC leapfrog steps per proposal')
parser.add_argument('--eps0', type=float, default=0.05, help='HMC initial step size')
parser.add_argument('--target-accept', type=float, default=0.651, help='Target accept rate')
parser.add_argument('--fp32', action='store_true', help='Use float32 precision')
parser.add_argument('--no-vram-workaround', action='store_true', help='Disable VRAM workarounds')
parser.add_argument('--skip-hmc', action='store_true', help='Skip MCMC, do MAP only')
args = parser.parse_args()


if args.fast:
    N_E, N_COS, E_REBIN, C_REBIN = 10, 12, 1, 1
    mode_label = "FAST (10x12 bin centres)"
else:
    N_E, N_COS, E_REBIN, C_REBIN = 200, 120, 20, 10
    mode_label = f"FINE ({N_E}x{N_COS} -> rebin {E_REBIN}x{C_REBIN})"

E_edges = mof.logspace(0.1, 20.0, N_E + 1)
C_edges = mof.linspace(-1.0, 1.0, N_COS + 1)
E_c = np.array(mof.to_center(E_edges))
C_c = np.array(mof.to_center(C_edges))
scale = float(mof.scale_factor_6y)
radii, density, Ye = default_prem()

sampler_label = {'hmc': 'HMCSampler (custom)', 'numpyro': 'NumPyro NUTS (z-space)',
                 'both': 'HMCSampler + NumPyro NUTS'}[args.sampler]

print(f"{'='*72}")
print(f"JAX + ArviZ Unified Fit -- Neutrino Oscillations")
print(f"{'='*72}")
print(f"Grid:        {mode_label}")
print(f"Points:      {N_E * N_COS}")
print(f"Sampler:     {sampler_label}")
if not args.skip_hmc:
    print(f"Chains:      {args.chains} chains x {args.samples} samples  warmup={args.warmup}")

pi = mof.load_physics_input(E_edges, C_edges, scale)
flux = {k: jnp.array(pi[f'flux_{k}']) for k in ['numu','numubar','nue','nuebar']}
xsec = {k: jnp.array(pi[f'xsec_{k}']) for k in ['numu','numubar','nue','nuebar']}
dist_path, rhoe_path = precompute_path_data(jnp.array(C_c), radii, density, Ye)

PRIOR_MEAN = {'DM2': 2.455e-3, 'Dm2': 7.53e-5, 'T23': 0.558, 'T13': 2.19e-2, 'T12': 0.307, 'DCP': 1.19 * np.pi}
PRIOR_SIGMA = {'DM2': 0.028e-3, 'Dm2': 0.18e-5, 'T23': 0.018, 'T13': 0.07e-2, 'T12': 0.013, 'DCP': 0.22 * np.pi}

def sin2_to_theta(sin2):
    return np.arcsin(np.sqrt(np.clip(sin2, 0.0, 1.0)))

_th_truth = np.array([PRIOR_MEAN['DM2'], PRIOR_MEAN['Dm2'],
                      sin2_to_theta(PRIOR_MEAN['T23']), sin2_to_theta(PRIOR_MEAN['T13']),
                      PRIOR_MEAN['DCP'], sin2_to_theta(PRIOR_MEAN['T12'])])
P_nom = oscillation_probabilities(jnp.array(E_c), jnp.array(C_c),
    float(_th_truth[5]), float(_th_truth[3]), float(_th_truth[2]),
    float(_th_truth[4]), float(_th_truth[1]), float(_th_truth[0]), radii, density, Ye)
ev_nom = event_rate(np.array(P_nom), flux, xsec)
data = {k: jnp.array(rebin_2d(ev_nom[k], E_REBIN, C_REBIN))
        for k in ['numu','numubar','nue','nuebar']}
for k in data:
    print(f"  {k:>8s} sum = {float(data[k].sum()):.1f}")

# Build JAX neg-log-posterior
print("\n--- Building JAX neg-log-posterior ---")
nllp = build_jax_log_prob(jnp.array(E_c), jnp.array(C_c), dist_path, rhoe_path,
                          flux, xsec, data, PRIOR_MEAN, PRIOR_SIGMA, E_REBIN, C_REBIN)

nllp_jit = jax.jit(nllp)
grd_jit = jax.jit(jax.grad(nllp))
_ = nllp_jit(jnp.array(_th_truth))
_ = grd_jit(jnp.array(_th_truth))
print(f"  NLLP at truth: {float(nllp_jit(jnp.array(_th_truth))):.4f}")
print(f"  |grad| at truth: {float(np.linalg.norm(np.array(grd_jit(jnp.array(_th_truth))))):.2e}")


# Frequentist: MAP
print(f"\n{'='*72}")
print("FREQUENTIST: Maximum a Posteriori (MAP) estimate")
print(f"{'='*72}")
t0 = time.time()
bounds = [(-1.0, 1.0), (1e-7, 1e-3), (0.1, np.pi/2-0.01), (0.01, np.pi/2-0.01),
          (-np.pi, np.pi), (0.1, np.pi/2-0.01)]
map_est, map_res = fit_map(nllp, _th_truth, bounds=bounds, maxiter=500)
dt_map = time.time() - t0
print(f"  MAP NLLP:    {map_res.fun:.4f}")
print(f"  Evaluations: {map_res.nfev}  time={dt_map:.1f}s")
pnames = ['DM2', 'Dm2', 'th23', 'th13', 'dcp', 'th12']
for i, name in enumerate(pnames):
    print(f"  {name:<6} = {map_est[i]:.4e}  (truth = {_th_truth[i]:.4e})")

if args.skip_hmc:
    print("\nSkipping MCMC. Done.")
    sys.exit(0)


def run_hmc(nllp_fn, map_est, prior_mean, prior_sigma):
    print(f"\n{'='*72}")
    print("BAYESIAN: Adaptive HMC sampling (HMCSampler)")
    print(f"{'='*72}")

    np.random.seed(42)
    pert = np.random.randn(6) * 0.3
    z_init = np.array(map_est) + pert * np.array([
        prior_sigma['DM2'], prior_sigma['Dm2'], 0.05, 0.01, 0.1, 0.05])
    z_init[2] = np.clip(z_init[2], 0.02, np.pi/2 - 0.02)
    z_init[3] = np.clip(z_init[3], 0.002, np.pi/2 - 0.002)
    z_init[5] = np.clip(z_init[5], 0.02, np.pi/2 - 0.02)
    z_init[4] = np.arctan2(np.sin(z_init[4]), np.cos(z_init[4]))

    th23_m = float(map_est[2]); th13_m = float(map_est[3]); th12_m = float(map_est[5])
    d23 = 1.0/np.sin(2.0*th23_m); d13 = 1.0/np.sin(2.0*th13_m); d12 = 1.0/np.sin(2.0*th12_m)
    pv = np.array([prior_sigma['DM2']**2, prior_sigma['Dm2']**2,
                   (prior_sigma['T23']*d23)**2, (prior_sigma['T13']*d13)**2,
                   prior_sigma['DCP']**2, (prior_sigma['T12']*d12)**2])
    imd = 1.0 / np.maximum(pv, 1e-30)

    sa = HMCSampler(nllp_fn, eps_0=args.eps0, n_leapfrog=args.leapfrog,
                    target_accept=args.target_accept, initial_mass_diag=imd)

    print(f"  Warmup ({args.warmup} steps)...")
    t0 = time.time()
    sa.warmup(n_steps=args.warmup, z_init=z_init, adapt_step=True, adapt_mass=False)
    dt_w = time.time() - t0
    print(f"  Warmup time: {dt_w:.1f}s")

    print(f"  Production ({args.samples} samples x {args.chains} chains)...")
    t0 = time.time()
    chains = sa.sample(n_samples=args.samples, n_chains=args.chains)
    dt_s = time.time() - t0
    print(f"  Sample time: {dt_s:.1f}s  Total: {dt_w + dt_s:.1f}s")
    sa.diagnostics()

    posterior = {name: np.array(chains[:, :, i]) for i, name in enumerate(pnames)}
    return az.from_dict({"posterior": posterior}, sample_dims=["chain", "draw"])


def run_numpyro(nllp_fn, prior_mean, prior_sigma):
    print(f"\n{'='*72}")
    print("BAYESIAN: NumPyro NUTS sampling (z-space rescaling)")
    print(f"{'='*72}")
    print(f"  Warmup ({args.warmup} steps)...")
    t0 = time.time()
    idata = fit_numpyro_nuts_exact(nllp_fn, prior_mean, prior_sigma,
                                    n_draws=args.samples,
                                    n_warmup=args.warmup,
                                    n_chains=args.chains,
                                    target_accept=args.target_accept)
    dt = time.time() - t0
    print(f"  Total time: {dt:.1f}s")
    return idata


if args.sampler == 'hmc':
    idata = run_hmc(nllp, map_est, PRIOR_MEAN, PRIOR_SIGMA)
elif args.sampler == 'numpyro':
    idata = run_numpyro(nllp, PRIOR_MEAN, PRIOR_SIGMA)
elif args.sampler == 'both':
    idata_hmc = run_hmc(nllp, map_est, PRIOR_MEAN, PRIOR_SIGMA)
    idata_nr = run_numpyro(nllp, PRIOR_MEAN, PRIOR_SIGMA)
    print(f"\n{'='*72}")
    print("SAMPLER COMPARISON")
    print(f"{'='*72}")
    for i, name in enumerate(pnames):
        hmc_m = float(idata_hmc.posterior[name].mean()); hmc_s = float(idata_hmc.posterior[name].std())
        nr_m  = float(idata_nr.posterior[name].mean());  nr_s  = float(idata_nr.posterior[name].std())
        diff = abs(hmc_m - nr_m) / max(hmc_s, nr_s, 1e-30)
        print(f"  {name:<6} HMC={hmc_m:.4e}±{hmc_s:.1e}  NUTS={nr_m:.4e}±{nr_s:.1e}  Δ/σ={diff:.2f}")
    idata = idata_hmc

# Summary
print(f"\n{'='*72}")
print("ARVIZ POSTERIOR SUMMARY")
print(f"{'='*72}")
print(az.summary(idata).to_string())

print(f"\n{'='*72}")
print("BIAS vs PRIOR MEAN (Asimov truth)")
print(f"{'='*72}")
print(f"  {'Param':<8} {'Truth':>12} {'Posterior':>14} {'+/-':>10} {'Bias':>12}")
print(f"  {'-'*8} {'-'*12} {'-'*14} {'-'*10} {'-'*12}")
for i, name in enumerate(pnames):
    truth = _th_truth[i]
    pm_val = float(idata.posterior[name].mean())
    ps_val = float(idata.posterior[name].std())
    bias = pm_val - truth
    print(f"  {name:<8} {truth:12.4e} {pm_val:14.4e} {ps_val:10.4e} {bias:+12.4e}")

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pymc_fit.nc')
idata.to_netcdf(outpath)
print(f"\nSaved to {outpath}")
print(f"\n{'='*72}")
print("Done. Unified Bayesian + frequentist fit complete.")
print(f"{'='*72}")
