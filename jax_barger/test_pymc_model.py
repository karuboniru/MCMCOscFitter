"""Numerical cross-check: JAX log-prob (new API) vs existing custom HMC.

Tests:
  1. logp evaluation matches build_neg_log_posterior
  2. Gradient matches jax.grad(neg_log_prob_raw)
  3. MAP convergence compares custom vs fit_map()
  4. NumPyro NUTS posterior agrees with HMCSampler (within sampling error)
"""

import sys, os, time, math
import numpy as np

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(REPO_ROOT, 'build', 'pybind'))
sys.path.insert(0, REPO_ROOT)

import mcmcoscfitter as mof
import jax, jax.numpy as jnp
from jax_barger.earth import default_prem, precompute_path_data
from jax_barger.barger import oscillation_probabilities
from jax_barger.event_rate import event_rate, rebin_2d
from jax_barger.mcmc import build_neg_log_posterior, HMCSampler
from jax_barger.pymc_model import build_jax_log_prob, fit_map, fit_numpyro_nuts_exact

PRIOR_MEAN = {'DM2': 2.455e-3, 'Dm2': 7.53e-5, 'T23': 0.558,
              'T13': 2.19e-2, 'T12': 0.307, 'DCP': 1.19 * np.pi}
PRIOR_SIGMA = {'DM2': 0.028e-3, 'Dm2': 0.18e-5, 'T23': 0.018,
               'T13': 0.07e-2, 'T12': 0.013, 'DCP': 0.22 * np.pi}

def sin2_to_theta(sin2):
    return np.arcsin(np.sqrt(np.clip(sin2, 0.0, 1.0)))

N_E, N_COS, E_REBIN, C_REBIN = 10, 12, 1, 1
E_edges = mof.logspace(0.1, 20.0, N_E + 1)
C_edges = mof.linspace(-1.0, 1.0, N_COS + 1)
E_c = np.array(mof.to_center(E_edges))
C_c = np.array(mof.to_center(C_edges))
scale = float(mof.scale_factor_6y)
radii, density, Ye = default_prem()
pi = mof.load_physics_input(E_edges, C_edges, scale)
flux = {k: jnp.array(pi[f'flux_{k}']) for k in ['numu','numubar','nue','nuebar']}
xsec = {k: jnp.array(pi[f'xsec_{k}']) for k in ['numu','numubar','nue','nuebar']}
dist_path, rhoe_path = precompute_path_data(jnp.array(C_c), radii, density, Ye)

_th = jnp.array([PRIOR_MEAN['DM2'], PRIOR_MEAN['Dm2'],
                 sin2_to_theta(PRIOR_MEAN['T23']), sin2_to_theta(PRIOR_MEAN['T13']),
                 PRIOR_MEAN['DCP'], sin2_to_theta(PRIOR_MEAN['T12'])])
P_nom = oscillation_probabilities(jnp.array(E_c), jnp.array(C_c),
    float(_th[5]), float(_th[3]), float(_th[2]),
    float(_th[4]), float(_th[1]), float(_th[0]), radii, density, Ye)
ev_nom = event_rate(np.array(P_nom), flux, xsec)
data = {k: jnp.array(rebin_2d(ev_nom[k], E_REBIN, C_REBIN))
        for k in ['numu','numubar','nue','nuebar']}

print("=" * 60)
print("JAX Log-Prob Cross-check vs Custom HMC")
print("=" * 60)

nllp_custom = build_neg_log_posterior(
    jnp.array(E_c), jnp.array(C_c), dist_path, rhoe_path,
    flux, xsec, data, PRIOR_MEAN, PRIOR_SIGMA, E_REBIN, C_REBIN)
nllp_new = build_jax_log_prob(
    jnp.array(E_c), jnp.array(C_c), dist_path, rhoe_path,
    flux, xsec, data, PRIOR_MEAN, PRIOR_SIGMA, E_REBIN, C_REBIN)

# Test 1: logp
print("\n--- Test 1: neg_log_prob evaluation ---")
rng = np.random.RandomState(42)
max_err = 0.0
for i in range(20):
    pert = rng.randn(6) * 0.1
    th_test = np.array(_th) + pert * np.array([PRIOR_SIGMA['DM2'], PRIOR_SIGMA['Dm2'], 0.05, 0.01, 0.1, 0.05])
    th_test[2] = np.clip(th_test[2], 0.02, np.pi/2 - 0.02)
    th_test[3] = np.clip(th_test[3], 0.002, np.pi/2 - 0.002)
    th_test[5] = np.clip(th_test[5], 0.02, np.pi/2 - 0.02)
    th_test[4] = np.arctan2(np.sin(th_test[4]), np.cos(th_test[4]))
    v1 = float(nllp_custom(jnp.array(th_test)))
    v2 = float(nllp_new(jnp.array(th_test)))
    err = abs(v1 - v2)
    max_err = max(max_err, err)
print(f"  max |delta| = {max_err:.2e} over 20 points")
assert max_err < 1e-5, f"FAILED: max error {max_err:.2e} > 1e-5"
print("  Test 1 PASSED")

# Test 2: gradients
print("\n--- Test 2: gradient evaluation ---")
g_custom = jax.jit(jax.grad(nllp_custom))
g_new = jax.jit(jax.grad(nllp_new))
max_gerr = 0.0
for i in range(5):
    pert = rng.randn(6) * 0.05
    th_test = np.array(_th) + pert * np.array([PRIOR_SIGMA['DM2'], PRIOR_SIGMA['Dm2'], 0.05, 0.01, 0.1, 0.05])
    th_test[2] = np.clip(th_test[2], 0.02, np.pi/2 - 0.02)
    th_test[3] = np.clip(th_test[3], 0.002, np.pi/2 - 0.002)
    th_test[5] = np.clip(th_test[5], 0.02, np.pi/2 - 0.02)
    th_test[4] = np.arctan2(np.sin(th_test[4]), np.cos(th_test[4]))
    gc = np.array(g_custom(jnp.array(th_test)))
    gn = np.array(g_new(jnp.array(th_test)))
    ger = np.max(np.abs(gc - gn))
    max_gerr = max(max_gerr, ger)
print(f"  max |grad delta| = {max_gerr:.2e}")
assert max_gerr < 1e-5, f"FAILED: max grad error {max_gerr:.2e} > 1e-5"
print("  Test 2 PASSED")

# Test 3: MAP convergence
print("\n--- Test 3: MAP convergence ---")
bounds = [(-1.0, 1.0), (1e-7, 1e-3), (0.1, np.pi/2-0.01), (0.01, np.pi/2-0.01), (-np.pi, np.pi), (0.1, np.pi/2-0.01)]
map_new, res_new = fit_map(nllp_new, _th, bounds=bounds)
map_old, res_old = fit_map(nllp_custom, _th, bounds=bounds)
nll_new_at_map = float(nllp_new(jnp.array(map_new)))
nll_old_at_map = float(nllp_custom(jnp.array(map_old)))
print(f"  New fit: nllp={nll_new_at_map:.4f}  evals={res_new.nfev}")
print(f"  Old fit: nllp={nll_old_at_map:.4f}  evals={res_old.nfev}")
assert abs(nll_new_at_map - nll_old_at_map) < 1.0
print("  Test 3 PASSED")

# Test 4: NumPyro NUTS vs HMCSampler posterior agreement
print("\n--- Test 4: NumPyro NUTS vs HMCSampler ---")
print("  Running HMCSampler (warmup=50, draws=50, chains=2)...")
imd = np.ones(6)
sa = HMCSampler(nllp_new, eps_0=0.05, n_leapfrog=20, target_accept=0.651, initial_mass_diag=imd)
sa.warmup(n_steps=50, z_init=np.array(map_new), adapt_step=True, adapt_mass=False)
ch = sa.sample(n_samples=50, n_chains=2)
hmc_post = {name: np.array(ch[:, :, i]) for i, name in enumerate(['DM2','Dm2','th23','th13','dcp','th12'])}

print("  Running NumPyro NUTS (warmup=50, draws=50, chains=2)...")
import arviz as az
nr_idata = fit_numpyro_nuts_exact(nllp_new, PRIOR_MEAN, PRIOR_SIGMA,
                                    n_warmup=50, n_draws=50, n_chains=2,
                                    target_accept=0.9, seed=42)

print("  Comparing posteriors:")
pnames = ['DM2','Dm2','th23','th13','dcp','th12']
all_pass = True
for i, name in enumerate(pnames):
    hmc_m = float(hmc_post[name].mean()); hmc_s = float(hmc_post[name].std())
    nr_m  = float(nr_idata.posterior[name].mean()); nr_s  = float(nr_idata.posterior[name].std())
    diff = abs(hmc_m - nr_m) / max(hmc_s, nr_s, 1e-30)
    ok = diff < 3.0  # within 3 sigma
    flag = "OK" if ok else "FAIL"
    if not ok: all_pass = False
    print(f"  {name:<6} HMC={hmc_m:.4e}+/-{hmc_s:.1e}  NUTS={nr_m:.4e}+/-{nr_s:.1e}  Δ/σ={diff:.2f} [{flag}]")
assert all_pass, "Test 4 FAILED: posteriors disagree >3 sigma"
print("  Test 4 PASSED")

print(f"\n{'=' * 60}")
print("All tests passed!")
print(f"{'=' * 60}")
