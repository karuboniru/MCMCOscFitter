"""Unified Bayesian + frequentist fit interface using JAX Barger propagator.

Key API:
    build_jax_log_prob(...) -> nllp_fn(theta_vec)
    fit_map(nllp_fn, theta_init) -> (theta_map, scipy_result)
    fit_numpyro_nuts_exact(nllp_fn, prior_mean, prior_sigma) -> arviz DataTree

The NumPyro NUTS sampler operates in *z-space* where all six parameters
are affine-transformed to approximately O(1) scale, eliminating the
11-order-of-magnitude mass-matrix stiffness that plagues raw theta-space
sampling.  The transformation is exact up to an additive Jacobian constant.

For lower-overhead HMC, use the existing ``jax_barger.mcmc.HMCSampler``
which employs prior-based mass-matrix preconditioning.
"""

import numpy as np
import jax, jax.numpy as jnp

from jax_barger.config import DTYPE, DTYPE_NP, VRAM_SAFE

__all__ = ["build_jax_log_prob", "fit_map", "fit_numpyro_nuts_exact"]


_PNAMES = ["DM2", "Dm2", "T23", "T13", "DCP", "T12"]
_CHANNELS = ["numu", "numubar", "nue", "nuebar"]


# ═══════════════════════════════════════════════════════════════════════════════
# JAX physics pipeline builders
# ═══════════════════════════════════════════════════════════════════════════════

def _build_jax_neg_log_like(E_grid, cos_grid, dist_path, rhoe_path,
                             flux, xsec, data, E_rebin, C_rebin):
    """Build a pure-JAX negative log-likelihood (Poisson chi2/2) function."""
    from jax_barger.pmns import build_pmns, build_dm, compute_mass_order
    from jax_barger.barger import oscillation_prob_layer
    from jax_barger.event_rate import event_rate, poisson_chi2, rebin_2d

    _osc = jax.checkpoint(oscillation_prob_layer) if VRAM_SAFE else oscillation_prob_layer
    _vm = jax.vmap(
        jax.vmap(_osc,
                 in_axes=(0, None, None, None, None, None, None, None)),
        in_axes=(None, 0, 0, None, None, None, None, None))

    def nll_fn(theta_vec):
        DM2, Dm2, th23, th13, dcp, th12 = theta_vec

        Ur, Ui = build_pmns(th12, th13, th23, dcp)
        dm = build_dm(Dm2, DM2)
        order = compute_mass_order(dm)

        Pn = _vm(E_grid, dist_path, rhoe_path, 0, Ur, Ui, dm, order)
        Pa = _vm(E_grid, dist_path, rhoe_path, 1, Ur, -Ui, dm, order)
        P = jnp.transpose(jnp.stack([Pn, Pa], 0), (0, 4, 3, 2, 1))

        ev = event_rate(P, flux, xsec)
        chi2 = sum(
            poisson_chi2(data[ch], rebin_2d(ev[ch], E_rebin, C_rebin))
            for ch in _CHANNELS)
        return 0.5 * chi2

    return nll_fn


def build_jax_log_prob(E_grid, cos_grid, dist_path, rhoe_path,
                       flux, xsec, data,
                       prior_mean, prior_sigma,
                       E_rebin, C_rebin):
    """Build a pure-JAX negative log-posterior function.

    The returned function ``neg_log_prob(theta_vec) -> scalar`` includes
    both the Poisson likelihood and the Gaussian pull priors (in sin^2(theta)
    for mixing angles, cyclic for delta_CP).

    Parameters
    ----------
    E_grid, cos_grid : jnp.array
    dist_path, rhoe_path : jnp.array
        Precomputed path data from ``precompute_path_data``.
    flux, xsec : dict[str, jnp.array]
    data : dict[str, jnp.array]
        Observed event-rate histograms at analysis binning.
    prior_mean, prior_sigma : dict
        PDG central values and widths in sin^2(theta) convention.
    E_rebin, C_rebin : int

    Returns
    -------
    neg_log_prob : callable
        ``neg_log_prob(theta_vec) -> scalar`` where
        ``theta_vec = [DM2, Dm2, th23, th13, dcp, th12]``.
    """
    nll_fn = _build_jax_neg_log_like(
        E_grid, cos_grid, dist_path, rhoe_path,
        flux, xsec, data, E_rebin, C_rebin)

    pm_arr = jnp.array([prior_mean[n] for n in _PNAMES])
    ps_arr = jnp.array([prior_sigma[n] for n in _PNAMES])

    def neg_log_prob(theta_vec):
        DM2, Dm2, th23, th13, dcp, th12 = theta_vec

        nll = nll_fn(theta_vec)

        sin2_23 = jnp.sin(th23) ** 2
        sin2_13 = jnp.sin(th13) ** 2
        sin2_12 = jnp.sin(th12) ** 2

        delta = jnp.array([
            DM2     - pm_arr[0],
            Dm2     - pm_arr[1],
            sin2_23 - pm_arr[2],
            sin2_13 - pm_arr[3],
            dcp     - pm_arr[4],
            sin2_12 - pm_arr[5],
        ])
        d_dcp = jnp.arctan2(jnp.sin(delta[4]), jnp.cos(delta[4]))
        delta = delta.at[4].set(d_dcp)

        chi2_pull = jnp.sum((delta / ps_arr) ** 2)
        log_prior = -0.5 * chi2_pull

        return nll - log_prior

    return neg_log_prob


# ═══════════════════════════════════════════════════════════════════════════════
# MAP estimation
# ═══════════════════════════════════════════════════════════════════════════════

def fit_map(neg_log_prob_fn, theta_init, bounds=None, maxiter=200):
    """Find posterior mode via L-BFGS-B with analytical JAX gradients.

    Parameters
    ----------
    neg_log_prob_fn : callable
        JAX-compatible ``nllp(theta_vec) -> scalar``.
    theta_init : array-like
        Initial guess (6,) in theta-space.
    bounds : list of (lo, hi) or None
    maxiter : int

    Returns
    -------
    theta_map : (6,) np.ndarray
    result : scipy OptimizeResult
    """
    from scipy.optimize import minimize

    fn_jit = jax.jit(neg_log_prob_fn)
    grd_jit = jax.jit(jax.grad(neg_log_prob_fn))

    def _obj(x):
        f = float(fn_jit(jnp.array(x)))
        g = np.array(grd_jit(jnp.array(x)), dtype=DTYPE_NP)
        return f, g

    res = minimize(_obj, np.array(theta_init, dtype=DTYPE_NP),
                   method='L-BFGS-B', jac=True,
                   bounds=bounds,
                   options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
    return res.x, res


# ═══════════════════════════════════════════════════════════════════════════════
# z-space rescaling for well-conditioned NumPyro NUTS
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_centers_scales(prior_mean, prior_sigma):
    """Convert sin^2(theta)-space prior specs to theta-space centers and scales.

    For DM2, Dm2, and delta_CP the prior is already Gaussian in the natural
    parameter, so ``center = prior_mean`` and ``scale = prior_sigma`` directly.

    For the three mixing angles (T23, T13, T12) the prior is Gaussian in
    sin^2(theta).  We linearise the transformation at the prior centre using
    the delta-method Jacobian:

        dθ / d(sin²θ) = 1 / sin(2θ)   →   σ_θ = σ_sin²θ / |sin(2θ₀)|

    This is the same Jacobian used by ``prior_mass_diag_h()`` in the
    HMCSampler setup (run_hierarchy_mcmc.py).

    Parameters
    ----------
    prior_mean : dict
        ``{DM2, Dm2, T23, T13, DCP, T12}`` in sin^2(theta) convention.
    prior_sigma : dict
        Prior widths in same units.

    Returns
    -------
    centers : (6,) np.ndarray  -- theta-space centres [DM2, Dm2, th23, th13, dcp, th12]
    scales  : (6,) np.ndarray  -- theta-space widths  (same order)
    """
    def sin2_to_theta(s2):
        return np.arcsin(np.sqrt(np.clip(s2, 0.0, 1.0)))

    centers = np.empty(6, dtype=DTYPE_NP)
    scales  = np.empty(6, dtype=DTYPE_NP)

    centers[0] = prior_mean['DM2']
    scales[0]  = prior_sigma['DM2']

    centers[1] = prior_mean['Dm2']
    scales[1]  = prior_sigma['Dm2']

    th23_0 = sin2_to_theta(prior_mean['T23'])
    jac23 = np.sin(2.0 * th23_0)
    centers[2] = th23_0
    scales[2]  = prior_sigma['T23'] / max(abs(jac23), 1e-3)

    th13_0 = sin2_to_theta(prior_mean['T13'])
    jac13 = np.sin(2.0 * th13_0)
    centers[3] = th13_0
    scales[3]  = prior_sigma['T13'] / max(abs(jac13), 1e-3)

    centers[4] = prior_mean['DCP']
    scales[4]  = prior_sigma['DCP']

    th12_0 = sin2_to_theta(prior_mean['T12'])
    jac12 = np.sin(2.0 * th12_0)
    centers[5] = th12_0
    scales[5]  = prior_sigma['T12'] / max(abs(jac12), 1e-3)

    return centers, scales


# ═══════════════════════════════════════════════════════════════════════════════
# NumPyro NUTS sampler — exact posterior via z → θ affine lift
# ═══════════════════════════════════════════════════════════════════════════════

def fit_numpyro_nuts_exact(nllp_fn, prior_mean, prior_sigma,
                           n_draws=2000, n_warmup=800,
                           n_chains=4, target_accept=0.9,
                           dense_mass=True, seed=42):
    """Run NUTS sampling on the EXACT posterior via z-space rescaling.

    The sampler operates in dimensionless z-space:

        θⱼ = centerⱼ + zⱼ · scaleⱼ

    where centers and scales are derived from ``prior_mean`` / ``prior_sigma``
    using the same delta-method Jacobian as the custom HMCSampler mass matrix.
    The z → θ map is affine, so the z-space posterior is mathematically
    identical to the theta-space posterior (the constant Jacobian cancels).

    NumPyro sees an approximately isotropic 6-d posterior with all parameter
    variances O(1), allowing NUTS with dense mass matrix to adapt rapidly.

    Parameters
    ----------
    nllp_fn : callable
        ``neg_log_posterior(theta_vec) -> scalar`` from
        :func:`build_jax_log_prob`.  Must include both Poisson likelihood
        AND pull priors in sin^2(theta) convention.
    prior_mean : dict
    prior_sigma : dict
    n_draws : int
        Post-warmup draws per chain.
    n_warmup : int
        Warmup steps per chain.
    n_chains : int
        Number of independent chains.
    target_accept : float
        NUTS target acceptance probability (0.65–0.95).
    dense_mass : bool
        Use full 6×6 mass matrix adaptation (recommended).
    seed : int
        Random seed.

    Returns
    -------
    idata : arviz DataTree
        With ``posterior`` group containing variables
        [DM2, Dm2, th23, th13, dcp, th12] in theta-space.
    """
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    import arviz as az

    centers, scales = _compute_centers_scales(prior_mean, prior_sigma)
    centers_jnp = jnp.array(centers, dtype=DTYPE)
    scales_jnp  = jnp.array(scales,  dtype=DTYPE)
    param_names = ['DM2', 'Dm2', 'th23', 'th13', 'dcp', 'th12']

    def numpyro_model():
        z = numpyro.sample(
            'z',
            dist.ImproperUniform(dist.constraints.real,
                                 batch_shape=(), event_shape=(6,)),
        )
        theta = centers_jnp + z * scales_jnp
        nllp = nllp_fn(theta)
        numpyro.factor('log_posterior', -nllp)

    nuts_kernel = NUTS(
        numpyro_model,
        target_accept_prob=target_accept,
        dense_mass=dense_mass,
        step_size=0.5,
        max_tree_depth=8,
    )

    rng_key = jax.random.PRNGKey(seed)

    mcmc = MCMC(
        nuts_kernel,
        num_warmup=n_warmup,
        num_samples=n_draws,
        num_chains=n_chains,
        chain_method='sequential',
        progress_bar=True,
    )

    init_z = jnp.zeros(6, dtype=DTYPE)
    if n_chains > 1:
        init_z = jnp.tile(init_z[None, :], (n_chains, 1))

    mcmc.run(rng_key, init_params={'z': init_z})

    z_samples = mcmc.get_samples()['z']  # shape: (n_chains * n_draws, 6)
    theta_samples = np.array(z_samples) * scales[None, :] + centers[None, :]

    n_total = n_chains * n_draws
    posterior = {}
    for i, name in enumerate(param_names):
        posterior[name] = theta_samples[:, i].reshape(n_chains, n_draws)

    idata = az.from_dict({"posterior": posterior},
                         sample_dims=["chain", "draw"])
    return idata
