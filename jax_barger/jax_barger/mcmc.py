"""Gradient-accelerated MCMC via Hamiltonian Monte Carlo (HMC) with adaptation.

Implements:
  - Hamiltonian Monte Carlo (HMC) with leapfrog integrator
  - Adaptive step size via dual averaging (NUTS paper, Algorithm 6)
  - Adaptive mass matrix from sample covariance (warmup phase)
  - Multi-chain production sampling with jax.lax.scan for JIT speed

The sampler operates in θ-space [DM2, Dm2, θ₂₃, θ₁₃, δCP, θ₁₂] to avoid
the gradient singularity at sin²θ = 1.

Typical usage:
    from jax_barger.mcmc import build_neg_log_posterior, HMCSampler

    neg_log_prob = build_neg_log_posterior(..., prior_mean, prior_sigma, ...)
    sampler = HMCSampler(neg_log_prob)
    sampler.warmup(800)
    chains = sampler.sample(2000, n_chains=4)
    sampler.diagnostics()
"""

import math
import jax
import jax.numpy as jnp
import numpy as np
from jax_barger.config import DTYPE, DTYPE_NP


# ═══════════════════════════════════════════════════════════════════════════════
# Neg-log-posterior builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_neg_log_posterior(E_grid, cos_grid, dist_path, rhoe_path,
                            flux, xsec, data,
                            prior_mean, prior_sigma,
                            E_rebin, C_rebin):
    """Build a JIT-compilable negative log-posterior in θ-space.

    The 6 continuous parameters in theta_vec are:
        [DM2, Dm2, θ₂₃, θ₁₃, δCP, θ₁₂]
    where θⱼ are mixing *angles* (radians), not sin²θ.

    The prior is Gaussian in sin²θ for the three mixing angles, and directly
    Gaussian for DM2, Dm2, δCP (with cyclic wrap for δCP).  The prior_mean
    and prior_sigma dicts use the same convention as compare_fit.py:
        {'DM2': ..., 'Dm2': ..., 'T23': ..., 'T13': ..., 'DCP': ..., 'T12': ...}
    where Tij are sin²θ values.

    Args:
        E_grid: (nE_fine,) jnp.array, fine-grid energies [GeV]
        cos_grid: (nCos_fine,) jnp.array, fine-grid cos(zenith)
        dist_path: (nCos_fine, MAX_LAYERS) jnp.array, precomputed layer distances
        rhoe_path: (nCos_fine, MAX_LAYERS) jnp.array, precomputed layer densities
        flux: dict of (nE_fine, nCos_fine) arrays by channel
        xsec: dict of (nE_fine,) arrays by channel
        data: dict of (nE_analysis, nCos_analysis) arrays by channel
        prior_mean: dict with C++ convention keys ('T23' means sin²θ₂₃)
        prior_sigma: dict in same units as prior_mean
        E_rebin: int, E rebinning factor
        C_rebin: int, cosθ rebinning factor

    Returns:
        neg_log_prob(theta_vec) → scalar float64
            Negative log-posterior: -log P(θ | data) suitable for HMC potential.
    """
    from jax_barger.pmns import build_pmns, build_dm, compute_mass_order
    from jax_barger.barger import oscillation_prob_layer
    from jax_barger.event_rate import event_rate, poisson_chi2, rebin_2d

    # Pre-build vmap-over-(E, cosθ) oscillator
    _vm = jax.vmap(
        jax.vmap(oscillation_prob_layer,
                 in_axes=(0, None, None, None, None, None, None, None)),
        in_axes=(None, 0, 0, None, None, None, None, None))

    # Prior in physics units (matching C++ convention)
    _pm = jnp.array([prior_mean[n] for n in _PNAMES])
    _ps = jnp.array([prior_sigma[n] for n in _PNAMES])
    # Prior sin²θ values for pull terms
    _pm_sin2_23 = _pm[2]
    _pm_sin2_13 = _pm[3]
    _pm_sin2_12 = _pm[5]

    def neg_log_prob(theta_vec):
        DM2, Dm2, th23, th13, dcp, th12 = theta_vec

        # ── Forward model ──
        Ur, Ui = build_pmns(th12, th13, th23, dcp)
        dm = build_dm(Dm2, DM2)
        order = compute_mass_order(dm)

        Pn = _vm(E_grid, dist_path, rhoe_path, 0, Ur, Ui, dm, order)
        Pa = _vm(E_grid, dist_path, rhoe_path, 1, Ur, -Ui, dm, order)
        P = jnp.transpose(jnp.stack([Pn, Pa], 0), (0, 4, 3, 2, 1))

        ev = event_rate(P, flux, xsec)
        chi2_poisson = sum(
            poisson_chi2(data[ch], rebin_2d(ev[ch], E_rebin, C_rebin))
            for ch in _CHANNELS)
        log_like = -0.5 * chi2_poisson

        # ── Prior (pull terms) ──
        sin2_23 = jnp.sin(th23) ** 2
        sin2_13 = jnp.sin(th13) ** 2
        sin2_12 = jnp.sin(th12) ** 2

        delta = jnp.array([
            DM2      - _pm[0],
            Dm2      - _pm[1],
            sin2_23  - _pm_sin2_23,
            sin2_13  - _pm_sin2_13,
            dcp      - _pm[4],
            sin2_12  - _pm_sin2_12,
        ])
        # Cyclic wrap for δCP
        d_dcp = jnp.arctan2(jnp.sin(delta[4]), jnp.cos(delta[4]))
        delta = delta.at[4].set(d_dcp)

        chi2_pull = jnp.sum((delta / _ps) ** 2)
        log_prior = -0.5 * chi2_pull

        return -(log_like + log_prior)

    return neg_log_prob


_PNAMES = ['DM2', 'Dm2', 'T23', 'T13', 'DCP', 'T12']
_CHANNELS = ['numu', 'numubar', 'nue', 'nuebar']


# ═══════════════════════════════════════════════════════════════════════════════
# HMC core: kinetic energy, leapfrog, proposal
# ═══════════════════════════════════════════════════════════════════════════════

def _kinetic_energy(r, M_inv):
    """K(r) = 0.5 * rᵀ M⁻¹ r."""
    return 0.5 * jnp.dot(r, M_inv @ r)


def _leapfrog_step(z, r, eps, M_inv, grad_fn):
    """Single Verlet (leapfrog) step for HMC.

    Args:
        z: position (parameter vector)
        r: momentum
        eps: step size
        M_inv: inverse mass matrix (d, d)
        grad_fn: gradient of neg_log_prob

    Returns:
        (z_new, r_new)
    """
    r_half = r - 0.5 * eps * grad_fn(z)
    z_new = z + eps * (M_inv @ r_half)
    r_new = r_half - 0.5 * eps * grad_fn(z_new)
    return z_new, r_new


def _hmc_proposal(z, eps, n_steps, M, M_inv, neg_log_prob_fn, grad_fn, rng_key):
    """Full HMC proposal: sample momentum, run leapfrog, Metropolis-Hastings.

    Args:
        z: current position (d,)
        eps: step size
        n_steps: number of leapfrog steps
        M: mass matrix (d, d), covariance-shaped → r ~ N(0, M)
        M_inv: inverse mass matrix (d, d), used in leapfrog
        neg_log_prob_fn: potential energy U(z)
        grad_fn: gradient of U
        rng_key: JAX PRNG key

    Returns:
        z_out: proposed or current position
        accepted: bool (True if proposal accepted)
        alpha: Metropolis acceptance probability
    """
    key_mom, key_accept = jax.random.split(rng_key)
    dim = z.shape[0]

    # Sample momentum r ~ N(0, M)
    L_m = jnp.linalg.cholesky(M)
    r_init = L_m @ jax.random.normal(key_mom, (dim,))

    # Leapfrog integration
    def _step(i, state):
        zz, rr = state
        return _leapfrog_step(zz, rr, eps, M_inv, grad_fn)

    z_final, r_final = jax.lax.fori_loop(0, n_steps, _step, (z, r_init))

    # Hamiltonian at start and end
    U_init = neg_log_prob_fn(z)
    K_init = _kinetic_energy(r_init, M_inv)
    U_final = neg_log_prob_fn(z_final)
    K_final = _kinetic_energy(r_final, M_inv)

    # log acceptance probability (higher → more likely to accept)
    log_alpha = (U_init + K_init) - (U_final + K_final)

    u = jax.random.uniform(key_accept)
    accept = jnp.log(u) < log_alpha
    safe = jnp.logical_and(accept, jnp.isfinite(log_alpha))

    z_out = jnp.where(safe, z_final, z)
    alpha = jnp.clip(jnp.exp(jnp.minimum(log_alpha, 0.0)), 0.0, 1.0)
    return z_out, safe, alpha


# ═══════════════════════════════════════════════════════════════════════════════
# HMCSampler
# ═══════════════════════════════════════════════════════════════════════════════

class HMCSampler:
    """Adaptive Hamiltonian Monte Carlo sampler.

    Stores the raw (non-JIT) neg-log-posterior and gradient functions.  Warmup
    JIT-compiles them locally for speed; production sampling uses ``jax.lax.scan``
    which JIT-compiles the full chain, tracing through the raw functions.

    Attributes:
        neg_log_prob_raw:  raw neg-log-posterior callable
        grad_fn_raw:       raw gradient of neg_log_prob
        z_current:         current parameter vector (after warmup or sampling)
        eps:               current (adapted) step size
        M:                 mass matrix (d, d)
        M_inv:             inverse mass matrix (d, d)
        chains:            (n_chains, n_samples, d) array after sampling
        diagnostics_:      dict populated after sampling
    """

    def __init__(self, neg_log_prob_fn, mass_matrix=None,
                 eps_0=0.05, n_leapfrog=30,
                 target_accept=0.651,
                 initial_mass_diag=None):
        """Initialize the HMC sampler.

        Args:
            neg_log_prob_fn: callable θ → scalar, negative log posterior
            mass_matrix:     (d, d) array or None
            eps_0:           initial step size
            n_leapfrog:      number of leapfrog steps per proposal
            target_accept:   target Metropolis acceptance rate
            initial_mass_diag: (6,) array, diagonal of initial mass matrix.
                Set to 1/σ² where σ is the prior uncertainty.  Greatly
                improves warmup stability for stiff posteriors.
        """
        self.neg_log_prob_raw = neg_log_prob_fn
        self.grad_fn_raw = jax.grad(neg_log_prob_fn)
        self.eps = eps_0
        self.n_leapfrog = n_leapfrog
        self.target_accept = target_accept
        self.z_current = None
        self.chains = []
        self.diagnostics_ = {}

        dim = 6
        if mass_matrix is not None:
            self.M = np.array(mass_matrix, dtype=DTYPE_NP)
            self.M_inv = np.linalg.inv(self.M + 1e-10 * np.eye(dim))
        elif initial_mass_diag is not None:
            diag = np.array(initial_mass_diag, dtype=DTYPE_NP)
            self.M = np.diag(diag)
            self.M_inv = np.diag(1.0 / np.maximum(diag, 1e-30))
        else:
            self.M = np.eye(dim)
            self.M_inv = np.eye(dim)

    # ── Warmup ──────────────────────────────────────────────────────────────

    def warmup(self, n_steps=800, z_init=None, adapt_step=True,
               adapt_mass=True, adapt_mass_start=None,
               adapt_mass_interval=1, adapt_mass_window=None):
        """Warmup phase: tune step size and mass matrix.

        Args:
            n_steps:             total warmup iterations
            z_init:              initial position (6,) or None (zeros(6))
            adapt_step:          use dual averaging for step size
            adapt_mass:          estimate mass matrix from sample covariance
            adapt_mass_start:    iteration at which mass-matrix collection begins
                                 (default: int(n_steps * 0.3))
            adapt_mass_interval: number of iterations between sample collections
            adapt_mass_window:   number of samples for covariance estimate
                                 (default: max(12, int(n_steps * 0.15)))

        Returns:
            self (for chaining)
        """
        if z_init is None:
            z_init = jnp.zeros(6, dtype=DTYPE)

        z = jnp.array(z_init, dtype=DTYPE)
        dim = z.shape[0]

        # --- Set adaptation defaults ---
        if adapt_mass_start is None:
            adapt_mass_start = max(25, int(n_steps * 0.3))
        if adapt_mass_window is None:
            adapt_mass_window = max(12, int(n_steps * 0.15))

        # --- Locally JIT-compile for fast eval during warmup ---
        _nlp_jit = jax.jit(self.neg_log_prob_raw)
        _grd_jit = jax.jit(self.grad_fn_raw)

        # --- Dual averaging state ---
        if adapt_step:
            mu = math.log(10.0 * self.eps)
            log_eps_bar = 0.0
            H_bar = 0.0
            gamma_da = 0.05
            t0 = 10.0
            kappa = 0.75
            step_counter = 0

        # --- Mass-matrix adaptation state ---
        sample_buf = [] if adapt_mass else None

        accepted = 0
        mass_updated = False  # track whether mass matrix was updated

        for m in range(n_steps):
            # HMC proposal
            rng_key = jax.random.PRNGKey(m * 31337 + 7)
            z_new, acc, alpha = _hmc_proposal(
                z, self.eps, self.n_leapfrog,
                jnp.array(self.M, dtype=DTYPE),
                jnp.array(self.M_inv, dtype=DTYPE),
                _nlp_jit, _grd_jit, rng_key)

            z = z_new
            accepted += int(acc)

            # --- Dual averaging update ---
            if adapt_step:
                step_counter += 1
                iter_m = float(step_counter)
                eta = 1.0 / (iter_m + t0)
                alpha_safe = float(alpha) if np.isfinite(alpha) else 0.0
                H_bar = (1.0 - eta) * H_bar + eta * (self.target_accept - alpha_safe)
                log_eps = mu - jnp.sqrt(iter_m) / gamma_da * H_bar
                log_eps_bar = (iter_m ** (-kappa) * log_eps
                               + (1.0 - iter_m ** (-kappa)) * log_eps_bar)
                new_eps = float(jnp.exp(log_eps_bar))
                if np.isfinite(new_eps) and new_eps > 0:
                    self.eps = new_eps
                else:
                    self.eps *= 0.5  # fallback: halve step size if NaN

            # --- Mass-matrix update (mid-warmup, then re-adapt ε) ---
            if adapt_mass and m >= max(adapt_mass_start, int(n_steps * 0.6)):
                if (m - max(adapt_mass_start, int(n_steps * 0.6))) % max(1, adapt_mass_interval) == 0:
                    sample_buf.append(np.array(z))
                if len(sample_buf) >= adapt_mass_window and not mass_updated:
                    self._update_mass_matrix(np.array(sample_buf))
                    sample_buf = []
                    mass_updated = True
                    # Re-initialize dual averaging for the adapted mass matrix
                    if adapt_step:
                        mu = math.log(10.0 * self.eps)
                        H_bar = 0.0
                        step_counter = 0

            # Progress
            if (m + 1) % 200 == 0 or m == 0 or m == n_steps - 1:
                acc_rate = accepted / (m + 1)
                print(f"  warmup {m + 1:4d}/{n_steps}  "
                      f"eps={self.eps:.4f}  acc={acc_rate:.3f}  "
                      f"U={float(_nlp_jit(z)):.2f}")

        # Final mass-matrix update (only if not already updated mid-warmup)
        if adapt_mass and not mass_updated and sample_buf and len(sample_buf) >= dim * 4:
            self._update_mass_matrix(np.array(sample_buf))

        self.z_current = np.array(z)
        print(f"  warmup complete: eps={self.eps:.4f}  final_accept={accepted / n_steps:.3f}")
        return self

    def _update_mass_matrix(self, samples):
        """Update mass matrix from sample covariance.

        Mass matrix M = Σ^{-1} where Σ is the sample covariance.
        Regularization scales with variance to avoid swamping tiny entries.
        """
        dim = samples.shape[1]
        if samples.shape[0] < dim * 2:
            return
        cov = np.cov(samples.T)
        reg = np.diag(np.maximum(1e-6 * np.abs(np.diag(cov)), 1e-12))
        cov += reg
        self.M_inv = cov                    # M^{-1} = posterior covariance
        self.M = np.linalg.inv(cov)         # M = inverse covariance

    # ── Production sampling (scan-based, JIT-compiled) ────────────────────

    def sample(self, n_samples=2000, n_chains=4):
        """Run production HMC sampling on multiple independent chains.

        The scan body captures ``self.neg_log_prob_raw`` and
        ``self.grad_fn_raw`` in a closure so that JAX can trace
        through the full forward model during JIT compilation.

        Args:
            n_samples: samples per chain
            n_chains:  number of independent chains

        Returns:
            chains: (n_chains, n_samples, d) np.ndarray
        """
        if self.z_current is None:
            raise RuntimeError("Call warmup() before sample().")

        # Capture raw functions + fixed quantities in closure scope
        _nlp = self.neg_log_prob_raw
        _grd = self.grad_fn_raw
        _M = jnp.array(self.M, dtype=DTYPE)
        _M_inv = jnp.array(self.M_inv, dtype=DTYPE)
        _eps = float(self.eps)
        _nsteps = self.n_leapfrog

        z0 = jnp.array(self.z_current, dtype=DTYPE)
        dim = z0.shape[0]

        chains_list = []

        for c in range(n_chains):
            # Perturb initial position using the posterior covariance shape
            seed = 42 + c * 10007
            key_init = jax.random.PRNGKey(seed)
            L_inv = jnp.linalg.cholesky(_M_inv)  # M_inv ≈ posterior covariance Σ
            w_perturb = L_inv @ jax.random.normal(key_init, (dim,))
            z_chain = z0 + 0.5 * w_perturb

            # Build scan kernel (closure captures _nlp, _grd, _M, _M_inv, _eps, _nsteps)
            def _kernel(z_i, rng_key):
                z_new, acc, _alpha = _hmc_proposal(
                    z_i, _eps, _nsteps, _M, _M_inv, _nlp, _grd, rng_key)
                return z_new, (z_new, acc)

            # Run chain via jax.lax.scan (JIT-compiles the full path)
            key_master = jax.random.PRNGKey(seed + 999983)
            keys = jax.random.split(key_master, n_samples)

            final_z, (chain, accepted) = jax.lax.scan(_kernel, z_chain, keys)

            chain_np = np.array(chain)
            acc_np = np.array(accepted, dtype=DTYPE_NP)
            chains_list.append(chain_np)

            # Quick summary using local JIT'd eval
            _nlp_jit = jax.jit(self.neg_log_prob_raw)
            _nlp_vmap = jax.vmap(_nlp_jit)
            acc_rate = float(acc_np.mean())
            mean_u = float(_nlp_vmap(jnp.array(chain_np)).mean())
            print(f"  chain {c + 1}/{n_chains}: {n_samples} samples, "
                  f"accept={acc_rate:.3f}  "
                  f"meanU={mean_u:.1f}")

        self.chains = np.array(chains_list)  # (n_chains, n_samples, d)
        self.diagnostics_ = self._compute_diagnostics(self.chains)
        return self.chains

    def _compute_diagnostics(self, chains):
        """Compute MCMC diagnostics."""
        n_chains, n_samples, d = chains.shape
        diag = {}

        # Acceptance rate (approximate from chain info — we don't store per-step accepts here)
        # Instead, mean and ESS per parameter
        for i, name in enumerate(_PNAMES):
            param_chains = chains[:, :, i]  # (n_chains, n_samples)
            mean_val = float(np.mean(param_chains))
            std_val = float(np.std(param_chains))
            q16, q84 = np.percentile(param_chains, [16, 84], axis=(0, 1))
            q2_5, q97_5 = np.percentile(param_chains, [2.5, 97.5], axis=(0, 1))
            ess = self._ess(param_chains)
            rhat = self._rhat(param_chains)

            diag[name] = {
                'mean': mean_val,
                'std': std_val,
                'ci68': (float(q16), float(q84)),
                'ci95': (float(q2_5), float(q97_5)),
                'ess': float(ess),
                'rhat': float(rhat),
            }

        diag['n_chains'] = n_chains
        diag['n_samples'] = n_samples
        return diag

    @staticmethod
    def _ess(chains):
        """Effective sample size (simplified: mean of per-chain ESS)."""
        n_chains, n_samples = chains.shape
        if n_samples < 2:
            return 1.0
        # Autocorrelation up to lag where it crosses 0 or n_samples/4
        max_lag = min(n_samples // 4, 100)
        ess_vals = []
        for c in range(n_chains):
            x = chains[c] - chains[c].mean()
            var = np.dot(x, x) / n_samples
            rho = np.ones(max_lag + 1)
            for lag in range(1, max_lag + 1):
                rho[lag] = np.dot(x[lag:], x[:-lag])
                rho[lag] = rho[lag] / (np.dot(x[n_samples - lag:], x[n_samples - lag:]) + 1e-30)
            rho /= rho[0] + 1e-30
            # Sum pairs
            tau = 1.0
            for lag in range(1, max_lag, 2):
                pair_sum = rho[lag] + rho[min(lag + 1, max_lag)]
                if pair_sum <= 0:
                    break
                tau += 2.0 * pair_sum
            ess_vals.append(n_samples / tau if tau > 0 else 1.0)
        return float(np.mean(ess_vals))

    @staticmethod
    def _rhat(chains):
        """Gelman-Rubin R-hat statistic."""
        n_chains, n_samples = chains.shape
        if n_chains < 2:
            return 1.0
        # Split each chain in half
        half = n_samples // 2
        split = np.concatenate([chains[:, :half], chains[:, half:2 * half]], axis=0)
        m, N = split.shape  # m = 2*n_chains, N = half
        means = split.mean(axis=1)
        grand_mean = means.mean()
        B = N / (m - 1) * np.sum((means - grand_mean) ** 2)
        W = split.var(axis=1).mean()
        var_plus = (N - 1) / N * W + B / N
        rhat = np.sqrt(var_plus / W) if W > 0 else 1.0
        return float(np.clip(rhat, 1.0, None))

    # ── Reporting ──────────────────────────────────────────────────────────

    def diagnostics(self):
        """Print MCMC diagnostics table."""
        if not self.diagnostics_:
            print("No diagnostics available. Run sample() first.")
            return

        d = self.diagnostics_
        print(f"\n{'=' * 72}")
        print(f"MCMC Diagnostics ({d['n_chains']} chains × {d['n_samples']} samples)")
        print(f"{'=' * 72}")
        print(f"{'Parameter':<10} {'Mean':>12} {'± Std':>10}  "
              f"{'68% CI':>20} {'95% CI':>24}  {'ESS':>8} {'R̂':>6}")
        print(f"{'─' * 10} {'─' * 12} {'─' * 10}  {'─' * 20} {'─' * 24}  {'─' * 8} {'─' * 6}")

        for name in _PNAMES:
            di = d[name]
            ci68 = f"[{di['ci68'][0]:.4e}, {di['ci68'][1]:.4e}]"
            ci95 = f"[{di['ci95'][0]:.4e}, {di['ci95'][1]:.4e}]"
            print(f"{name:<10} {di['mean']:12.4e} {di['std']:10.4e}  "
                  f"{ci68:>20} {ci95:>24}  {di['ess']:8.1f} {di['rhat']:6.3f}")

        print(f"{'=' * 72}")

    def save(self, filepath):
        """Save chains and diagnostics to .npz file."""
        np.savez(filepath, chains=self.chains, diagnostics=np.array([self.diagnostics_]))
        print(f"Saved to {filepath}")

    def correlation_matrix(self):
        """Return (6, 6) posterior correlation matrix from chains."""
        if self.chains is None:
            raise RuntimeError("No chains. Run sample() first.")
        n_chains, n_samples, d = self.chains.shape
        flat = self.chains.reshape(-1, d)
        return np.corrcoef(flat.T)


# ═══════════════════════════════════════════════════════════════════════════════
# Evidence computation — Laplace approximation
# ═══════════════════════════════════════════════════════════════════════════════

def laplace_log_evidence(neg_log_prob_fn, theta_map):
    """Compute Laplace-approximated log marginal likelihood.

    Uses finite-difference Hessian on CPU to avoid GPU OOM on fine grids.

    Args:
        neg_log_prob_fn: callable θ → scalar, negative log posterior
        theta_map: (d,) float64, MAP estimate

    Returns:
        ln_z: float, log marginal likelihood
        details: dict with keys ``neg_log_map``, ``log_det_hessian``
    """
    t = np.array(theta_map, dtype=DTYPE_NP)
    d = t.shape[0]

    # Evaluate on CPU to avoid GPU memory contention
    with jax.default_device(jax.devices('cpu')[0]):
        neg_log_map = float(neg_log_prob_fn(jnp.array(t)))

    # Finite-difference Hessian (CPU-safe)
    eps = 1e-4
    H = np.zeros((d, d))
    cpu_dev = jax.devices('cpu')[0]
    for i in range(d):
        ei = np.zeros(d); ei[i] = eps
        for j in range(i, d):
            ej = np.zeros(d); ej[j] = eps
            with jax.default_device(cpu_dev):
                fpp = float(neg_log_prob_fn(jnp.array(t + ei + ej)))
                fpm = float(neg_log_prob_fn(jnp.array(t + ei - ej)))
                fmp = float(neg_log_prob_fn(jnp.array(t - ei + ej)))
                fmm = float(neg_log_prob_fn(jnp.array(t - ei - ej)))
            h_ij = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
            H[i, j] = h_ij
            H[j, i] = h_ij

    sign, log_det = np.linalg.slogdet(H)
    if sign <= 0:
        log_det = float(np.linalg.slogdet(H + 1e-6 * np.eye(d))[1])

    ln_z = -neg_log_map + 0.5 * d * math.log(2.0 * math.pi) - 0.5 * float(log_det)

    return ln_z, {'neg_log_map': neg_log_map, 'log_det_hessian': float(log_det)}


# ═══════════════════════════════════════════════════════════════════════════════
# MAP estimation — L-BFGS-B with analytical gradients
# ═══════════════════════════════════════════════════════════════════════════════

def find_map(neg_log_prob_fn, theta_init, bounds=None, maxiter=200):
    """Find posterior mode via L-BFGS-B.

    Args:
        neg_log_prob_fn: callable θ → scalar
        theta_init: (d,) initial guess
        bounds: list of (lo, hi) pairs or None
        maxiter: max L-BFGS-B iterations

    Returns:
        theta_map: (d,) np.ndarray
        result: scipy OptimizeResult
    """
    from scipy.optimize import minimize

    fn_jit = jax.jit(neg_log_prob_fn)
    grd_jit = jax.jit(jax.grad(neg_log_prob_fn))

    count = [0]

    def _obj(x):
        count[0] += 1
        f = float(fn_jit(jnp.array(x)))
        g = np.array(grd_jit(jnp.array(x)), dtype=DTYPE_NP)
        return f, g

    res = minimize(_obj, np.array(theta_init, dtype=DTYPE_NP),
                   method='L-BFGS-B', jac=True,
                   bounds=bounds,
                   options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})

    return res.x, res


# ═══════════════════════════════════════════════════════════════════════════════
# Correlation matrix formatting
# ═══════════════════════════════════════════════════════════════════════════════

def format_correlation(corr_matrix, param_names=None):
    """Pretty-print a correlation matrix.

    Args:
        corr_matrix: (d, d) ndarray
        param_names: list of strings or None (uses _PNAMES)
    """
    if param_names is None:
        param_names = _PNAMES
    d = corr_matrix.shape[0]
    header = " " * 10 + "".join(f"{n:>10}" for n in param_names)
    sep = "─" * 10 + "".join("─" * 10 for _ in range(d))
    lines = [header, sep]
    for i, name in enumerate(param_names):
        row = f"{name:<10}" + "".join(f"{corr_matrix[i, j]:10.3f}" for j in range(d))
        lines.append(row)
    return "\n".join(lines)
