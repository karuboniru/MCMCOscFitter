"""JAX-based Barger neutrino oscillation propagator with differentiable event rates.

Key API:
    from jax_barger import oscillation_probabilities, oscillation_probabilities_from_params
    from jax_barger.event_rate import event_rate, poisson_chi2, total_chi2
    from jax_barger.earth import default_prem
    from jax_barger.config import R_earth, h_prod, tworttwoGf, LoEfac, MAX_LAYERS
    from jax_barger.mcmc import HMCSampler, find_map, laplace_log_evidence
"""

from jax_barger.barger import (
    oscillation_probabilities,
    oscillation_probabilities_from_params,
)
from jax_barger.event_rate import event_rate, poisson_chi2, total_chi2, rebin_2d
from jax_barger.earth import default_prem, electron_density

__all__ = [
    "oscillation_probabilities",
    "oscillation_probabilities_from_params",
    "event_rate",
    "poisson_chi2",
    "total_chi2",
    "rebin_2d",
    "default_prem",
    "electron_density",
]
