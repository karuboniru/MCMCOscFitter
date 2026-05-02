"""Earth PREM density model and path geometry for neutrino oscillations.

Implements the same path-length logic as the CUDAProb3 GPU kernel:
    external/CUDAProb3/src/kernels/oscillation_kernel.cu
    external/CUDAProb3/src/physics/earth_model.cuh

All layer data is padded to MAX_LAYERS for uniform JAX vmap/JIT tracing.
"""

import jax.numpy as jnp
from jax_barger.config import R_earth, h_prod

MAX_LAYERS = 10   # max possible: atmosphere + 4 shells in + deepest + 4 shells out


def default_prem():
    """Return the default PREM Earth model (5 layers including center).

    Radii are in descending order (outermost first), matching the convention
    used by cudaprob3::PREMModel.

    Returns:
        radii:   (5,) float64, shell outer radii [km], descending
        density: (5,) float64, mass density [g/cm³]
        Ye:      (5,) float64, electron fraction
    """
    radii   = jnp.array([6371.0, 5701.0, 3480.0, 1220.0, 0.0])
    density = jnp.array([3.3, 5.0, 11.3, 13.0, 13.0])
    Ye      = jnp.array([0.497, 0.497, 0.468, 0.468, 0.468])
    return radii, density, Ye


def electron_density(density, Ye):
    """Effective electron density for matter potential."""
    return density * Ye


def cos_limits(radii):
    """Compute cos(zenith) thresholds for each PREM shell boundary."""
    RE2 = R_earth * R_earth
    n = radii.shape[0]
    cos_lim = jnp.zeros(n)
    for i in range(n):
        r2 = radii[i] * radii[i]
        cos_lim = cos_lim.at[i].set(
            jnp.where(i == 0, 0.0, -jnp.sqrt(jnp.maximum(0.0, 1.0 - r2 / RE2)))
        )
    return cos_lim


def max_layers(cos_theta, cos_lim):
    """Number of PREM shell boundaries crossed for a given cosθ."""
    return jnp.sum(cos_lim > cos_theta)


def path_length_geometry(cos_theta):
    """Compute total path geometry for a neutrino produced at height h_prod.

    Returns:
        path_length:       total path from production to far-side exit [km]
        total_earth_length: total chord through Earth [km]
    """
    RE = R_earth
    h  = h_prod
    sin2 = 1.0 - cos_theta * cos_theta
    path_length = jnp.sqrt((RE + h)**2 - RE**2 * sin2) - RE * cos_theta
    total_earth = jnp.where(cos_theta < 0, -2.0 * cos_theta * RE, 0.0)
    return path_length, total_earth


def precompute_path_data(cos_grid, prem_radii, prem_density, prem_Ye):
    """Precompute padded layer distances and densities for all cosθ values.

    Returns arrays of shape (nCos, MAX_LAYERS) that can be vmapped over.

    Args:
        cos_grid:     (nCos,) float64, cosine(zenith) values
        prem_radii:   (5,) float64, PREM radii [km], descending
        prem_density: (5,) float64, PREM density [g/cm³]
        prem_Ye:      (5,) float64, PREM electron fraction

    Returns:
        dist:  (nCos, MAX_LAYERS) float64, path length per layer [km]
        rhoe:  (nCos, MAX_LAYERS) float64, electron density per layer
    """
    rho_e = electron_density(prem_density, prem_Ye)
    cos_lim = cos_limits(prem_radii)
    RE = R_earth

    nCos = cos_grid.shape[0]
    dist = jnp.zeros((nCos, MAX_LAYERS))
    rhoe_arr = jnp.zeros((nCos, MAX_LAYERS))

    for ic in range(nCos):
        cos_theta = cos_grid[ic]
        ml = max_layers(cos_theta, cos_lim)
        path_len, earth_len = path_length_geometry(cos_theta)
        sin2 = 1.0 - cos_theta * cos_theta

        # Layer 0: atmosphere
        atm_dist = jnp.where(cos_theta >= 0, path_len, path_len - earth_len)
        dist = dist.at[ic, 0].set(atm_dist)

        # Ingoing Earth layers (1 to ml)
        for lyr in range(1, ml + 1):
            i = lyr - 1   # PREM shell index
            cross_this = 2.0 * jnp.sqrt(jnp.maximum(0.0, prem_radii[i]**2 - RE**2 * sin2))
            cross_next = 2.0 * jnp.sqrt(jnp.maximum(0.0, prem_radii[i + 1]**2 - RE**2 * sin2))
            d = jnp.where(i < ml - 1, 0.5 * (cross_this - cross_next), cross_this)
            # Ingoing: store at position lyr
            dist = dist.at[ic, lyr].set(d)
            rhoe_arr = rhoe_arr.at[ic, lyr].set(rho_e[jnp.minimum(i, rho_e.shape[0] - 1)])

        # Outgoing Earth layers: mirrored from ingoing, but in REVERSE order.
        # Physically: on the way out, the neutrino crosses the innermost shell
        # first, then progressively outer shells.
        # Store outgoing layer at index ml+lyr with the value from ingoing layer ml-lyr.
        for lyr in range(1, ml):
            dist = dist.at[ic, ml + lyr].set(dist[ic, ml - lyr])
            rhoe_arr = rhoe_arr.at[ic, ml + lyr].set(rhoe_arr[ic, ml - lyr])

    return dist, rhoe_arr
