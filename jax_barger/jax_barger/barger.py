"""Barger neutrino oscillation propagator in JAX.

Core propagation engine: given PMNS parameters, a neutrino energy, and
precomputed layer path data, computes the 3×3 oscillation probability matrix.

Matches the convention and output of CUDAProb3's ParProb3ppOscillation.
"""

import jax
import jax.numpy as jnp

from jax_barger.config import MAX_LAYERS
from jax_barger.pmns import build_pmns, build_dm, compute_mass_order
from jax_barger.earth import precompute_path_data
from jax_barger.matter import get_matter_eigenvalues, get_product_matrix


def oscillation_prob_layer(E, dist_arr, rhoe_arr, is_antineutrino,
                            U_re, U_im, dm, order):
    """Compute the 3×3 oscillation probability matrix for a single (E, cosθ).

    Uses precomputed layer data (padded to MAX_LAYERS). Layers with zero
    density or zero distance produce identity transitions.

    Args:
        E:               scalar, neutrino energy [GeV]
        dist_arr:        (MAX_LAYERS,) float64, path lengths per layer [km]
        rhoe_arr:        (MAX_LAYERS,) float64, electron density per layer
        is_antineutrino: bool scalar (0 or 1)
        U_re, U_im:      (3,3) float64, PMNS matrix (already conjugated for ν̄)
        dm:              (3,3) float64, vacuum mass differences
        order:           (3,) int, eigenstate ordering

    Returns:
        P: (3,3) float64, P[to_flavor, from_flavor]
    """
    A_re = jnp.eye(3)
    A_im = jnp.zeros((3, 3))

    def _apply_layer(lyr_idx, A_state):
        A_re, A_im = A_state
        d = dist_arr[lyr_idx]
        rho = rhoe_arr[lyr_idx]

        should_skip = d <= 0.0

        dmMatMat, dmMatVac = get_matter_eigenvalues(
            E, rho, is_antineutrino, U_re, U_im, dm, order)
        X_re, X_im = get_product_matrix(
            d, E, rho, is_antineutrino, U_re, U_im, dmMatVac, dmMatMat)

        U = U_re + 1j * U_im
        X = X_re + 1j * X_im
        A_layer = U @ X @ U.conj().T

        A = A_re + 1j * A_im
        A_new = A_layer @ A
        A_new_re = jnp.real(A_new)
        A_new_im = jnp.imag(A_new)

        return (
            jnp.where(should_skip, A_re, A_new_re),
            jnp.where(should_skip, A_im, A_new_im),
        )

    A_re, A_im = jax.lax.fori_loop(0, MAX_LAYERS, _apply_layer, (A_re, A_im))
    P = A_re**2 + A_im**2
    return P


# ─── Vectorized over (E, cosθ) ───────────────────────────────────────────────

_vmap_osc = jax.vmap(
    jax.vmap(oscillation_prob_layer,
             in_axes=(0, None, None, None, None, None, None, None)),
    in_axes=(None, 0, 0, None, None, None, None, None))


def oscillation_probabilities(E_grid, cos_grid, theta12, theta13, theta23,
                               deltacp, dm21sq, dm32sq, radii, density, Ye):
    """Compute full 3-flavour oscillation probability grid.

    Args:
        E_grid:   (nE,) float64, energy values [GeV]
        cos_grid: (nCos,) float64, cosine(zenith) values
        theta12, theta13, theta23: mixing angles [rad]
        deltacp:  Dirac CP phase [rad]
        dm21sq:   Δm²₂₁ [eV²]
        dm32sq:   Δm²₃₂ [eV²] (>0 NH, <0 IH)
        radii:    (5,) float64, PREM radii [km], descending
        density:  (5,) float64, PREM density [g/cm³]
        Ye:       (5,) float64, electron fraction

    Returns:
        P: (2, 3, 3, nE, nCos) float64
            axis 0: neutrino/antineutrino
            axes 1-2: to_flavor, from_flavor (nue=0, numu=1, nutau=2)
            axes 3-4: E, cosθ grid
    """
    U_re, U_im = build_pmns(theta12, theta13, theta23, deltacp)
    dm = build_dm(dm21sq, dm32sq)
    order = compute_mass_order(dm)

    # Precompute path data for all cosθ values (padded to MAX_LAYERS)
    dist, rhoe = precompute_path_data(cos_grid, radii, density, Ye)

    # Neutrino: original PMNS. Result shape: (nCos, nE, 3, 3)
    P_nu = _vmap_osc(E_grid, dist, rhoe, 0, U_re, U_im, dm, order)

    # Antineutrino: U → U* (complex conjugate), flip imaginary parts
    P_anu = _vmap_osc(E_grid, dist, rhoe, 1, U_re, -U_im, dm, order)

    # Transpose to (2, 3, 3, nE, nCos) with C++ convention:
    # axis 0: nu/antinu, axis 1: from_flavor, axis 2: to_flavor,
    # axis 3: E, axis 4: cosθ
    P = jnp.stack([P_nu, P_anu], axis=0)  # (2, nCos, nE, 3, 3)
    # Reorder: (2, 3(=from), 3(=to), nE, nCos)
    P = jnp.transpose(P, (0, 4, 3, 2, 1))  # (2, from, to, nE, nCos)
    return P


def oscillation_probabilities_from_params(E_grid, cos_grid, prem, params):
    """Convenience wrapper accepting a params dict."""
    return oscillation_probabilities(
        E_grid, cos_grid,
        params['theta12'], params['theta13'], params['theta23'],
        params['deltacp'], params['dm21sq'], params['dm32sq'],
        prem['radii'], prem['density'], prem['Ye'])
