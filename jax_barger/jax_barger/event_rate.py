"""Event rate calculation from oscillation probabilities.

Implements the same logic as BinnedInteraction::UpdatePrediction():
    event_rate(channel, bin) = sum[
        P(from_nua → to_nuch) * flux(from_nua, cosθ, E) * xsec(to_nuch, E)
    ]

for all four channels: numu, numubar, nue, nuebar.
Includes Poisson chi2 statistic.
"""

import jax.numpy as jnp


def event_rate(P, flux, xsec):
    """Compute expected event rate histograms from oscillation probabilities.

    Args:
        P:    (2, 3, 3, nE, nCos) float, oscillation probabilities
               axis 0: nu/antinu, axis 1: from flavor, axis 2: to flavor
               (nue=0, numu=1, nutau=2)
        flux: dict with keys 'numu','numubar','nue','nuebar'
               each (nE, nCos) float, atmospheric flux
        xsec: dict with keys 'numu','numubar','nue','nuebar'
               each (nE,) float, cross-section

    Returns:
        events: dict with same 4 keys, each (nE, nCos) float
    """
    P_nu = P[0]; P_anu = P[1]

    # numu channel: muon-like, from νe or νμ → νμ
    events_numu = (
        P_nu[0, 1] * flux['nue']        # P(νe→νμ) × Φ_νe
        + P_nu[1, 1] * flux['numu']     # P(νμ→νμ) × Φ_νμ
    ) * xsec['numu'][:, None]

    events_numubar = (
        P_anu[0, 1] * flux['nuebar']     # P(ν̄e→ν̄μ) × Φ_ν̄e
        + P_anu[1, 1] * flux['numubar']  # P(ν̄μ→ν̄μ) × Φ_ν̄μ
    ) * xsec['numubar'][:, None]

    # nue channel: electron-like, from νe or νμ → νe
    events_nue = (
        P_nu[0, 0] * flux['nue']        # P(νe→νe) × Φ_νe
        + P_nu[1, 0] * flux['numu']     # P(νμ→νe) × Φ_νμ
    ) * xsec['nue'][:, None]

    events_nuebar = (
        P_anu[0, 0] * flux['nuebar']     # P(ν̄e→ν̄e) × Φ_ν̄e
        + P_anu[1, 0] * flux['numubar']  # P(ν̄μ→ν̄e) × Φ_ν̄μ
    ) * xsec['nuebar'][:, None]

    return {
        'numu':    events_numu,
        'numubar': events_numubar,
        'nue':     events_nue,
        'nuebar':  events_nuebar,
    }


def poisson_chi2(data, pred):
    """Poisson log-likelihood chi2: 2 * sum(pred - data + data * log(data/pred))."""
    safe_data = jnp.maximum(data, 1e-30)
    safe_pred = jnp.maximum(pred, 1e-30)
    return 2.0 * jnp.sum(safe_pred - safe_data + safe_data * jnp.log(safe_data / safe_pred))


def rebin_2d(arr, E_rebin, cos_rebin):
    """Rebin a 2D array (nE, nCos) by summing groups of bins."""
    nE, nC = arr.shape
    nE_new = nE // E_rebin
    nC_new = nC // cos_rebin
    reshaped = arr[:nE_new * E_rebin, :nC_new * cos_rebin].reshape(
        nE_new, E_rebin, nC_new, cos_rebin)
    return reshaped.sum(axis=(1, 3))


def total_chi2(events_data, events_pred):
    """Sum of Poisson chi2 over all four channels."""
    chi2 = 0.0
    for ch in ['numu', 'numubar', 'nue', 'nuebar']:
        chi2 += poisson_chi2(events_data[ch], events_pred[ch])
    return chi2
