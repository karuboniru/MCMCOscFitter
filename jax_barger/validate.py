"""Forward validation: compare JAX Barger output against C++ ParProb3ppOscillation.

Usage:
    cd /var/home/yan/codes/MCMCOscFitter/jax_barger
    JAX_PLATFORMS=cpu .venv/bin/python validate.py

Tests:
    1. Oscillation probabilities: JAX vs C++ (3-flavour, 2D grid)
    2. Event rates: JAX vs C++ (using same flux/xsec)

The C++ CUDAProb3 uses float precision (oscillaton_calc_precision = float),
while JAX uses float64. Expected absolute agreement is at the ~1e-4 level
(limited by float32 precision).
"""

import sys
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

# Add build/pybind to path for mcmcoscfitter
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(REPO_ROOT, 'build', 'pybind'))
sys.path.insert(0, REPO_ROOT)

import mcmcoscfitter as mof
from jax_barger.barger import oscillation_probabilities
from jax_barger.earth import default_prem
from jax_barger.event_rate import event_rate


DEFAULT_PARAMS = {
    'theta12': np.arcsin(np.sqrt(0.307)),
    'theta13': np.arcsin(np.sqrt(0.02195)),
    'theta23': np.arcsin(np.sqrt(0.565)),
    'deltacp':  3.98,
    'dm21sq':   7.53e-5,
    'dm32sq':   2.517e-3,
}

TEST_PARAMS = [
    DEFAULT_PARAMS,
    {'theta12': np.arcsin(np.sqrt(0.307)), 'theta13': np.arcsin(np.sqrt(0.0221)),
     'theta23': np.arcsin(np.sqrt(0.572)), 'deltacp': 4.71,
     'dm21sq': 7.53e-5, 'dm32sq': -2.498e-3},
    {'theta12': np.arcsin(np.sqrt(0.307)), 'theta13': np.arcsin(np.sqrt(0.02195)),
     'theta23': np.arcsin(np.sqrt(0.565)), 'deltacp': 0.0,
     'dm21sq': 7.53e-5, 'dm32sq': 2.517e-3},
    {'theta12': np.arcsin(np.sqrt(0.307)), 'theta13': np.arcsin(np.sqrt(0.02195)),
     'theta23': np.arcsin(np.sqrt(0.565)), 'deltacp': np.pi/2,
     'dm21sq': 7.53e-5, 'dm32sq': 2.517e-3},
]


def params_to_mof(params):
    p = mof.OscillationParameters()
    p.set_toggle(mof.all_on)
    p.set_param(mof.Param(
        params['dm32sq'], params['dm21sq'],
        np.sin(params['theta23'])**2, np.sin(params['theta13'])**2,
        np.sin(params['theta12'])**2, params['deltacp']))
    return p


def make_bin_edges(centers, lo=None, hi=None):
    """Construct bin edges from bin centers, ensuring boundaries are valid."""
    n = len(centers)
    edges = np.zeros(n + 1)
    if n == 1:
        edges[0] = centers[0] * 0.9
        edges[1] = centers[0] * 1.1
        if lo is not None and edges[0] < lo:
            edges[0] = lo
        if hi is not None and edges[1] > hi:
            edges[1] = hi
        return edges

    half = (centers[1] - centers[0]) / 2.0
    edges[0] = centers[0] - half
    if lo is not None and edges[0] < lo:
        edges[0] = lo
    half = (centers[-1] - centers[-2]) / 2.0
    edges[-1] = centers[-1] + half
    if hi is not None and edges[-1] > hi:
        edges[-1] = hi
    for i in range(1, n):
        edges[i] = (centers[i-1] + centers[i]) / 2.0
    return edges


def validate_probabilities(E_grid, cos_grid, params, prem, atol=1e-4):
    """Compare JAX and C++ oscillation probabilities."""
    P_jax = np.array(oscillation_probabilities(
        jnp.array(E_grid), jnp.array(cos_grid),
        params['theta12'], params['theta13'], params['theta23'],
        params['deltacp'], params['dm21sq'], params['dm32sq'],
        prem['radii'], prem['density'], prem['Ye']))

    E_c = E_grid.astype(np.float32)
    c_c = cos_grid.astype(np.float32)
    E_edges = make_bin_edges(E_grid, lo=1e-3)
    c_edges = make_bin_edges(cos_grid, lo=-1.0, hi=1.0)

    p_mof = params_to_mof(params)
    prop = mof.ParProb3ppOscillation(E_c, c_c)
    P_cpp = np.array(prop.get_prob_hists_3f(
        E_edges.tolist(), c_edges.tolist(), p_mof))

    max_err = np.max(np.abs(P_jax - P_cpp))
    rms_err = np.sqrt(np.mean((P_jax - P_cpp)**2))
    mask = np.abs(P_cpp) > 1e-6
    rel_errors = np.abs(P_jax[mask] - P_cpp[mask]) / np.abs(P_cpp[mask])
    max_rel = np.max(rel_errors) if len(rel_errors) > 0 else 0.0

    return max_err, max_rel, rms_err


def validate_event_rates(E_grid, cos_grid, params, prem, atol=1e-4):
    """Compare JAX and C++ event rates using the same flux/xsec."""
    E_edges = make_bin_edges(E_grid, lo=1e-3)
    c_edges = make_bin_edges(cos_grid, lo=-1.0, hi=1.0)
    E_c = E_grid.astype(np.float32)
    c_c = cos_grid.astype(np.float32)

    # Load flux/xsec from C++
    pi = mof.load_physics_input(E_edges.tolist(), c_edges.tolist())
    flux = {k: np.array(pi[f'flux_{k}']) for k in ['numu', 'numubar', 'nue', 'nuebar']}
    xsec = {k: np.array(pi[f'xsec_{k}']) for k in ['numu', 'numubar', 'nue', 'nuebar']}

    # JAX probabilities and event rates
    P_jax = np.array(oscillation_probabilities(
        jnp.array(E_grid), jnp.array(cos_grid),
        params['theta12'], params['theta13'], params['theta23'],
        params['deltacp'], params['dm21sq'], params['dm32sq'],
        prem['radii'], prem['density'], prem['Ye']))
    events_jax = event_rate(P_jax, flux, xsec)

    # C++: use BinnedInteraction for full pipeline
    p_mof = params_to_mof(params)
    prop = mof.ParProb3ppOscillation(E_c, c_c)
    histos = mof.BinnedHistograms(
        flux_numu=pi['flux_numu'], flux_numubar=pi['flux_numubar'],
        flux_nue=pi['flux_nue'], flux_nuebar=pi['flux_nuebar'],
        xsec_numu=pi['xsec_numu'], xsec_numubar=pi['xsec_numubar'],
        xsec_nue=pi['xsec_nue'], xsec_nuebar=pi['xsec_nuebar'],
        Ebins=E_edges.tolist(), costhbins=c_edges.tolist())
    model = mof.BinnedInteraction(
        E_edges.tolist(), c_edges.tolist(), prop, histos)
    model.set_toggle(mof.all_on)
    model.set_param(mof.Param(
        params['dm32sq'], params['dm21sq'],
        np.sin(params['theta23'])**2, np.sin(params['theta13'])**2,
        np.sin(params['theta12'])**2, params['deltacp']))
    model.update_prediction()  # set_param does NOT trigger recomputation
    data = model.generate_data()
    events_cpp = {k: np.array(getattr(data, k)) for k in ['numu', 'numubar', 'nue', 'nuebar']}

    results = {}
    for ch in ['numu', 'numubar', 'nue', 'nuebar']:
        max_err = np.max(np.abs(events_jax[ch] - events_cpp[ch]))
        rms_err = np.sqrt(np.mean((events_jax[ch] - events_cpp[ch])**2))
        mask = np.abs(events_cpp[ch]) > 1e-6
        rels = np.abs(events_jax[ch][mask] - events_cpp[ch][mask]) / np.abs(events_cpp[ch][mask])
        max_rel = np.max(rels) if len(rels) > 0 else 0.0
        results[ch] = (max_err, max_rel, rms_err)
    return results


def main():
    print("=" * 60)
    print("JAX Barger Forward Validation")
    print("=" * 60)
    devices = jax.devices()
    print(f"JAX devices: {devices}")

    E_grid   = np.logspace(np.log10(0.1), np.log10(20.0), 30)
    cos_grid = np.linspace(-1.0, 1.0, 15)
    radii, density, Ye = default_prem()
    prem = {'radii': radii, 'density': density, 'Ye': Ye}

    print(f"\nGrid: {len(E_grid)} E × {len(cos_grid)} cosθ, PREM: {len(radii)} layers")
    print(f"E range: [{E_grid[0]:.2f}, {E_grid[-1]:.2f}] GeV")

    # Warmup
    print("\nWarming up JAX...")
    _ = oscillation_probabilities(
        jnp.array(E_grid[:2]), jnp.array(cos_grid[:2]),
        DEFAULT_PARAMS['theta12'], DEFAULT_PARAMS['theta13'], DEFAULT_PARAMS['theta23'],
        DEFAULT_PARAMS['deltacp'], DEFAULT_PARAMS['dm21sq'], DEFAULT_PARAMS['dm32sq'],
        prem['radii'], prem['density'], prem['Ye'])
    t0 = time.time()
    _ = oscillation_probabilities(
        jnp.array(E_grid), jnp.array(cos_grid),
        DEFAULT_PARAMS['theta12'], DEFAULT_PARAMS['theta13'], DEFAULT_PARAMS['theta23'],
        DEFAULT_PARAMS['deltacp'], DEFAULT_PARAMS['dm21sq'], DEFAULT_PARAMS['dm32sq'],
        prem['radii'], prem['density'], prem['Ye'])
    print(f"JIT warmup: {time.time()-t0:.1f}s")

    all_ok = True

    # Test 1: Oscillation probabilities
    print("\n─── Test 1: Oscillation probabilities ───")
    print(f"  (C++ uses float32; expected agreement ~1e-4)")
    for i, params in enumerate(TEST_PARAMS):
        t0 = time.time()
        max_err, max_rel, rms_err = validate_probabilities(
            E_grid, cos_grid, params, prem)
        dt = time.time() - t0
        labels = ["NH(δ=3.98)", "IH(δ=4.71)", "NH(δ=0)", "NH(δ=π/2)"]
        ok = max_err < 1e-3  # float32 tolerance
        status = "✓" if ok else "✗"
        print(f"  [{status}] {labels[i]}: max_abs={max_err:.2e} max_rel={max_rel:.2e} "
              f"rms={rms_err:.2e} ({dt:.1f}s)")
        all_ok = all_ok and ok

    # Test 2: Event rates
    print("\n─── Test 2: Event rates ───")
    t0 = time.time()
    results = validate_event_rates(E_grid, cos_grid, DEFAULT_PARAMS, prem)
    dt = time.time() - t0
    for ch in ['numu', 'numubar', 'nue', 'nuebar']:
        max_err, max_rel, rms_err = results[ch]
        ok = max_err < 1.0 or max_rel < 0.1
        status = "✓" if ok else "✗"
        print(f"  [{status}] {ch}: max_abs={max_err:.2e} max_rel={max_rel:.2e} "
              f"rms={rms_err:.2e}")
        all_ok = all_ok and ok
    print(f"  Event rate time: {dt:.1f}s")

    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("ALL VALIDATION TESTS PASSED ✓")
    else:
        print("SOME VALIDATION TESTS FAILED ✗")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
