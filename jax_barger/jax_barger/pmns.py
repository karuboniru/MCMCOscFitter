"""PMNS matrix builder for neutrino oscillations.

Implements the standard PDG parameterization, matching the convention in
external/Prob3plusplus/mosc.c (setmix_sin) and
external/CUDAProb3/include/cudaprob3/oscillation_params.hpp (fillMixMatrix).

The PMNS matrix U is a 3×3 unitary matrix constructed from three mixing angles
and one Dirac CP-violating phase:

  U = [ c12·c13                           s12·c13                       s13·e^{-iδ}      ]
      [ -s12·c23 - c12·s23·s13·e^{iδ}     c12·c23 - s12·s23·s13·e^{iδ}  s23·c13         ]
      [  s12·s23 - c12·c23·s13·e^{iδ}    -c12·s23 - s12·c23·s13·e^{iδ}  c23·c13         ]

For antineutrinos: U → U* (complex conjugate, equivalent to δCP → -δCP).
"""

import jax.numpy as jnp


def build_pmns(theta12, theta13, theta23, deltacp):
    """Build the 3×3 PMNS mixing matrix.

    Args:
        theta12:  mixing angle θ₁₂ [rad]
        theta13:  mixing angle θ₁₃ [rad]
        theta23:  mixing angle θ₂₃ [rad]
        deltacp:  Dirac CP phase δ_CP [rad]

    Returns:
        U_re: (3, 3) float64, real part of PMNS matrix
        U_im: (3, 3) float64, imaginary part of PMNS matrix
    """
    s12 = jnp.sin(theta12); c12 = jnp.cos(theta12)
    s13 = jnp.sin(theta13); c13 = jnp.cos(theta13)
    s23 = jnp.sin(theta23); c23 = jnp.cos(theta23)
    sd  = jnp.sin(deltacp); cd  = jnp.cos(deltacp)

    U_re = jnp.zeros((3, 3))
    U_im = jnp.zeros((3, 3))

    U_re = U_re.at[0, 0].set( c12 * c13)
    U_re = U_re.at[0, 1].set( s12 * c13)
    U_re = U_re.at[0, 2].set( s13 * cd)
    U_im = U_im.at[0, 2].set(-s13 * sd)

    U_re = U_re.at[1, 0].set(-s12 * c23 - c12 * s23 * s13 * cd)
    U_im = U_im.at[1, 0].set(-c12 * s23 * s13 * sd)
    U_re = U_re.at[1, 1].set( c12 * c23 - s12 * s23 * s13 * cd)
    U_im = U_im.at[1, 1].set(-s12 * s23 * s13 * sd)
    U_re = U_re.at[1, 2].set( s23 * c13)

    U_re = U_re.at[2, 0].set( s12 * s23 - c12 * c23 * s13 * cd)
    U_im = U_im.at[2, 0].set(-c12 * c23 * s13 * sd)
    U_re = U_re.at[2, 1].set(-c12 * s23 - s12 * c23 * s13 * cd)
    U_im = U_im.at[2, 1].set(-s12 * c23 * s13 * sd)
    U_re = U_re.at[2, 2].set( c23 * c13)

    return U_re, U_im


def build_dm(dm21sq, dm32sq):
    """Build the vacuum mass-squared difference matrix DM[i,j] = m_i - m_j.

    The mass basis is:
        m₀ = 0 (lightest mass)
        m₁ = m₀ + Δm²₂₁ = 0 + Δm²₂₁
        m₂ = m₁ + Δm²₃₂ = Δm²₂₁ + Δm²₃₂  (NH) or = Δm²₂₁ + Δm²₃₂ (IH)

    A small epsilon shift is applied to prevent exact degeneracies.

    Args:
        dm21sq:  Δm²₂₁ [eV²], always positive
        dm32sq:  Δm²₃₂ [eV²], positive for NH, negative for IH

    Returns:
        dm: (3, 3) float64, DM[i][j] = m_i - m_j
    """
    eps = 5.0e-9
    mVac0 = jnp.where(dm21sq == 0.0, -eps, 0.0)
    mVac1 = dm21sq
    mVac2 = dm21sq + dm32sq + jnp.where(dm32sq == 0.0, eps, 0.0)

    dm = jnp.zeros((3, 3))
    dm = dm.at[0, 1].set(mVac0 - mVac1); dm = dm.at[1, 0].set(-(mVac0 - mVac1))
    dm = dm.at[0, 2].set(mVac0 - mVac2); dm = dm.at[2, 0].set(-(mVac0 - mVac2))
    dm = dm.at[1, 2].set(mVac1 - mVac2); dm = dm.at[2, 1].set(-(mVac1 - mVac2))
    return dm


def compute_mass_order(dm):
    """Determine how matter eigenstates map to vacuum eigenstates.

    Solves the vacuum cubic eigenvalue problem to find the three vacuum mass
    roots, then matches each matter root to the closest vacuum root.

    Args:
        dm: (3, 3) float64, mass difference matrix

    Returns:
        order: (3,) int32, ORDER[i] = index of matter eigenstate matching vacuum state i
    """
    dm01 = dm[0, 1]
    dm02 = dm[0, 2]
    alphaV = dm01 + dm02
    betaV  = dm01 * dm02

    tmpV_raw = alphaV * alphaV - 3.0 * betaV
    tmpV = jnp.maximum(tmpV_raw, 0.0)

    argtmp = (2.0 * alphaV**3 - 9.0 * alphaV * betaV) / (2.0 * jnp.sqrt(tmpV**3))
    arg = jnp.clip(argtmp, -1.0, 1.0)

    pi = jnp.pi
    th0 = jnp.arccos(arg) / 3.0
    base = -(2.0 / 3.0) * jnp.sqrt(tmpV)
    shift = dm[0, 0] - alphaV / 3.0

    mMatV = jnp.array([
        base * jnp.cos(th0) + shift,
        base * jnp.cos(th0 - 2.0 * pi / 3.0) + shift,
        base * jnp.cos(th0 + 2.0 * pi / 3.0) + shift,
    ])

    order = jnp.zeros(3, dtype=jnp.int32)
    for i in range(3):
        best = jnp.argmin(jnp.abs(dm[i, 0] - mMatV))
        order = order.at[i].set(best)
    return order
