"""Matter-effect cubic eigenvalue equation for the Barger propagator.

Implements getMfast() and get_product() from
external/CUDAProb3/src/physics/barger.cuh (Barger et al., PRD 22.11, 1980).

All computations use vectorized JAX operations to avoid excessive computation
graph size from Python for-loops with scatter updates.
"""

import jax.numpy as jnp
from jax_barger.config import tworttwoGf, LoEfac


def get_matter_eigenvalues(E, rho_e, is_antineutrino, U_re, U_im, dm, order):
    """Compute matter-modified mass eigenvalues and their differences.

    Args:
        E:                 neutrino energy [GeV]
        rho_e:             electron density [mol/cm³-equivalent]
        is_antineutrino:   bool/int, if True use ν̄ convention (V > 0)
        U_re, U_im:        (3,3) float, PMNS matrix
        dm:                (3,3) float, vacuum mass differences DM[i][j] = mᵢ - mⱼ
        order:             (3,) int, eigenstate ordering for mass hierarchy

    Returns:
        dmMatMat: (3,3) float, m_i^mat - m_j^mat
        dmMatVac: (3,3) float, m_i^mat - m_j^vac
    """
    fac = jnp.where(is_antineutrino,
                      tworttwoGf * E * rho_e,
                     -tworttwoGf * E * rho_e)

    # |U_ei|² for i=0,1,2
    Ue_sq = U_re[0]**2 + U_im[0]**2  # (3,)

    # Eq. 22: cubic coefficients
    alpha = fac + dm[0, 1] + dm[0, 2]
    beta  = (dm[0, 1] * dm[0, 2]
             + fac * (dm[0, 1] * (1.0 - Ue_sq[1])
                     + dm[0, 2] * (1.0 - Ue_sq[2])))
    gamma = fac * dm[0, 1] * dm[0, 2] * Ue_sq[0]

    # Eq. 21: solve cubic via trigonometric formula
    tmp_raw = alpha**2 - 3.0 * beta
    tmp = jnp.maximum(tmp_raw, 0.0)

    argtmp = (2.0 * alpha**3 - 9.0 * alpha * beta + 27.0 * gamma) / (
        2.0 * jnp.sqrt(tmp**3))
    arg = jnp.clip(argtmp, -1.0, 1.0)

    pi = jnp.pi
    theta0 = jnp.arccos(arg) / 3.0
    theta1 = theta0 - 2.0 * pi / 3.0
    theta2 = theta0 + 2.0 * pi / 3.0

    base  = -(2.0 / 3.0) * jnp.sqrt(tmp)
    shift = dm[0, 0] - alpha / 3.0

    mMatU = jnp.array([
        base * jnp.cos(theta0) + shift,
        base * jnp.cos(theta1) + shift,
        base * jnp.cos(theta2) + shift,
    ])  # (3,)

    # Reorder to vacuum ordering
    mMat = mMatU[order]  # (3,)

    # dmMatMat[i][j] = mMat[i] - mMat[j],  dmMatVac[i][j] = mMat[i] - dm[j,0]
    dmMatMat = mMat[:, None] - mMat[None, :]       # (3,3)
    dmMatVac = mMat[:, None] - dm[None, :, 0]       # (3,3)

    return dmMatMat, dmMatVac


def get_product_matrix(L, E, rho_e, is_antineutrino, U_re, U_im,
                       dmMatVac, dmMatMat):
    """Compute the X matrix via Eq. (11) from Barger et al.

    X[i][j] = sum_k exp(-i·LoEfac·dmMatVac[k][0]·L/E) ·
              (sum_l twoEHmM[i][l][(k+1)%3] * twoEHmM[l][j][(k+2)%3])
              / div[k]

    Returns:
        X_re: (3,3) float, real part
        X_im: (3,3) float, imaginary part
    """
    fac = jnp.where(is_antineutrino, tworttwoGf * E * rho_e, -tworttwoGf * E * rho_e)

    # Build 2EH-M_j matrices: shape (3, 3, 3) = [n, m, j]
    # Base: -fac * U_en^* U_em, same for all j
    base_re = -fac * (U_re[0, :, None] * U_re[0, None, :]
                      + U_im[0, :, None] * U_im[0, None, :])  # (3,3)
    base_im = -fac * (U_re[0, :, None] * U_im[0, None, :]
                      - U_im[0, :, None] * U_re[0, None, :])  # (3,3)

    # twoEHmM[n,m,j] for j=0,1,2
    twoEHmM_re = jnp.broadcast_to(base_re[..., None], (3, 3, 3))
    twoEHmM_im = jnp.broadcast_to(base_im[..., None], (3, 3, 3))

    # Subtract dmMatVac[j][n] on diagonal
    twoEHmM_re = twoEHmM_re.at[0, 0, :].add(-dmMatVac[:, 0])
    twoEHmM_re = twoEHmM_re.at[1, 1, :].add(-dmMatVac[:, 1])
    twoEHmM_re = twoEHmM_re.at[2, 2, :].add(-dmMatVac[:, 2])

    # Compute product[n,m,k] via matrix multiply over l
    # For k=0: sum_l M[n,l,1] * M[l,m,2] = (M[:,:,1] @ M[:,:,2])[n,m]
    # For k=1: sum_l M[n,l,2] * M[l,m,0] = (M[:,:,2] @ M[:,:,0])[n,m]
    # For k=2: sum_l M[n,l,0] * M[l,m,1] = (M[:,:,0] @ M[:,:,1])[n,m]

    def _matmul_re_im(A_re, A_im, B_re, B_im):
        return (A_re @ B_re - A_im @ B_im, A_re @ B_im + A_im @ B_re)

    prod_re = jnp.zeros((3, 3, 3))
    prod_im = jnp.zeros((3, 3, 3))

    for k in range(3):
        k1 = (k + 1) % 3
        k2 = (k + 2) % 3
        pr, pi = _matmul_re_im(
            twoEHmM_re[:, :, k1], twoEHmM_im[:, :, k1],
            twoEHmM_re[:, :, k2], twoEHmM_im[:, :, k2])
        prod_re = prod_re.at[:, :, k].set(pr)
        prod_im = prod_im.at[:, :, k].set(pi)

    # Divide by eigenvalue differences
    d01 = dmMatMat[0, 1]; d02 = dmMatMat[0, 2]
    d12 = dmMatMat[1, 2]; d10 = dmMatMat[1, 0]
    d20 = dmMatMat[2, 0]; d21 = dmMatMat[2, 1]

    div = jnp.array([d01 * d02, d12 * d10, d20 * d21])
    prod_re = prod_re / div[None, None, :]
    prod_im = prod_im / div[None, None, :]

    # X[i,j] = sum_k exp(-i * LoEfac * dmMatVac[k,0] * L/E) * product[i,j,k]
    arg = -LoEfac * dmMatVac[:, 0] * L / E  # (3,) = [k]
    c = jnp.cos(arg); s = jnp.sin(arg)      # (3,)

    # X_re[n,m] = sum_k (c[k]*prod_re[n,m,k] - s[k]*prod_im[n,m,k])
    X_re = jnp.sum(c[None, None, :] * prod_re - s[None, None, :] * prod_im, axis=-1)
    X_im = jnp.sum(c[None, None, :] * prod_im + s[None, None, :] * prod_re, axis=-1)

    return X_re, X_im
