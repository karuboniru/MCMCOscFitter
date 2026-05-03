"""Corner-plot of MCMC posterior samples.

Loads chains from .npz files (produced by run_mcmc.py or
run_hierarchy_mcmc.py) and draws a publication-quality corner plot:

  - diagonal: 1D histograms with 68/95% quantile lines
  - lower triangle: 2D density (Gaussian KDE on a grid)

Parameters are displayed in physics units (sin²θ not θ, eV² not rad).

Usage:
    cd jax_barger
    PYTHONPATH=../build/pybind:.. .venv/bin/python plot_corner.py hmc_chains_nh.npz [hmc_chains_ih.npz]
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

# ── Parameter display config ──

_LABELS = [
    r'$|\Delta m^2_{32}|\;\mathrm{[eV^2]}$',
    r'$\Delta m^2_{21}\;\mathrm{[eV^2]}$',
    r'$\sin^2\theta_{23}$',
    r'$\sin^2\theta_{13}$',
    r'$\delta_{\rm CP}\;\mathrm{[rad]}$',
    r'$\sin^2\theta_{12}$',
]

_SHORT = ['DM2', 'Dm2', 's2_23', 's2_13', 'DCP', 's2_12']


def load_chains(path):
    """Load chains from .npz: return (n_chains, n_samples, 6) array in θ-space."""
    data = np.load(path, allow_pickle=True)
    chains = data['chains']
    d = data['diagnostics'].item()
    return chains, d


def theta_to_display(chains):
    """Convert θ-space [DM2, Dm2, θ23, θ13, δCP, θ12] → display units.

    Display units: |DM2| (abs eV²), Dm2 (eV²); θ→sin²θ for T23,T13,T12.
    """
    out = chains.copy()
    out[..., 0] = np.abs(out[..., 0])      # |Δm²₃₂|
    for i in [2, 3, 5]:                    # θ23, θ13, θ12 → sin²θ
        out[..., i] = np.sin(out[..., i]) ** 2
    return out


def corner_plot(chains_dict, outpath, title=None):
    """Draw a corner plot for one or two posterior samples.

    Args:
        chains_dict: {label: (n_chains, n_chain, 6) array}
        outpath: output filename (.pdf or .png)
        title: optional super-title
    """
    n_params = 6
    # Flatten all chains per model
    flat = {}
    colors = {}
    for label, ch in chains_dict.items():
        flat[label] = ch.reshape(-1, n_params)
    colors_list = ['#1f77b4', '#d62728']
    for i, label in enumerate(flat):
        colors[label] = colors_list[i % len(colors_list)]

    fig, axes = plt.subplots(n_params, n_params, figsize=(14, 12))
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)

    # ── Global x-range per parameter ──
    all_flat = np.concatenate(list(flat.values()), axis=0)
    ranges = []
    for j in range(n_params):
        lo, hi = np.percentile(all_flat[:, j], [0.5, 99.5])
        margin = 0.05 * (hi - lo)
        ranges.append((lo - margin, hi + margin))

    # ── Fill panels ──
    for row in range(n_params):
        for col in range(n_params):
            ax = axes[row, col]

            if col > row:
                # upper triangle: hide
                ax.set_visible(False)
                continue

            if col == row:
                # ── diagonal: 1D histograms ──
                for label, samples in flat.items():
                    ax.hist(samples[:, col], bins=40, density=True,
                            alpha=0.45, color=colors[label], edgecolor='none')
                    # 68% & 95% quantile lines
                    qs = np.percentile(samples[:, col], [2.5, 16, 50, 84, 97.5])
                    for q in [qs[0], qs[4]]:
                        ax.axvline(q, color=colors[label], lw=0.6, ls='--', alpha=0.5)
                    for q in [qs[1], qs[3]]:
                        ax.axvline(q, color=colors[label], lw=0.8, ls=':', alpha=0.7)
                    ax.axvline(qs[2], color=colors[label], lw=1.0, alpha=0.8)
                ax.set_xlim(ranges[col])
                ax.set_yticks([])

            else:
                # ── lower triangle: 2D density ──
                for label, samples in flat.items():
                    x = samples[:, col]
                    y = samples[:, row]
                    try:
                        kde = gaussian_kde(np.vstack([x, y]))
                        xi = np.linspace(ranges[col][0], ranges[col][1], 80)
                        yi = np.linspace(ranges[row][0], ranges[row][1], 80)
                        Xi, Yi = np.meshgrid(xi, yi)
                        Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
                        levels = [0.393, 0.865]  # ~68% and ~95% for 2D Gaussian
                        levels = [np.max(Zi) * f for f in [0.05, 0.25, 0.5, 0.8]]
                        ax.contour(Xi, Yi, Zi, levels=levels[1:],
                                   colors=colors[label], linewidths=0.8, alpha=0.7)
                        ax.contourf(Xi, Yi, Zi, levels=[levels[0], levels[-1]],
                                    colors=[colors[label]], alpha=0.08)
                    except (np.linalg.LinAlgError, ValueError):
                        ax.scatter(x[::max(1, len(x)//2000)], y[::max(1, len(x)//2000)],
                                   s=1, alpha=0.15, color=colors[label])

                ax.set_xlim(ranges[col])
                ax.set_ylim(ranges[row])

            # ── Tick labels ──
            if row == n_params - 1:
                ax.set_xlabel(_LABELS[col], fontsize=9)
                ax.tick_params(axis='x', labelsize=7)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis='x', labelsize=7)
            if col == 0:
                ax.set_ylabel(_LABELS[row], fontsize=9)
                ax.tick_params(axis='y', labelsize=7)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis='y', labelsize=7)

            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))

    legend_patches = [plt.Line2D([0], [0], color=colors[label], lw=2, label=label)
                      for label in flat]
    fig.legend(handles=legend_patches, loc='upper right',
               bbox_to_anchor=(0.98, 0.98), fontsize=11,
               framealpha=0.9, edgecolor='#cccccc')

    plt.tight_layout(rect=[0, 0, 1, 0.96] if title else [0, 0, 1, 1])
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved corner plot to {outpath}')


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python plot_corner.py chain1.npz [chain2.npz]')
        sys.exit(1)

    chains_dict = {}
    labels = iter(['NH', 'IH'])
    for path in sys.argv[1:]:
        ch, diag = load_chains(path)
        ch_display = theta_to_display(ch)
        label = next(labels)
        print(f'{label}: {ch.shape[0]} chains × {ch.shape[1]} samples, '
              f'accept ~{float(diag.get("accept_rate", np.nan)):.2f}')
        chains_dict[label] = ch_display

    title = 'Posterior distributions — NH Asimov data' if len(chains_dict) == 1 else \
            'Posterior distributions — NH vs IH'
    out = 'posterior_corner.png'
    corner_plot(chains_dict, out, title=title)
