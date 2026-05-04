"""Corner-plot of MCMC posterior samples.

Loads chains from .npz files and draws a publication-quality corner plot:

  - diagonal: 1D histograms with 68/95% quantile lines
  - lower triangle: 2D density (Gaussian KDE on a grid)
  - upper-right: metadata panel (active pulls, grid, precision)

Parameters are displayed in physics units (|Δm²₃₂| in eV², sin²θ for angles).

Usage:
    cd jax_barger
    PYTHONPATH=../build/pybind:.. .venv/bin/python plot_corner.py \\
        --nh hmc_chains_nh_fine2k.npz --ih hmc_chains_ih_fine2k.npz \\
        --pull "full pulls (DM2, Dm2, T23, T13, DCP, T12)" \\
        --grid "fine 200E×120cosθ" --precision fp32
"""

import sys, argparse
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


def load_chains(path):
    """Load chains from .npz: return (n_chains, n_samples, 6) in θ-space."""
    data = np.load(path, allow_pickle=True)
    return data['chains']


def theta_to_display(chains):
    """Convert θ-space → display units: |DM2|, Dm2, sin²θₓ."""
    out = chains.copy()
    out[..., 0] = np.abs(out[..., 0])
    for i in [2, 3, 5]:
        out[..., i] = np.sin(out[..., i]) ** 2
    return out


def corner_plot(chains_dict, outbase, title=None, pull_info=None,
                grid_info=None, prec_info=None):
    """Draw a corner plot.  Saves {outbase}.png, {outbase}.pdf, {outbase}.eps.

    Args:
        chains_dict: {label: (n_chain, n_samples, 6)}
        outbase:     output basename (without extension)
        title:       figure super-title
        pull_info:   string describing active pull terms
        grid_info:   string describing evaluation grid
        prec_info:   string describing floating-point precision
    """
    n_params = 6
    flat = {lbl: ch.reshape(-1, n_params) for lbl, ch in chains_dict.items()}

    colors_list = ['#1f77b4', '#d62728']
    colors = {lbl: colors_list[i % 2] for i, lbl in enumerate(flat)}

    fig, axes = plt.subplots(n_params, n_params, figsize=(14, 12))
    if title:
        fig.suptitle(title, fontsize=14, y=0.975)

    # ── Global x-range per parameter ──
    all_flat = np.concatenate(list(flat.values()), axis=0)
    ranges = []
    for j in range(n_params):
        lo, hi = np.percentile(all_flat[:, j], [0.5, 99.5])
        m = 0.05 * (hi - lo)
        ranges.append((lo - m, hi + m))

    # ── Fill panels ──
    for row in range(n_params):
        for col in range(n_params):
            ax = axes[row, col]
            if col > row:
                ax.set_visible(False)
                continue

            if col == row:
                for lbl, samples in flat.items():
                    ax.hist(samples[:, col], bins=40, density=True,
                            alpha=0.45, color=colors[lbl], edgecolor='none')
                    qs = np.percentile(samples[:, col], [2.5, 16, 50, 84, 97.5])
                    for q in [qs[0], qs[4]]:
                        ax.axvline(q, color=colors[lbl], lw=0.5, ls='--', alpha=0.4)
                    for q in [qs[1], qs[3]]:
                        ax.axvline(q, color=colors[lbl], lw=0.6, ls=':', alpha=0.6)
                    ax.axvline(qs[2], color=colors[lbl], lw=1.0)
                ax.set_xlim(ranges[col])
                ax.set_yticks([])
            else:
                for lbl, samples in flat.items():
                    x, y = samples[:, col], samples[:, row]
                    try:
                        kde = gaussian_kde(np.vstack([x, y]))
                        xi = np.linspace(ranges[col][0], ranges[col][1], 60)
                        yi = np.linspace(ranges[row][0], ranges[row][1], 60)
                        Xi, Yi = np.meshgrid(xi, yi)
                        Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
                        lvls = [np.max(Zi) * f for f in [0.05, 0.25, 0.5, 0.8]]
                        ax.contour(Xi, Yi, Zi, levels=lvls[1:],
                                   colors=colors[lbl], linewidths=0.7, alpha=0.6)
                    except (np.linalg.LinAlgError, ValueError):
                        ss = max(1, len(x) // 500)
                        ax.scatter(x[::ss], y[::ss], s=0.5, alpha=0.1,
                                   color=colors[lbl])
                ax.set_xlim(ranges[col]); ax.set_ylim(ranges[row])

            # Tick labels
            if row == n_params - 1:
                ax.set_xlabel(_LABELS[col], fontsize=9)
                ax.tick_params(axis='x', labelsize=7)
            else:
                ax.set_xticklabels([]); ax.tick_params(axis='x', labelsize=7)
            if col == 0 and row > 0:
                ax.set_ylabel(_LABELS[row], fontsize=9)
                ax.tick_params(axis='y', labelsize=7)
            else:
                ax.set_yticklabels([]); ax.tick_params(axis='y', labelsize=7)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))

    # ── Legend ──
    legend_patches = [plt.Line2D([0], [0], color=colors[lbl], lw=2, label=lbl)
                      for lbl in flat]
    fig.legend(handles=legend_patches, loc='upper right',
               bbox_to_anchor=(0.98, 0.98), fontsize=11,
               framealpha=0.9, edgecolor='#cccccc')

    # ── Info panel (upper-right blank area) ──
    if pull_info or grid_info or prec_info:
        lines = ['Observational errors not considered']
        if pull_info:
            lines.append(f'Pull terms: {pull_info}')
        if grid_info:
            lines.append(f'Grid: {grid_info}')
        if prec_info:
            lines.append(f'Precision: {prec_info}')
        text = '\n'.join(lines)
        fig.text(0.82, 0.88, text, fontsize=8, va='top', ha='left',
                 family='monospace',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                           edgecolor='#aaaaaa', alpha=0.85))

    plt.tight_layout(rect=[0, 0, 1, 0.96] if title else [0, 0, 1, 1])

    # ── Save multiple formats ──
    for ext in ['png', 'pdf', 'eps']:
        path = f'{outbase}.{ext}'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved {path}')

    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Corner plot of MCMC posterior samples')
    parser.add_argument('--nh', required=True, help='NH chain .npz file')
    parser.add_argument('--ih', default=None, help='IH chain .npz file (optional)')
    parser.add_argument('--basename', default='posterior_corner', help='Output basename')
    parser.add_argument('--pull', default=None,
                        help='Description of active pull terms')
    parser.add_argument('--grid', default=None, help='Grid description')
    parser.add_argument('--precision', default=None, help='Precision info')
    args = parser.parse_args()

    chains_dict = {}
    ch = load_chains(args.nh)
    chains_dict['NH'] = theta_to_display(ch)
    print(f'NH: {ch.shape[0]} chains × {ch.shape[1]} samples')

    if args.ih:
        ch = load_chains(args.ih)
        chains_dict['IH'] = theta_to_display(ch)
        print(f'IH: {ch.shape[0]} chains × {ch.shape[1]} samples')

    title = 'Posterior distributions — NH Asimov data' if len(chains_dict) == 1 \
            else 'Posterior distributions — NH vs IH'

    corner_plot(chains_dict, args.basename, title=title,
                pull_info=args.pull, grid_info=args.grid,
                prec_info=args.precision)
