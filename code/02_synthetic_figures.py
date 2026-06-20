#!/usr/bin/env python3
"""Generate publication figures for Fault RMT Phase Framework."""

import argparse
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import analysis as A

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'text.color': '#1a1a1a', 'axes.labelcolor': '#1a1a1a',
    'xtick.color': '#333', 'ytick.color': '#333',
    'axes.edgecolor': '#555', 'grid.color': '#ddd',
    'font.family': 'serif', 'font.size': 11,
})


def main(args):
    base = os.path.dirname(__file__)
    rdir = os.path.join(base, '..', 'results')
    os.makedirs(args.output_dir, exist_ok=True)
    ext = args.format

    summary = pd.read_csv(os.path.join(rdir, 'fault_results.csv'))
    fparams = pd.read_csv(os.path.join(base, '..', 'data', 'fault_zones.csv'))
    summary = summary.merge(fparams[['key', 'color', 'marker']], on='key', how='left')
    with open(os.path.join(rdir, 'spacings.json')) as f:
        spacings = json.load(f)
    power = pd.read_csv(os.path.join(rdir, 'power_curve.csv'))
    recovery = pd.read_csv(os.path.join(rdir, 'beta_recovery.csv'))

    s_th = np.linspace(0, 4.0, 500)
    summary = summary.sort_values('beta_hat', ascending=False).reset_index(drop=True)

    # ═══ Figure 1: 2x4 KDE panels ═══
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for idx, row in summary.iterrows():
        ax = axes[idx//4, idx%4]
        s = np.array(spacings[row['key']])
        bw = max(0.2, 0.8/np.sqrt(len(s)))
        kde = gaussian_kde(s, bw_method=bw)
        sk = np.linspace(0, 4, 300)
        ax.fill_between(sk, kde(sk), alpha=0.25, color=row['color'])
        ax.plot(sk, kde(sk), color=row['color'], lw=2.5, label='Data KDE')
        ax.plot(s_th, A.wigner_gue(s_th), '#b2182b', lw=1.5, ls='--', alpha=0.6, label='GUE')
        ax.plot(s_th, A.wigner_goe(s_th), '#ff7f00', lw=1.5, ls=':', alpha=0.6, label='GOE')
        ax.plot(s_th, A.poisson_pdf(s_th), '#4daf4a', lw=1.5, ls='-.', alpha=0.6, label='Poisson')
        ax.set_title(f"{row['name']}\n$\\hat{{\\beta}}$={row['beta_hat']:.2f}, "
                     f"$\\langle r\\rangle$={row['r']:.3f}, $n$={row['n_intervals']}", fontsize=9)
        ax.set_xlim(0, 3.5); ax.set_ylim(0, 1.4)
        ax.set_xlabel('$s$', fontsize=9); ax.set_ylabel('$p(s)$', fontsize=9)
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')
    fig.suptitle('Normalized inter-event spacing distributions (eight fault zones)',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(f'{args.output_dir}/fig1_kde_panels.{ext}', dpi=300, bbox_inches='tight')
    plt.close()

    # ═══ Figure 2: Spacing ratio bar ═══
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(summary))
    ax.barh(y, summary['r'], xerr=summary['r_err'], height=0.55,
            color=summary['color'], alpha=0.75, edgecolor='#555', capsize=3)
    ax.axvline(0.386, color='#4daf4a', lw=2.5, ls='--', label='Poisson (0.386)')
    ax.axvline(0.536, color='#ff7f00', lw=2.5, ls=':', label='GOE (0.536)')
    ax.axvline(0.603, color='#b2182b', lw=2.5, label='GUE (0.603)')
    ax.set_yticks(y); ax.set_yticklabels(summary['name'], fontsize=10)
    ax.set_xlabel('$\\langle r \\rangle$', fontsize=12)
    ax.set_xlim(0.3, 0.75)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.2, axis='x'); ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(f'{args.output_dir}/fig2_spacing_ratio.{ext}', dpi=300)
    plt.close()

    # ═══ Figure 3: Phase diagram ═══
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in summary.iterrows():
        x = row['complexity'] + row['fluid']*0.3
        yv = row['beta_hat']
        ax.scatter(x, yv, c=row['color'], s=180, marker=row['marker'],
                   edgecolors='k', linewidths=0.8, zorder=5)
        ax.annotate(row['name'], (x, yv), textcoords="offset points", xytext=(10, 7),
                    fontsize=9, arrowprops=dict(arrowstyle='->', lw=0.6))
    ax.axhspan(-0.1, 0.5, alpha=0.06, color='#4daf4a')
    ax.axhspan(0.5, 1.5, alpha=0.06, color='#ff7f00')
    ax.axhspan(1.5, 3.1, alpha=0.06, color='#b2182b')
    ax.text(0.5, 0.15, 'Poisson regime ($\\hat{\\beta} < 0.5$)', fontsize=10, style='italic', color='#4daf4a')
    ax.text(0.5, 0.9, 'GOE regime', fontsize=10, style='italic', color='#cc7000')
    ax.text(0.5, 2.2, 'GUE regime ($\\hat{\\beta} > 1.5$)', fontsize=10, style='italic', color='#b2182b')
    ax.axhline(0.5, color='#aaa', ls=':', lw=0.8)
    ax.axhline(1.5, color='#aaa', ls=':', lw=0.8)
    ax.set_xlabel('Geological complexity index ($C + 0.3F$)', fontsize=12)
    ax.set_ylabel('Estimated repulsion parameter $\\hat{\\beta}$', fontsize=12)
    ax.set_xlim(0, 12); ax.set_ylim(-0.1, 3.1)
    legend_el = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#555', markersize=9, label='Strike-slip'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor='#555', markersize=9, label='Subduction'),
        Line2D([0],[0], marker='^', color='w', markerfacecolor='#555', markersize=9, label='Intraplate'),
    ]
    ax.legend(handles=legend_el, fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(f'{args.output_dir}/fig3_phase_diagram.{ext}', dpi=300)
    plt.close()

    # ═══ Figure 4: Beta recovery (CRITICAL — reviewer point 1) ═══
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.fill_between(recovery['beta_true'], recovery['beta_hat_lo'],
                    recovery['beta_hat_hi'], alpha=0.2, color='#2166ac',
                    label='16–84% band')
    ax.plot(recovery['beta_true'], recovery['beta_hat_mean'], 'o-',
            color='#2166ac', lw=2, label='Mean recovered $\\hat{\\beta}$')
    ax.plot([0, 2], [0, 2], 'k--', lw=1.5, alpha=0.5, label='Perfect recovery')
    # Overlay actual fault recoveries
    for _, row in summary.iterrows():
        ax.scatter(row['beta_in'], row['beta_hat'], c=row['color'],
                   s=90, marker=row['marker'], edgecolors='k', linewidths=0.8, zorder=6)
    ax.set_xlabel('Input $\\beta$ (physical prior)', fontsize=12)
    ax.set_ylabel('Recovered $\\hat{\\beta}$', fontsize=12)
    ax.set_xlim(-0.05, 2.05); ax.set_ylim(-0.05, 3.1)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.2)
    ax.set_title('Estimator bias: $\\hat{\\beta}$ systematically overestimates low $\\beta$',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{args.output_dir}/fig4_beta_recovery.{ext}', dpi=300)
    plt.close()

    # ═══ Figure 5: Power analysis (NEW — reviewer point 6) ═══
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(power['n'], power['power_reject_poisson_if_GOE'], 'o-',
            color='#b2182b', lw=2, markersize=7)
    ax.axhline(0.8, color='#555', ls='--', lw=1, alpha=0.6, label='80% power threshold')
    ax.axhline(1.0, color='#aaa', ls=':', lw=0.8)
    ax.fill_between(power['n'], 0, power['power_reject_poisson_if_GOE'],
                    alpha=0.1, color='#b2182b')
    # Mark the smallest fault n
    n_min = summary['n_intervals'].min()
    ax.axvline(n_min, color='#2166ac', ls='-.', lw=1.5,
               label=f'Smallest fault sample ($n={n_min}$)')
    ax.set_xlabel('Sample size $n$', fontsize=12)
    ax.set_ylabel('Power to reject Poisson (if true process is GOE)', fontsize=11)
    ax.set_xscale('log')
    ax.set_xlim(18, 520); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.2, which='both')
    ax.set_title('Statistical power of the Poisson-rejection test vs. sample size',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{args.output_dir}/fig5_power_analysis.{ext}', dpi=300)
    plt.close()

    print(f"All 5 figures saved to {os.path.abspath(args.output_dir)}/")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default='../figures')
    ap.add_argument('--format', default='pdf', choices=['pdf', 'png', 'svg'])
    main(ap.parse_args())
