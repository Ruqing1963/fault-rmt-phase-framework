#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  Fault RMT Phase Framework — Main Analysis Pipeline
  ─────────────────────────────────────────────────
  Forward-modeling framework for RMT phase transitions in fault rhythms.

  Author: Ruqing Chen, GUT Geoservice Inc., Montreal
  Repository: https://github.com/Ruqing1963/fault-rmt-phase-framework

  Usage:
    python analysis.py [--n-bootstrap 5000] [--output-dir ../results]
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
# Theoretical distributions
# ═══════════════════════════════════════════════════════════════════════════

def wigner_gue(s):
    return (32.0/np.pi**2)*s**2*np.exp(-4.0*s**2/np.pi)

def wigner_goe(s):
    return (np.pi/2.0)*s*np.exp(-np.pi*s**2/4.0)

def poisson_pdf(s):
    return np.exp(-s)

def make_cdf(pdf_func, s_max=8.0, n=10000):
    s = np.linspace(0, s_max, n)
    c = cumulative_trapezoid(pdf_func(s), s, initial=0)
    c /= c[-1]
    return interp1d(s, c, bounds_error=False, fill_value=(0, 1))

POI_CDF = lambda x: 1 - np.exp(-x)


# ═══════════════════════════════════════════════════════════════════════════
# Catalog generation (Wigner-Dyson process with tunable beta)
# ═══════════════════════════════════════════════════════════════════════════

def gen_catalog(n, rate, b_value, mc, beta, seed):
    """
    Generate synthetic earthquake catalog with prescribed level repulsion beta.

    beta=0: Poisson; beta=1: GOE; beta=2: GUE.
    Non-integer beta uses linear blending of inter-event times between
    the bracketing integer-beta processes.
    """
    rng = np.random.default_rng(seed)
    mi = 1.0 / rate  # mean interval (years)

    def sample_wd(target_beta, m):
        """Sample m inter-event times from Wigner surmise of given integer beta."""
        if target_beta == 0:
            return rng.exponential(mi, m)
        if target_beta == 1:
            c = np.pi/4
            pdf = lambda s: (np.pi/2)*s*np.exp(-np.pi*s**2/4)
            smax_pdf = pdf(np.sqrt(1/(2*c)))
        else:  # beta == 2
            c = 4.0/np.pi
            pdf = lambda s: (32/np.pi**2)*s**2*np.exp(-4*s**2/np.pi)
            smax_pdf = pdf(np.sqrt(2/(2*c)))
        out = []
        while len(out) < m:
            s = rng.exponential(1.6)
            if rng.random() < pdf(s) / (smax_pdf * 1.15):
                out.append(s * mi)
        return np.array(out[:m])

    if beta <= 0:
        iv = sample_wd(0, n)
    elif beta < 1:
        lo, hi = sample_wd(0, n), sample_wd(1, n)
        iv = (1 - beta) * lo + beta * hi
    elif beta == 1:
        iv = sample_wd(1, n)
    elif beta < 2:
        w = beta - 1
        lo, hi = sample_wd(1, n), sample_wd(2, n)
        iv = (1 - w) * lo + w * hi
    else:
        iv = sample_wd(2, n)

    mags = mc + rng.exponential(1.0/(b_value*np.log(10)), n)
    times = np.cumsum(iv)
    return times, mags, iv


# ═══════════════════════════════════════════════════════════════════════════
# Declustering (Reasenberg-style window)
# ═══════════════════════════════════════════════════════════════════════════

def decluster(times, tau0_days=3.0, max_days=60.0):
    """Window-based declustering. Returns mainshock times and mask."""
    if len(times) < 3:
        return times, np.ones(len(times), dtype=bool)
    max_w = max_days / 365.25
    is_main = np.ones(len(times), dtype=bool)
    for i in range(len(times)):
        if not is_main[i]:
            continue
        for j in range(i+1, len(times)):
            dt = times[j] - times[i]
            if dt > max_w:
                break
            window = (tau0_days / 365.25) * (1 + 0.5 * np.log10(max(dt*365.25, 0.1)))
            if dt < window:
                is_main[j] = False
    return times[is_main], is_main


# ═══════════════════════════════════════════════════════════════════════════
# Statistical tools
# ═══════════════════════════════════════════════════════════════════════════

def spacing_ratio(sp):
    if len(sp) < 3:
        return np.nan, np.nan
    r = np.minimum(sp[:-1], sp[1:]) / np.maximum(sp[:-1], sp[1:])
    return np.mean(r), np.std(r) / np.sqrt(len(r))

def anderson_darling(data, cdf_func):
    n = len(data)
    x = np.sort(data)
    F = np.clip(cdf_func(x), 1e-15, 1-1e-15)
    i = np.arange(1, n+1)
    return -n - np.sum((2*i-1)*(np.log(F) + np.log(1-F[::-1])))/n

def beta_from_r(r):
    """Piecewise-linear beta estimate from spacing ratio."""
    if np.isnan(r):
        return np.nan
    if r <= 0.386:
        return 0.0
    elif r <= 0.536:
        return (r - 0.386)/(0.536 - 0.386)
    elif r <= 0.603:
        return 1.0 + (r - 0.536)/(0.603 - 0.536)
    else:
        return np.clip(2.0 + (r - 0.603)/0.1, 0, 3)


# ═══════════════════════════════════════════════════════════════════════════
# Power analysis (NEW — addresses reviewer point 6)
# ═══════════════════════════════════════════════════════════════════════════

def poisson_power_analysis(n, n_trials=2000, alpha=0.05, seed=7):
    """
    Estimate the statistical power of the KS test to reject Poisson
    when the true process is GOE, at sample size n.

    Returns the fraction of GOE-generated samples (size n) for which
    the Poisson hypothesis is correctly rejected at level alpha.
    """
    rng = np.random.default_rng(seed)
    rejections = 0
    for _ in range(n_trials):
        # Generate GOE sample of size n
        c = np.pi/4
        pdf = lambda s: (np.pi/2)*s*np.exp(-np.pi*s**2/4)
        smax = pdf(np.sqrt(1/(2*c)))
        out = []
        while len(out) < n:
            s = rng.exponential(1.6)
            if rng.random() < pdf(s)/(smax*1.15):
                out.append(s)
        sample = np.array(out[:n])
        sample = sample / np.mean(sample)
        _, p = stats.kstest(sample, POI_CDF)
        if p < alpha:
            rejections += 1
    return rejections / n_trials


def beta_recovery_curve(beta_true_grid, n, rate=4.0, b=0.9, mc=3.5,
                        n_realizations=200, seed=11):
    """
    For each true beta, generate many catalogs and record the distribution
    of recovered beta_hat. Quantifies the estimator bias (reviewer point 1).
    """
    rng = np.random.default_rng(seed)
    results = []
    for bt in beta_true_grid:
        recovered = []
        for k in range(n_realizations):
            _, _, iv = gen_catalog(n, rate, b, mc, bt, seed=rng.integers(1, 2**31))
            s = iv / np.mean(iv)
            r, _ = spacing_ratio(s)
            recovered.append(beta_from_r(r))
        recovered = np.array(recovered)
        results.append({
            'beta_true': bt,
            'beta_hat_mean': np.mean(recovered),
            'beta_hat_median': np.median(recovered),
            'beta_hat_lo': np.percentile(recovered, 16),
            'beta_hat_hi': np.percentile(recovered, 84),
        })
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# Fault zone definitions
# ═══════════════════════════════════════════════════════════════════════════

def load_fault_params(csv_path):
    return pd.read_csv(csv_path)


def analyze_fault(row, n_bootstrap=5000):
    """Run full RMT analysis on one fault zone."""
    times, mags, iv = gen_catalog(
        int(row['n_events']), row['rate_per_yr'], row['b_value'],
        row['mc'], row['beta_in'], seed=abs(hash(row['key'])) % (2**31)
    )
    # Decluster
    times_dc, mask = decluster(times)
    iv_dc = np.diff(times_dc)
    if len(iv_dc) < 10:
        return None
    s = iv_dc / np.mean(iv_dc)

    gue_cdf = make_cdf(wigner_gue)
    goe_cdf = make_cdf(wigner_goe)

    ks_p, pp = stats.kstest(s, POI_CDF)
    ks_o, po = stats.kstest(s, goe_cdf)
    ks_u, pu = stats.kstest(s, gue_cdf)

    ad_p = anderson_darling(s, POI_CDF)
    ad_o = anderson_darling(s, goe_cdf)
    ad_u = anderson_darling(s, gue_cdf)

    r, r_err = spacing_ratio(s)
    beta_hat = beta_from_r(r)

    rng = np.random.default_rng(42)
    boot_r = np.array([spacing_ratio(rng.choice(s, len(s), True))[0]
                       for _ in range(n_bootstrap)])
    r_ci = np.percentile(boot_r, [2.5, 97.5])

    ks_scores = {'Poisson': ks_p, 'GOE': ks_o, 'GUE': ks_u}
    best = min(ks_scores, key=ks_scores.get)

    # Power to reject Poisson at this n (if true process were GOE)
    power = poisson_power_analysis(len(s), n_trials=1000)

    return {
        'key': row['key'], 'name': row['name'], 'group': row['group'],
        'complexity': row['complexity'], 'fluid': row['fluid'],
        'beta_in': row['beta_in'], 'beta_hat': beta_hat,
        'n_events': len(times_dc), 'n_intervals': len(s),
        's_norm': s.tolist(),
        'r': r, 'r_err': r_err, 'r_ci_lo': r_ci[0], 'r_ci_hi': r_ci[1],
        'var_s': float(np.var(s)),
        'ks_poi': ks_p, 'p_poi': pp,
        'ks_goe': ks_o, 'p_goe': po,
        'ks_gue': ks_u, 'p_gue': pu,
        'ad_poi': ad_p, 'ad_goe': ad_o, 'ad_gue': ad_u,
        'best_fit': best,
        'poisson_power': power,
    }


def main(args):
    print("=" * 72)
    print("  Fault RMT Phase Framework — Analysis Pipeline")
    print("  Author: Ruqing Chen, GUT Geoservice Inc., Montreal")
    print("=" * 72)

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.dirname(__file__)
    params = load_fault_params(os.path.join(base, '..', 'data', 'fault_zones.csv'))

    results = []
    for _, row in params.iterrows():
        print(f"\n  Analyzing: {row['name']} [{row['group']}]")
        res = analyze_fault(row, args.n_bootstrap)
        if res is None:
            print("    Insufficient data, skipped.")
            continue
        results.append(res)
        print(f"    n={res['n_intervals']}  beta_in={res['beta_in']:.2f}  "
              f"beta_hat={res['beta_hat']:.2f}  <r>={res['r']:.4f}")
        print(f"    KS: Poi p={res['p_poi']:.4f} | GOE p={res['p_goe']:.4f} | "
              f"GUE p={res['p_gue']:.4f}  -> best={res['best_fit']}")
        print(f"    Poisson-rejection power at n={res['n_intervals']}: "
              f"{res['poisson_power']:.2f}")

    # Save per-fault results
    results_sorted = sorted(results, key=lambda x: x['beta_hat'], reverse=True)
    summary = pd.DataFrame([{k: v for k, v in r.items() if k != 's_norm'}
                            for r in results_sorted])
    summary.to_csv(os.path.join(args.output_dir, 'fault_results.csv'), index=False)
    with open(os.path.join(args.output_dir, 'fault_results.json'), 'w') as f:
        json.dump(results_sorted, f, indent=2)

    # Save spacings
    spac = {r['key']: r['s_norm'] for r in results}
    with open(os.path.join(args.output_dir, 'spacings.json'), 'w') as f:
        json.dump(spac, f)

    # ── Power analysis curve ──
    print("\n  Running power analysis across sample sizes...")
    n_grid = [20, 40, 60, 80, 100, 120, 150, 200, 300, 500]
    power_curve = pd.DataFrame({
        'n': n_grid,
        'power_reject_poisson_if_GOE': [poisson_power_analysis(n, 800) for n in n_grid],
    })
    power_curve.to_csv(os.path.join(args.output_dir, 'power_curve.csv'), index=False)
    print("    " + "  ".join(f"n={n}:{p:.2f}" for n, p
          in zip(power_curve['n'], power_curve['power_reject_poisson_if_GOE'])))

    # ── Beta recovery curve ──
    print("\n  Running beta-recovery diagnostic...")
    beta_grid = np.linspace(0, 2.0, 11)
    recovery = beta_recovery_curve(beta_grid, n=120, n_realizations=150)
    recovery.to_csv(os.path.join(args.output_dir, 'beta_recovery.csv'), index=False)
    print("    beta_true -> beta_hat_mean:")
    for _, r in recovery.iterrows():
        print(f"      {r['beta_true']:.2f} -> {r['beta_hat_mean']:.2f} "
              f"[{r['beta_hat_lo']:.2f}, {r['beta_hat_hi']:.2f}]")

    print(f"\n  Results saved to {os.path.abspath(args.output_dir)}/")
    print("  Done.")
    return results_sorted


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-bootstrap', type=int, default=5000)
    ap.add_argument('--output-dir', type=str, default='../results')
    main(ap.parse_args())
