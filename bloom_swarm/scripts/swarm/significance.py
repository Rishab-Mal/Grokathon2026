#!/usr/bin/env python3
"""
Statistical significance testing for forecast comparisons.

Compares primary method against baselines and individual agents using:
- Diebold-Mariano test with Harvey-Leybourne-Newbold small sample correction
- Moving block bootstrap
- Sign flip permutation test
- Holm correction for multiple comparisons
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .core import CONFIG, brier, set_seed

BOOTSTRAP_N = CONFIG["bootstrap_n"]
BLOCK_LENGTH = CONFIG["block_length"]
PERMUTATION_N = CONFIG["permutation_n"]


def diebold_mariano_hln(
    d: np.ndarray,
    h: int = 1,
) -> Tuple[float, float]:
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold small sample correction.

    Args:
        d: loss differential series (d_t = loss_1_t - loss_2_t)
        h: forecast horizon (1 for one-step ahead)

    Returns:
        (test_statistic, p_value)

    Negative statistic means method 1 is better (lower loss).
    """
    n = len(d)
    if n < 3:
        return (0.0, 1.0)

    d_bar = np.mean(d)

    # Autocovariance at lag 0
    gamma_0 = np.var(d, ddof=1)

    # Long-run variance estimate (Newey-West style)
    # For h-step ahead, include autocovariances up to lag h-1
    lrv = gamma_0
    for k in range(1, min(h, n - 1)):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        lrv += 2 * gamma_k

    if lrv <= 0:
        lrv = gamma_0  # Fallback

    # DM statistic
    dm_stat = d_bar / np.sqrt(lrv / n)

    # HLN correction: multiply by sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm_stat * correction

    # Two-sided p-value using normal approximation
    # (t-distribution would be more accurate for small n)
    from math import erf
    p_value = 2 * (1 - 0.5 * (1 + erf(abs(dm_hln) / np.sqrt(2))))

    return (float(dm_hln), float(p_value))


def moving_block_bootstrap(
    d: np.ndarray,
    block_length: int = BLOCK_LENGTH,
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = CONFIG["seed"],
) -> Dict[str, float]:
    """
    Moving block bootstrap for mean loss differential.

    Uses overlapping blocks to preserve autocorrelation structure.

    Returns:
        mean: bootstrap mean
        ci90: 90% confidence interval
        win_rate: fraction of bootstraps where primary wins (d < 0)
    """
    np.random.seed(seed)

    n = len(d)
    if n < block_length:
        block_length = max(1, n)

    # Number of blocks needed to cover n observations
    n_blocks = int(np.ceil(n / block_length))

    # Possible starting positions for blocks
    n_starts = n - block_length + 1
    if n_starts < 1:
        n_starts = 1

    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Sample block start positions
        starts = np.random.randint(0, n_starts, size=n_blocks)

        # Construct bootstrap sample
        sample = []
        for start in starts:
            sample.extend(d[start : start + block_length])

        # Truncate to original length
        sample = np.array(sample[:n])
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)

    # Confidence interval
    ci_lo = np.percentile(bootstrap_means, 5)
    ci_hi = np.percentile(bootstrap_means, 95)

    # Win rate (negative d means primary is better)
    win_rate = np.mean(bootstrap_means < 0)

    return {
        "mean": float(np.mean(bootstrap_means)),
        "ci90": [float(ci_lo), float(ci_hi)],
        "win_rate": float(win_rate),
    }


def sign_flip_permutation(
    d: np.ndarray,
    n_permutations: int = PERMUTATION_N,
    seed: int = CONFIG["seed"],
) -> float:
    """
    Sign flip permutation test for mean loss differential.

    Under the null hypothesis of equal performance, the signs of d are
    exchangeable. We count how often the permuted mean is as extreme as observed.

    Returns:
        p_value: two-sided p-value
    """
    np.random.seed(seed)

    n = len(d)
    if n == 0:
        return 1.0

    observed_mean = np.mean(d)

    count_extreme = 0
    for _ in range(n_permutations):
        # Randomly flip signs
        signs = np.random.choice([-1, 1], size=n)
        perm_mean = np.mean(d * signs)

        if abs(perm_mean) >= abs(observed_mean):
            count_extreme += 1

    # Add 1 to both numerator and denominator for conservative estimate
    p_value = (count_extreme + 1) / (n_permutations + 1)
    return float(p_value)


def holm_correction(p_values: List[float]) -> List[float]:
    """
    Holm-Bonferroni correction for multiple comparisons.

    More powerful than Bonferroni while still controlling FWER.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort indices by p-value
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]

    # Compute corrected p-values
    corrected = np.zeros(m)
    for i in range(m):
        corrected[sorted_idx[i]] = min(1.0, sorted_p[i] * (m - i))

    # Enforce monotonicity: corrected[i] >= corrected[i-1]
    for i in range(1, m):
        k = sorted_idx[i]
        k_prev = sorted_idx[i - 1]
        if corrected[k] < corrected[k_prev]:
            corrected[k] = corrected[k_prev]

    return list(corrected)


def get_verdict(
    d_mean: float,
    perm_p: float,
    alpha: float = 0.10,
) -> str:
    """
    Generate plain English verdict.

    Args:
        d_mean: mean loss differential (negative = primary better)
        perm_p: permutation test p-value
        alpha: significance level
    """
    if perm_p < alpha:
        if d_mean < 0:
            return "significantly better"
        else:
            return "significantly worse"
    else:
        if d_mean < 0:
            return "better but not significant"
        elif d_mean > 0:
            return "worse but not significant"
        else:
            return "no clear difference"


def compare_to_baselines(
    primary_preds: np.ndarray,
    baseline_preds: Dict[str, np.ndarray],
    y: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Compare primary method to each baseline.

    Returns list of comparison results.
    """
    results = []
    all_perm_p = []

    # First pass: compute all comparisons
    comparisons = []
    for name, preds in baseline_preds.items():
        # Loss differential: positive means primary is worse
        d = brier(primary_preds, y) - brier(preds, y)

        dm_stat, dm_p = diebold_mariano_hln(d)
        boot = moving_block_bootstrap(d)
        perm_p = sign_flip_permutation(d)

        comparisons.append({
            "vs": name,
            "dm_stat": dm_stat,
            "dm_p": dm_p,
            "boot_mean": boot["mean"],
            "boot_ci90": boot["ci90"],
            "boot_win_rate": boot["win_rate"],
            "perm_p": perm_p,
            "d_mean": float(np.mean(d)),
        })
        all_perm_p.append(perm_p)

    # Apply Holm correction
    holm_p = holm_correction(all_perm_p)

    # Second pass: add Holm p-values and verdicts
    for i, comp in enumerate(comparisons):
        comp["holm_p"] = holm_p[i]
        comp["verdict"] = get_verdict(comp["d_mean"], holm_p[i])
        results.append(comp)

    return results


def print_significance_table(comparisons: List[Dict[str, Any]]):
    """Print formatted significance table."""
    print("\n" + "=" * 100)
    print("SIGNIFICANCE TESTING (Primary vs Baselines)")
    print("=" * 100)
    print()
    print(f"{'vs':<25} {'DM stat':<10} {'DM p':<10} {'Boot mean':<12} {'Boot 90% CI':<20} {'Win%':<8} {'Perm p':<10} {'Holm p':<10} {'Verdict':<25}")
    print("-" * 140)

    for c in comparisons:
        ci_str = f"[{c['boot_ci90'][0]:.4f}, {c['boot_ci90'][1]:.4f}]"
        print(f"{c['vs']:<25} {c['dm_stat']:<10.4f} {c['dm_p']:<10.4f} {c['boot_mean']:<12.4f} {ci_str:<20} {c['boot_win_rate']*100:<8.1f} {c['perm_p']:<10.4f} {c['holm_p']:<10.4f} {c['verdict']:<25}")

    print("=" * 100)
    print("\nNote: With ~16 weeks, most comparisons will not reach statistical significance.")
    print("The bootstrap win rate shows the probability that the primary method outperforms.")
