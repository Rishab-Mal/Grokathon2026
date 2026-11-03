#!/usr/bin/env python3
"""
Scoring metrics for forecast evaluation.

For every agent and every method under both protocols:
- Brier score, log loss, spherical score
- Brier skill score vs base_rate
- Murphy decomposition into reliability, resolution, uncertainty
- Expected calibration error
- AUC (Mann-Whitney)
- Sharpness (variance of predictions)
- Accuracy at 0.5 threshold
- Calibration table per bin with Wilson intervals
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .core import CONFIG, brier, log_loss, clip


CALIBRATION_BINS = CONFIG["calibration_bins"]


def spherical_score(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Spherical score: y * p / ||p_vec|| + (1-y) * (1-p) / ||p_vec||
    where p_vec = (p, 1-p).

    This is a proper scoring rule that rewards calibration.
    """
    p = np.clip(p, 1e-10, 1 - 1e-10)
    norm = np.sqrt(p**2 + (1 - p)**2)
    return (y * p + (1 - y) * (1 - p)) / norm


def brier_skill_score(pred_brier: float, base_brier: float) -> float:
    """
    Brier Skill Score: 1 - (Brier / Brier_base).

    Positive means better than base rate, 1.0 is perfect.
    """
    if base_brier == 0:
        return 0.0 if pred_brier == 0 else -np.inf
    return 1.0 - (pred_brier / base_brier)


def murphy_decomposition(
    preds: np.ndarray,
    y: np.ndarray,
    n_bins: int = CALIBRATION_BINS,
) -> Dict[str, float]:
    """
    Murphy decomposition of Brier score into:
    - Reliability: how well calibrated (lower is better)
    - Resolution: how much predictions vary from base rate (higher is better)
    - Uncertainty: inherent uncertainty from y variance (constant for dataset)

    Brier = Reliability - Resolution + Uncertainty

    Uses equal-width bins.
    """
    n = len(preds)
    if n == 0:
        return {"reliability": 0.0, "resolution": 0.0, "uncertainty": 0.0, "gap": 0.0}

    # Base rate
    base_rate = y.mean()
    uncertainty = base_rate * (1 - base_rate)

    # Bin edges
    bin_edges = np.linspace(0, 1, n_bins + 1)

    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include right edge for last bin
        if i == n_bins - 1:
            mask = (preds >= lo) & (preds <= hi)
        else:
            mask = (preds >= lo) & (preds < hi)

        n_k = mask.sum()
        if n_k == 0:
            continue

        p_k = preds[mask].mean()  # mean predicted prob in bin
        y_k = y[mask].mean()  # observed rate in bin

        reliability += n_k * (p_k - y_k) ** 2
        resolution += n_k * (y_k - base_rate) ** 2

    reliability /= n
    resolution /= n

    # Verify decomposition
    actual_brier = np.mean(brier(preds, y))
    reconstructed = reliability - resolution + uncertainty
    gap = abs(actual_brier - reconstructed)

    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "gap": float(gap),
    }


def expected_calibration_error(
    preds: np.ndarray,
    y: np.ndarray,
    n_bins: int = CALIBRATION_BINS,
) -> float:
    """
    Expected Calibration Error (ECE).

    ECE = sum_k (n_k / n) * |p_k - y_k|

    where p_k is mean predicted prob in bin k and y_k is observed rate.
    """
    n = len(preds)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (preds >= lo) & (preds <= hi)
        else:
            mask = (preds >= lo) & (preds < hi)

        n_k = mask.sum()
        if n_k == 0:
            continue

        p_k = preds[mask].mean()
        y_k = y[mask].mean()
        ece += (n_k / n) * abs(p_k - y_k)

    return float(ece)


def auc_mann_whitney(preds: np.ndarray, y: np.ndarray) -> Optional[float]:
    """
    Compute AUC using Mann-Whitney U statistic.

    AUC = P(pred_positive > pred_negative)

    Returns None if y has only one class.
    """
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return None

    pos_preds = preds[pos_idx]
    neg_preds = preds[neg_idx]

    # Count pairs where positive > negative
    n_pos = len(pos_preds)
    n_neg = len(neg_preds)

    # Use broadcasting for efficiency
    comparisons = pos_preds[:, np.newaxis] > neg_preds[np.newaxis, :]
    ties = pos_preds[:, np.newaxis] == neg_preds[np.newaxis, :]

    # AUC = (n_greater + 0.5 * n_ties) / (n_pos * n_neg)
    n_greater = comparisons.sum()
    n_ties = ties.sum()

    auc = (n_greater + 0.5 * n_ties) / (n_pos * n_neg)
    return float(auc)


def sharpness(preds: np.ndarray) -> float:
    """
    Sharpness: variance of predictions.

    Higher sharpness means more decisive forecasts.
    """
    if len(preds) == 0:
        return 0.0
    return float(np.var(preds))


def accuracy_at_threshold(preds: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> float:
    """
    Classification accuracy at a given threshold.
    """
    if len(preds) == 0:
        return 0.0
    predicted_class = (preds >= threshold).astype(int)
    return float(np.mean(predicted_class == y))


def wilson_interval(successes: int, total: int, confidence: float = 0.90) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.

    Returns (lower, upper) bounds at the given confidence level.
    """
    if total == 0:
        return (0.0, 1.0)

    # Z-score for given confidence
    # 90% confidence -> z = 1.645
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.645)

    p_hat = successes / total
    denominator = 1 + z**2 / total

    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (float(lower), float(upper))


def calibration_table(
    preds: np.ndarray,
    y: np.ndarray,
    n_bins: int = CALIBRATION_BINS,
) -> List[Dict[str, Any]]:
    """
    Build calibration table with Wilson intervals.

    Returns list of dicts with:
    - bin: range label like "0.0 to 0.2"
    - mean_p: mean predicted probability in bin
    - observed: observed rate in bin
    - count: number of samples in bin
    - ci90: 90% Wilson interval for observed rate
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    table = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (preds >= lo) & (preds <= hi)
        else:
            mask = (preds >= lo) & (preds < hi)

        n_k = mask.sum()
        if n_k == 0:
            mean_p = (lo + hi) / 2
            observed = 0.0
            ci90 = (0.0, 1.0)
        else:
            mean_p = float(preds[mask].mean())
            successes = int(y[mask].sum())
            observed = float(y[mask].mean())
            ci90 = wilson_interval(successes, n_k, 0.90)

        table.append({
            "bin": f"{lo:.1f} to {hi:.1f}",
            "mean_p": round(mean_p, 4),
            "observed": round(observed, 4),
            "count": int(n_k),
            "ci90": [round(ci90[0], 4), round(ci90[1], 4)],
        })

    return table


def compute_all_metrics(
    preds: np.ndarray,
    y: np.ndarray,
    base_rate_brier: float,
) -> Dict[str, Any]:
    """
    Compute all metrics for a single prediction set.

    Returns dict with all metric values.
    """
    pred_brier = float(np.mean(brier(preds, y)))
    pred_log_loss = float(np.mean(log_loss(preds, y)))
    pred_spherical = float(np.mean(spherical_score(preds, y)))

    decomposition = murphy_decomposition(preds, y)

    return {
        "brier": pred_brier,
        "log_loss": pred_log_loss,
        "spherical": pred_spherical,
        "bss": brier_skill_score(pred_brier, base_rate_brier),
        "ece": expected_calibration_error(preds, y),
        "auc": auc_mann_whitney(preds, y),
        "sharpness": sharpness(preds),
        "accuracy": accuracy_at_threshold(preds, y, 0.5),
        "decomposition": decomposition,
        "calibration": calibration_table(preds, y),
    }


def compute_metrics_for_all(
    method_results: Dict[str, Dict[str, Dict]],
    agent_preds: Dict[str, np.ndarray],
    y: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute metrics for all methods and agents.

    Returns:
        {name: {protocol: metrics_dict}}
    """
    # Get base rate brier for BSS
    base_rate_brier_loo = method_results["base_rate"]["loo"]["brier"]
    base_rate_brier_wf = method_results["base_rate"]["wf"]["brier"]

    all_metrics = {}

    # Methods
    for method_name, result in method_results.items():
        all_metrics[method_name] = {
            "loo": compute_all_metrics(result["loo"]["preds"], y, base_rate_brier_loo),
            "wf": compute_all_metrics(result["wf"]["preds"], y, base_rate_brier_wf),
        }

    # Agents
    for agent_name, preds in agent_preds.items():
        valid = ~np.isnan(preds)
        if valid.sum() > 0:
            all_metrics[agent_name] = {
                "agent": compute_all_metrics(preds[valid], y[valid], base_rate_brier_loo),
            }

    return all_metrics


def print_metrics_summary(
    all_metrics: Dict[str, Dict[str, Any]],
    primary_method: str,
):
    """Print a summary of key metrics."""
    print("\n" + "=" * 80)
    print("PHASE 2: METRICS SUMMARY")
    print("=" * 80)

    print("\nDecomposition for primary method (LOO):")
    primary = all_metrics[primary_method]["loo"]
    decomp = primary["decomposition"]
    print(f"  Reliability:  {decomp['reliability']:.4f} (lower is better)")
    print(f"  Resolution:   {decomp['resolution']:.4f} (higher is better)")
    print(f"  Uncertainty:  {decomp['uncertainty']:.4f} (constant)")
    print(f"  Gap:          {decomp['gap']:.6f} (should be near 0)")
    print(f"  BSS:          {primary['bss']:.4f} (positive is good)")
    print(f"  AUC:          {primary['auc']:.4f}" if primary['auc'] else "  AUC: N/A")
    print(f"  ECE:          {primary['ece']:.4f}")

    print("\nCalibration table for primary method:")
    print(f"{'Bin':<15} {'Mean P':<10} {'Observed':<10} {'Count':<8} {'90% CI':<20}")
    print("-" * 63)
    for row in primary["calibration"]:
        ci_str = f"[{row['ci90'][0]:.2f}, {row['ci90'][1]:.2f}]"
        print(f"{row['bin']:<15} {row['mean_p']:<10.3f} {row['observed']:<10.3f} {row['count']:<8} {ci_str:<20}")

    print("=" * 80)
