#!/usr/bin/env python3
"""
Diversity analysis for forecast aggregation.

- Error correlation matrix between agents
- Effective number of independent agents (participation ratio)
- Diversity-Prediction Theorem verification
- Pairwise disagreement metrics
"""

from typing import Dict, List, Tuple, Any
import numpy as np

from .core import CONFIG, brier


def error_correlation_matrix(
    P: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute Pearson correlation matrix of agent errors (p_i - y).

    Returns:
        corr_matrix: K x K correlation matrix
        valid_agents: list of agent indices with valid data
    """
    T, K = P.shape

    # Compute errors
    errors = P - y[:, np.newaxis]  # T x K

    # Compute pairwise correlations
    corr_matrix = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            valid = ~np.isnan(errors[:, i]) & ~np.isnan(errors[:, j])
            n = valid.sum()

            if n < 2:
                corr_matrix[i, j] = np.nan
                continue

            ei = errors[valid, i]
            ej = errors[valid, j]

            # Pearson correlation
            ei_centered = ei - ei.mean()
            ej_centered = ej - ej.mean()

            denom = np.sqrt(np.sum(ei_centered**2) * np.sum(ej_centered**2))
            if denom > 0:
                corr_matrix[i, j] = np.sum(ei_centered * ej_centered) / denom
            else:
                corr_matrix[i, j] = 1.0 if i == j else 0.0

    return corr_matrix


def effective_n_agents(corr_matrix: np.ndarray) -> float:
    """
    Compute effective number of independent agents.

    Uses participation ratio of eigenvalues:
    N_eff = (sum(lambda))^2 / sum(lambda^2)

    This measures how many independent forecasters the K agents behave like.
    If all agents are identical, N_eff = 1.
    If all agents are uncorrelated, N_eff = K.
    """
    # Handle NaN values by treating them as zero correlation
    corr_clean = np.nan_to_num(corr_matrix, nan=0.0)

    # Set diagonal to 1
    np.fill_diagonal(corr_clean, 1.0)

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(corr_clean)

    # Only positive eigenvalues matter
    eigenvalues = eigenvalues[eigenvalues > 0]

    if len(eigenvalues) == 0:
        return 1.0

    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues**2)

    if sum_lambda_sq == 0:
        return 1.0

    n_eff = sum_lambda**2 / sum_lambda_sq
    return float(n_eff)


def diversity_prediction_theorem(
    P: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray = None,
) -> Dict[str, float]:
    """
    Verify the Diversity Prediction Theorem for linear pooling.

    For equal weights in probability space:
    crowd_error = avg_individual_error - diversity

    where:
    - crowd_error = (p_bar - y)^2 where p_bar is the pool
    - avg_individual_error = mean of (p_i - y)^2
    - diversity = mean of (p_i - p_bar)^2

    The identity should hold exactly within numerical precision.

    If weights are provided, uses weighted versions:
    - crowd_error = (weighted_p_bar - y)^2
    - avg_individual_error = weighted mean of (p_i - y)^2
    - diversity = weighted mean of (p_i - weighted_p_bar)^2

    Note: The primary method pools in log-odds space, so this identity
    is shown for the linear analogue. The theorem demonstrates why
    diversity among agents improves aggregate performance.
    """
    T, K = P.shape

    if weights is None:
        weights = np.ones(K) / K

    # Normalize weights
    weights = weights / weights.sum()

    results_per_week = []
    total_crowd_error = 0.0
    total_avg_individual = 0.0
    total_diversity = 0.0

    for t in range(T):
        P_row = P[t, :]
        valid = ~np.isnan(P_row)

        if not valid.any():
            continue

        # Renormalize weights for valid agents
        w = weights.copy()
        w[~valid] = 0
        if w.sum() > 0:
            w = w / w.sum()
        else:
            continue

        # Pool (weighted mean)
        p_bar = np.sum(w * np.where(valid, P_row, 0))

        # Crowd error
        crowd_error = (p_bar - y[t])**2

        # Individual errors
        individual_errors = np.where(valid, (P_row - y[t])**2, 0)
        avg_individual = np.sum(w * individual_errors)

        # Diversity
        disagreements = np.where(valid, (P_row - p_bar)**2, 0)
        diversity = np.sum(w * disagreements)

        # Verify identity
        reconstructed = avg_individual - diversity
        gap = abs(crowd_error - reconstructed)

        results_per_week.append({
            "week": t,
            "crowd_error": crowd_error,
            "avg_individual": avg_individual,
            "diversity": diversity,
            "gap": gap,
        })

        total_crowd_error += crowd_error
        total_avg_individual += avg_individual
        total_diversity += diversity

    n_valid = len(results_per_week)
    if n_valid == 0:
        return {
            "crowd_error": 0.0,
            "avg_individual": 0.0,
            "diversity": 0.0,
            "gap": 0.0,
            "identity_holds": True,
        }

    avg_crowd = total_crowd_error / n_valid
    avg_individual = total_avg_individual / n_valid
    avg_diversity = total_diversity / n_valid
    avg_gap = abs(avg_crowd - (avg_individual - avg_diversity))

    return {
        "crowd_error": float(avg_crowd),
        "avg_individual": float(avg_individual),
        "diversity": float(avg_diversity),
        "gap": float(avg_gap),
        "identity_holds": avg_gap < 1e-9,
        "per_week": results_per_week,
    }


def pairwise_disagreement(P: np.ndarray) -> Dict[str, float]:
    """
    Compute pairwise disagreement metrics.

    Disagreement is the mean absolute difference between agent pairs.
    """
    T, K = P.shape

    disagreements = []
    pairwise = {}

    for i in range(K):
        for j in range(i + 1, K):
            valid = ~np.isnan(P[:, i]) & ~np.isnan(P[:, j])
            if valid.sum() > 0:
                diff = np.abs(P[valid, i] - P[valid, j])
                mean_diff = float(np.mean(diff))
                disagreements.append(mean_diff)
                pairwise[(i, j)] = mean_diff

    overall = float(np.mean(disagreements)) if disagreements else 0.0

    return {
        "overall": overall,
        "pairwise": pairwise,
        "max": float(np.max(disagreements)) if disagreements else 0.0,
        "min": float(np.min(disagreements)) if disagreements else 0.0,
    }


def compute_diversity_analysis(
    P: np.ndarray,
    y: np.ndarray,
    agents: List[str],
    primary_weights: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Run complete diversity analysis.
    """
    corr_matrix = error_correlation_matrix(P, y)
    n_eff = effective_n_agents(corr_matrix)

    # Equal weights diversity theorem
    theorem_equal = diversity_prediction_theorem(P, y, weights=None)

    # Weighted diversity theorem (if weights provided)
    theorem_weighted = None
    if primary_weights is not None:
        theorem_weighted = diversity_prediction_theorem(P, y, weights=primary_weights)

    disagreement = pairwise_disagreement(P)

    return {
        "corr_matrix": corr_matrix.tolist(),
        "agents": agents,
        "effective_n": n_eff,
        "theorem_equal": theorem_equal,
        "theorem_weighted": theorem_weighted,
        "disagreement": disagreement,
    }


def print_diversity_summary(analysis: Dict[str, Any]):
    """Print diversity analysis summary."""
    print("\n" + "=" * 80)
    print("PHASE 3: DIVERSITY ANALYSIS")
    print("=" * 80)

    agents = analysis["agents"]
    K = len(agents)

    print(f"\nEffective number of independent agents: {analysis['effective_n']:.2f} / {K}")
    print(f"  (5 agents behave like {analysis['effective_n']:.1f} independent forecasters)")

    print("\nError correlation matrix:")
    corr = np.array(analysis["corr_matrix"])
    header = "           " + "  ".join(f"{a[:8]:>8}" for a in agents)
    print(header)
    for i, agent in enumerate(agents):
        row = f"{agent[:10]:<10}"
        for j in range(K):
            val = corr[i, j]
            if np.isnan(val):
                row += f"{'N/A':>10}"
            else:
                row += f"{val:>10.3f}"
        print(row)

    print("\nDiversity Prediction Theorem (equal weights, linear pool):")
    th = analysis["theorem_equal"]
    print(f"  Crowd error:        {th['crowd_error']:.6f}")
    print(f"  Avg individual:     {th['avg_individual']:.6f}")
    print(f"  Diversity:          {th['diversity']:.6f}")
    print(f"  Gap (should be ~0): {th['gap']:.9f}")
    print(f"  Identity holds:     {th['identity_holds']}")

    if analysis["theorem_weighted"]:
        print("\nDiversity Prediction Theorem (primary weights, linear pool):")
        tw = analysis["theorem_weighted"]
        print(f"  Crowd error:        {tw['crowd_error']:.6f}")
        print(f"  Avg individual:     {tw['avg_individual']:.6f}")
        print(f"  Diversity:          {tw['diversity']:.6f}")
        print(f"  Note: Primary method pools in log-odds, so this is the linear analogue.")

    print(f"\nPairwise disagreement:")
    print(f"  Overall mean: {analysis['disagreement']['overall']:.4f}")
    print(f"  Range: [{analysis['disagreement']['min']:.4f}, {analysis['disagreement']['max']:.4f}]")

    print("=" * 80)
