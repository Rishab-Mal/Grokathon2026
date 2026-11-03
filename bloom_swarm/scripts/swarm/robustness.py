#!/usr/bin/env python3
"""
Robustness and stress tests for forecast aggregation.

Tests under leave-one-out with primary method:
a. Liar injection: adversarial agent predicting wrong
b. Coin injection: random agent
c. Herd injection: duplicates of worst agent
d. Degradation curve: performance vs number of agents
e. Clip sensitivity: different clipping bounds
f. Missing data stress: random missing cells
"""

from typing import Dict, List, Any, Tuple
import numpy as np

from .core import CONFIG, clip, brier, set_seed
from .aggregators import get_method, PRIMARY_METHOD, METHODS, InverseBrierLogodds, BatesGranger
from .protocols import leave_one_out


SEED = CONFIG["seed"]


def inject_liar(
    P: np.ndarray,
    y: np.ndarray,
    agents: List[str],
    noise_sd: float = 0.1,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Inject a Liar agent that predicts clip(1 - y + noise).

    The weighting should push Liar's weight toward zero.

    Returns:
        P_new: expanded probability matrix
        agents_new: expanded agent list
        results: weight and Brier scores
    """
    set_seed()
    T, K = P.shape

    # Liar predictions: opposite of truth plus noise
    liar_preds = 1 - y + np.random.normal(0, noise_sd, T)
    liar_preds = clip(liar_preds)

    # Expand P
    P_new = np.column_stack([P, liar_preds])
    agents_new = agents + ["Liar"]

    # Run primary method
    preds_with, weights_with = leave_one_out(PRIMARY_METHOD, P_new, y, agents_new)
    preds_without, _ = leave_one_out(PRIMARY_METHOD, P, y, agents)

    # Get Liar's average weight
    liar_weights = [w.get("Liar", 0) for w in weights_with if w]
    avg_liar_weight = np.mean(liar_weights) if liar_weights else 0

    brier_with = float(np.mean(brier(preds_with, y)))
    brier_without = float(np.mean(brier(preds_without, y)))

    return P_new, agents_new, {
        "avg_liar_weight": avg_liar_weight,
        "brier_with_liar": brier_with,
        "brier_without_liar": brier_without,
        "brier_change": brier_with - brier_without,
        "weight_ratio": avg_liar_weight / (1 / (K + 1)),  # vs equal weight
    }


def inject_coin(
    P: np.ndarray,
    y: np.ndarray,
    agents: List[str],
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Inject a Coin agent with uniform random predictions.

    Returns:
        P_new: expanded probability matrix
        agents_new: expanded agent list
        results: weight and Brier scores
    """
    set_seed()
    T, K = P.shape

    # Coin predictions: uniform random
    coin_preds = np.random.uniform(0.03, 0.97, T)

    # Expand P
    P_new = np.column_stack([P, coin_preds])
    agents_new = agents + ["Coin"]

    # Run primary method
    preds_with, weights_with = leave_one_out(PRIMARY_METHOD, P_new, y, agents_new)
    preds_without, _ = leave_one_out(PRIMARY_METHOD, P, y, agents)

    # Get Coin's average weight
    coin_weights = [w.get("Coin", 0) for w in weights_with if w]
    avg_coin_weight = np.mean(coin_weights) if coin_weights else 0

    brier_with = float(np.mean(brier(preds_with, y)))
    brier_without = float(np.mean(brier(preds_without, y)))

    return P_new, agents_new, {
        "avg_coin_weight": avg_coin_weight,
        "brier_with_coin": brier_with,
        "brier_without_coin": brier_without,
        "brier_change": brier_with - brier_without,
    }


def inject_herd(
    P: np.ndarray,
    y: np.ndarray,
    agents: List[str],
    n_copies: int = 3,
    noise_sd: float = 0.02,
) -> Dict[str, Any]:
    """
    Inject copies of the worst agent with small noise.

    Tests whether Bates-Granger resists herding better than inverse Brier.
    """
    set_seed()
    T, K = P.shape

    # Find worst agent (highest Brier)
    agent_briers = []
    for k in range(K):
        valid = ~np.isnan(P[:, k])
        if valid.sum() > 0:
            agent_briers.append(float(np.mean(brier(P[valid, k], y[valid]))))
        else:
            agent_briers.append(0.5)

    worst_idx = np.argmax(agent_briers)
    worst_agent = agents[worst_idx]

    # Create copies
    copies = []
    for i in range(n_copies):
        copy_preds = P[:, worst_idx] + np.random.normal(0, noise_sd, T)
        copy_preds = clip(copy_preds)
        copies.append(copy_preds)

    # Expand P
    P_herd = np.column_stack([P] + copies)
    agents_herd = agents + [f"{worst_agent}_copy{i+1}" for i in range(n_copies)]

    # Run inverse Brier
    preds_ib, weights_ib = leave_one_out("inverse_brier_logodds", P_herd, y, agents_herd)
    brier_ib = float(np.mean(brier(preds_ib, y)))

    # Run Bates-Granger
    preds_bg, weights_bg = leave_one_out("bates_granger", P_herd, y, agents_herd)
    brier_bg = float(np.mean(brier(preds_bg, y)))

    # Original Briers
    preds_ib_orig, _ = leave_one_out("inverse_brier_logodds", P, y, agents)
    preds_bg_orig, _ = leave_one_out("bates_granger", P, y, agents)
    brier_ib_orig = float(np.mean(brier(preds_ib_orig, y)))
    brier_bg_orig = float(np.mean(brier(preds_bg_orig, y)))

    return {
        "worst_agent": worst_agent,
        "n_copies": n_copies,
        "inverse_brier": {
            "brier_original": brier_ib_orig,
            "brier_with_herd": brier_ib,
            "degradation": brier_ib - brier_ib_orig,
        },
        "bates_granger": {
            "brier_original": brier_bg_orig,
            "brier_with_herd": brier_bg,
            "degradation": brier_bg - brier_bg_orig,
        },
        "bg_resists_better": (brier_bg - brier_bg_orig) < (brier_ib - brier_ib_orig),
    }


def degradation_curve(
    P: np.ndarray,
    y: np.ndarray,
    agents: List[str],
    n_samples: int = 200,
) -> Dict[str, Any]:
    """
    Compute swarm Brier as agents are randomly removed.

    Returns mean and std of Brier at each number of remaining agents.
    """
    set_seed()
    T, K = P.shape

    results = {
        "n_agents": [],
        "mean_brier": [],
        "std_brier": [],
    }

    for n_keep in range(1, K + 1):
        briers = []

        for _ in range(n_samples):
            # Random subset of n_keep agents
            keep_idx = np.random.choice(K, n_keep, replace=False)

            # Mask out non-kept agents
            P_subset = P.copy()
            for k in range(K):
                if k not in keep_idx:
                    P_subset[:, k] = np.nan

            # Run primary method
            preds, _ = leave_one_out(PRIMARY_METHOD, P_subset, y, agents)
            briers.append(float(np.mean(brier(preds, y))))

        results["n_agents"].append(n_keep)
        results["mean_brier"].append(float(np.mean(briers)))
        results["std_brier"].append(float(np.std(briers)))

    return results


def clip_sensitivity(
    P: np.ndarray,
    y: np.ndarray,
    agents: List[str],
) -> Dict[str, Any]:
    """
    Test sensitivity to different clipping bounds.
    """
    clip_configs = [
        (0.01, 0.99),
        (0.03, 0.97),  # default
        (0.05, 0.95),
    ]

    results = {}

    for lo, hi in clip_configs:
        # Re-clip P
        P_clipped = np.clip(P, lo, hi)

        # Run primary method
        preds, _ = leave_one_out(PRIMARY_METHOD, P_clipped, y, agents)
        brier_score = float(np.mean(brier(preds, y)))

        results[f"[{lo}, {hi}]"] = brier_score

    # Compute changes from default
    default_brier = results["[0.03, 0.97]"]
    changes = {}
    for key in results:
        if key != "[0.03, 0.97]":
            changes[f"{key}_change"] = results[key] - default_brier
    results.update(changes)

    return results


def missing_data_stress(
    P: np.ndarray,
    y: np.ndarray,
    agents: List[str],
    fractions: List[float] = [0.1, 0.2, 0.3],
    n_repeats: int = 100,
) -> Dict[str, Any]:
    """
    Test robustness to random missing data.
    """
    set_seed()
    T, K = P.shape
    n_cells = T * K

    results = {}

    for frac in fractions:
        n_blank = int(frac * n_cells)
        briers = []

        for _ in range(n_repeats):
            # Random cells to blank
            flat_idx = np.random.choice(n_cells, n_blank, replace=False)
            rows = flat_idx // K
            cols = flat_idx % K

            P_missing = P.copy()
            P_missing[rows, cols] = np.nan

            # Run primary method
            preds, _ = leave_one_out(PRIMARY_METHOD, P_missing, y, agents)
            briers.append(float(np.mean(brier(preds, y))))

        results[f"{int(frac*100)}%_missing"] = {
            "mean_brier": float(np.mean(briers)),
            "std_brier": float(np.std(briers)),
            "min_brier": float(np.min(briers)),
            "max_brier": float(np.max(briers)),
        }

    return results


def compute_robustness_analysis(
    P: np.ndarray,
    y: np.ndarray,
    agents: List[str],
) -> Dict[str, Any]:
    """
    Run complete robustness analysis.
    """
    print("  Running liar injection...")
    _, _, liar = inject_liar(P, y, agents)

    print("  Running coin injection...")
    _, _, coin = inject_coin(P, y, agents)

    print("  Running herd injection...")
    herd = inject_herd(P, y, agents)

    print("  Running degradation curve...")
    degradation = degradation_curve(P, y, agents, n_samples=200)

    print("  Running clip sensitivity...")
    clip_sens = clip_sensitivity(P, y, agents)

    print("  Running missing data stress...")
    missing = missing_data_stress(P, y, agents)

    return {
        "liar": liar,
        "coin": coin,
        "herd": herd,
        "degradation": degradation,
        "clip": clip_sens,
        "missing": missing,
    }


def print_robustness_summary(analysis: Dict[str, Any]):
    """Print robustness analysis summary."""
    print("\n" + "=" * 80)
    print("PHASE 4: ROBUSTNESS ANALYSIS")
    print("=" * 80)

    print("\n1. LIAR INJECTION (adversarial agent predicting opposite of truth):")
    liar = analysis["liar"]
    print(f"   Average Liar weight: {liar['avg_liar_weight']:.4f}")
    print(f"   Expected if equal: {1/6:.4f}")
    print(f"   Brier without Liar: {liar['brier_without_liar']:.4f}")
    print(f"   Brier with Liar: {liar['brier_with_liar']:.4f}")
    print(f"   Change: {liar['brier_change']:+.4f}")

    print("\n2. COIN INJECTION (random agent):")
    coin = analysis["coin"]
    print(f"   Average Coin weight: {coin['avg_coin_weight']:.4f}")
    print(f"   Brier without Coin: {coin['brier_without_coin']:.4f}")
    print(f"   Brier with Coin: {coin['brier_with_coin']:.4f}")
    print(f"   Change: {coin['brier_change']:+.4f}")

    print("\n3. HERD INJECTION (3 copies of worst agent):")
    herd = analysis["herd"]
    print(f"   Worst agent: {herd['worst_agent']}")
    print(f"   Inverse Brier degradation: {herd['inverse_brier']['degradation']:+.4f}")
    print(f"   Bates-Granger degradation: {herd['bates_granger']['degradation']:+.4f}")
    print(f"   Bates-Granger resists better: {herd['bg_resists_better']}")

    print("\n4. DEGRADATION CURVE:")
    deg = analysis["degradation"]
    for i, n in enumerate(deg["n_agents"]):
        print(f"   {n} agents: Brier = {deg['mean_brier'][i]:.4f} +/- {deg['std_brier'][i]:.4f}")

    print("\n5. CLIP SENSITIVITY:")
    clip_sens = analysis["clip"]
    for key, val in clip_sens.items():
        if "change" not in key:
            print(f"   {key}: Brier = {val:.4f}")

    print("\n6. MISSING DATA STRESS:")
    missing = analysis["missing"]
    for key, val in missing.items():
        print(f"   {key}: Brier = {val['mean_brier']:.4f} +/- {val['std_brier']:.4f}")

    print("=" * 80)
