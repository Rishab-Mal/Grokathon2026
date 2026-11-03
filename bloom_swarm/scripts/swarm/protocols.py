#!/usr/bin/env python3
"""
Evaluation protocols for forecast aggregation.

Two protocols:
A. leave_one_out(method): for each week t, fit on all other weeks and predict week t
B. walk_forward(method): for each week t, fit only on weeks before t

Leakage guarantees (enforced in tests):
- Under LOO: changing y[t] must not change prediction for week t
- Under walk-forward: changing any y[s] with s >= t must not change prediction for week t
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

from .core import CONFIG, brier, log_loss
from .aggregators import get_method, Aggregator, METHODS, PRIMARY_METHOD, EqualLogodds


def leave_one_out(
    method_name: str,
    P: np.ndarray,
    y: np.ndarray,
    agents: List[str],
) -> Tuple[np.ndarray, List[Optional[Dict[str, float]]]]:
    """
    Leave-one-out evaluation protocol.

    For each week t:
    1. Fit method on all weeks except t
    2. Predict week t

    Args:
        method_name: name of aggregation method
        P: T by K probability matrix (may have NaN)
        y: length T outcome array
        agents: list of agent names

    Returns:
        preds: length T prediction array
        weights_per_week: list of weight dicts for each week (or None for non-weighting methods)
    """
    T = P.shape[0]
    preds = np.zeros(T)
    weights_per_week = []

    for t in range(T):
        # Create mask for training data (all except t)
        train_mask = np.ones(T, dtype=bool)
        train_mask[t] = False

        P_train = P[train_mask]
        y_train = y[train_mask]

        # Fit and predict
        method = get_method(method_name, agents)
        method.fit(P_train, y_train)
        pred = method.predict(P[t:t+1, :])[0]
        preds[t] = pred

        weights_per_week.append(method.weights())

    return preds, weights_per_week


def walk_forward(
    method_name: str,
    P: np.ndarray,
    y: np.ndarray,
    agents: List[str],
) -> Tuple[np.ndarray, List[Optional[Dict[str, float]]]]:
    """
    Walk-forward evaluation protocol.

    For each week t:
    1. If t < warmup, use equal_logodds
    2. Otherwise, fit method on weeks 0..t-1 and predict week t

    This is how the system would actually run in deployment.

    Args:
        method_name: name of aggregation method
        P: T by K probability matrix (may have NaN)
        y: length T outcome array
        agents: list of agent names

    Returns:
        preds: length T prediction array
        weights_per_week: list of weight dicts for each week (or None for non-weighting methods)
    """
    T = P.shape[0]
    warmup = CONFIG["walk_forward_warmup"]

    preds = np.zeros(T)
    weights_per_week = []

    for t in range(T):
        if t < warmup:
            # Use equal_logodds as fallback
            method = EqualLogodds(agents)
            method.fit(P[:t] if t > 0 else np.empty((0, P.shape[1])), y[:t] if t > 0 else np.array([]))
            pred = method.predict(P[t:t+1, :])[0]
            preds[t] = pred
            weights_per_week.append(method.weights())
        else:
            # Fit on weeks 0..t-1
            P_train = P[:t]
            y_train = y[:t]

            method = get_method(method_name, agents)
            method.fit(P_train, y_train)
            pred = method.predict(P[t:t+1, :])[0]
            preds[t] = pred
            weights_per_week.append(method.weights())

    return preds, weights_per_week


def evaluate_all_methods(
    P: np.ndarray,
    y: np.ndarray,
    agents: List[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluate all methods under both protocols.

    Returns:
        {method_name: {
            "loo": {"brier": float, "log_loss": float, "preds": array},
            "wf": {"brier": float, "log_loss": float, "preds": array}
        }}
    """
    results = {}

    for method_name in METHODS.keys():
        results[method_name] = {}

        # Leave-one-out
        loo_preds, loo_weights = leave_one_out(method_name, P, y, agents)
        results[method_name]["loo"] = {
            "brier": float(np.mean(brier(loo_preds, y))),
            "log_loss": float(np.mean(log_loss(loo_preds, y))),
            "preds": loo_preds,
            "weights": loo_weights,
        }

        # Walk-forward
        wf_preds, wf_weights = walk_forward(method_name, P, y, agents)
        results[method_name]["wf"] = {
            "brier": float(np.mean(brier(wf_preds, y))),
            "log_loss": float(np.mean(log_loss(wf_preds, y))),
            "preds": wf_preds,
            "weights": wf_weights,
        }

    return results


def evaluate_agents(
    P: np.ndarray,
    y: np.ndarray,
    agents: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate each individual agent.

    Returns:
        {agent_name: {"brier": float, "log_loss": float}}
    """
    results = {}
    K = P.shape[1]

    for k, agent_name in enumerate(agents):
        valid = ~np.isnan(P[:, k])
        if valid.sum() > 0:
            agent_preds = P[valid, k]
            agent_y = y[valid]
            results[agent_name] = {
                "brier": float(np.mean(brier(agent_preds, agent_y))),
                "log_loss": float(np.mean(log_loss(agent_preds, agent_y))),
            }
        else:
            results[agent_name] = {"brier": np.nan, "log_loss": np.nan}

    return results


def print_leaderboard(
    method_results: Dict[str, Dict[str, Dict[str, float]]],
    agent_results: Dict[str, Dict[str, float]],
    primary_method: str = PRIMARY_METHOD,
):
    """Print a formatted leaderboard table."""
    print("\n" + "=" * 80)
    print("PHASE 1 LEADERBOARD")
    print("=" * 80)

    # Methods
    print("\nMETHODS (sorted by LOO Brier):")
    print("-" * 60)
    print(f"{'Method':<25} {'LOO Brier':<12} {'LOO LogLoss':<12} {'WF Brier':<12}")
    print("-" * 60)

    sorted_methods = sorted(
        method_results.items(),
        key=lambda x: x[1]["loo"]["brier"]
    )

    for name, result in sorted_methods:
        marker = " *PRIMARY*" if name == primary_method else ""
        print(f"{name:<25} {result['loo']['brier']:<12.4f} {result['loo']['log_loss']:<12.4f} {result['wf']['brier']:<12.4f}{marker}")

    # Agents
    print("\nAGENTS (for reference):")
    print("-" * 40)
    print(f"{'Agent':<20} {'Brier':<12} {'LogLoss':<12}")
    print("-" * 40)

    sorted_agents = sorted(
        agent_results.items(),
        key=lambda x: x[1]["brier"] if not np.isnan(x[1]["brier"]) else 999
    )

    for name, result in sorted_agents:
        brier_str = f"{result['brier']:.4f}" if not np.isnan(result["brier"]) else "N/A"
        logloss_str = f"{result['log_loss']:.4f}" if not np.isnan(result["log_loss"]) else "N/A"
        print(f"{name:<20} {brier_str:<12} {logloss_str:<12}")

    print("=" * 80)
