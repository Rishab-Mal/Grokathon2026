#!/usr/bin/env python3
"""
Shapley value attribution for forecast agents.

Computes exact Shapley values over all 2^K coalitions.

Value function: v(S) = base_rate_brier - swarm_brier(S)
where swarm_brier(S) is the Brier score of the primary method using only agents in S.

Shapley value for agent i is the average marginal contribution across all coalitions.

Also computes leave-one-out ablation for comparison.
"""

from typing import Dict, List, Tuple, Any, FrozenSet
from itertools import combinations
import numpy as np

from .core import CONFIG, brier, set_seed
from .aggregators import get_method, PRIMARY_METHOD, METHODS
from .protocols import leave_one_out


def compute_coalition_value(
    coalition: FrozenSet[int],
    P: np.ndarray,
    y: np.ndarray,
    agents: List[str],
    base_rate_brier: float,
    cache: Dict[FrozenSet[int], float],
) -> float:
    """
    Compute value function for a coalition.

    v(S) = base_rate_brier - swarm_brier(S)
    v({}) = 0

    Args:
        coalition: set of agent indices
        P: T x K probability matrix
        y: T length outcome array
        agents: list of agent names
        base_rate_brier: Brier score of base rate
        cache: memoization cache

    Returns:
        Value of coalition
    """
    if coalition in cache:
        return cache[coalition]

    if len(coalition) == 0:
        cache[coalition] = 0.0
        return 0.0

    # Subset P to only include coalition agents
    P_subset = P.copy()
    for k in range(P.shape[1]):
        if k not in coalition:
            P_subset[:, k] = np.nan

    # Run leave-one-out with primary method
    preds, _ = leave_one_out(PRIMARY_METHOD, P_subset, y, agents)

    # Compute Brier score
    swarm_brier = float(np.mean(brier(preds, y)))

    # Value = improvement over base rate
    value = base_rate_brier - swarm_brier

    cache[coalition] = value
    return value


def exact_shapley_values(
    P: np.ndarray,
    y: np.ndarray,
    agents: List[str],
    base_rate_brier: float,
) -> Dict[str, float]:
    """
    Compute exact Shapley values for all agents.

    Shapley value for agent i:
    phi_i = sum over S not containing i of:
        (|S|! * (K - |S| - 1)!) / K! * (v(S + {i}) - v(S))

    This is the weighted average of marginal contributions.

    Docstring explanation:
    Shapley values fairly distribute the total value among agents based on
    their marginal contributions across all possible coalitions. This is
    fairer than leave-one-out ablation when agents have overlapping information,
    because it considers what each agent adds in all contexts, not just
    when all others are present.
    """
    set_seed()
    K = len(agents)
    cache: Dict[FrozenSet[int], float] = {}

    # Factorial helper
    def factorial(n):
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    k_factorial = factorial(K)
    shapley = np.zeros(K)

    # For each agent
    for i in range(K):
        phi_i = 0.0

        # For each subset S not containing i
        other_agents = [j for j in range(K) if j != i]

        for size in range(K):  # |S| from 0 to K-1
            for S_tuple in combinations(other_agents, size):
                S = frozenset(S_tuple)
                S_with_i = S | {i}

                # Marginal contribution
                v_S = compute_coalition_value(S, P, y, agents, base_rate_brier, cache)
                v_S_i = compute_coalition_value(S_with_i, P, y, agents, base_rate_brier, cache)
                marginal = v_S_i - v_S

                # Weight
                weight = factorial(size) * factorial(K - size - 1) / k_factorial
                phi_i += weight * marginal

        shapley[i] = phi_i

    return {agents[i]: float(shapley[i]) for i in range(K)}


def leave_one_out_ablation(
    P: np.ndarray,
    y: np.ndarray,
    agents: List[str],
    base_rate_brier: float,
) -> Dict[str, float]:
    """
    Compute leave-one-out ablation values.

    Ablation value for agent i = v(all) - v(all without i)

    This measures the drop in performance when removing each agent.
    It's simpler than Shapley but less fair when agents overlap.
    """
    K = len(agents)
    cache: Dict[FrozenSet[int], float] = {}

    all_agents = frozenset(range(K))
    v_all = compute_coalition_value(all_agents, P, y, agents, base_rate_brier, cache)

    ablation = {}
    for i in range(K):
        without_i = frozenset(j for j in range(K) if j != i)
        v_without = compute_coalition_value(without_i, P, y, agents, base_rate_brier, cache)
        ablation[agents[i]] = v_all - v_without

    return ablation


def verify_shapley_efficiency(
    shapley_values: Dict[str, float],
    P: np.ndarray,
    y: np.ndarray,
    agents: List[str],
    base_rate_brier: float,
) -> Tuple[bool, float]:
    """
    Verify Shapley efficiency axiom: sum of Shapley values = v(all agents).

    Returns:
        (passes, gap)
    """
    K = len(agents)
    cache: Dict[FrozenSet[int], float] = {}

    all_agents = frozenset(range(K))
    v_all = compute_coalition_value(all_agents, P, y, agents, base_rate_brier, cache)

    sum_shapley = sum(shapley_values.values())
    gap = abs(v_all - sum_shapley)

    return (gap < 1e-9, gap)


def compute_shapley_analysis(
    P: np.ndarray,
    y: np.ndarray,
    agents: List[str],
    base_rate_brier: float,
) -> Dict[str, Any]:
    """
    Run complete Shapley analysis.
    """
    shapley_values = exact_shapley_values(P, y, agents, base_rate_brier)
    ablation_values = leave_one_out_ablation(P, y, agents, base_rate_brier)

    # Find most valuable agent
    most_valuable = max(shapley_values.items(), key=lambda x: x[1])[0]

    # Verify efficiency
    efficiency_passes, efficiency_gap = verify_shapley_efficiency(
        shapley_values, P, y, agents, base_rate_brier
    )

    # Compute total value
    total_value = sum(shapley_values.values())

    # Shares
    if total_value > 0:
        shares = {k: v / total_value for k, v in shapley_values.items()}
    else:
        shares = {k: 1.0 / len(agents) for k in agents}

    return {
        "shapley": shapley_values,
        "ablation": ablation_values,
        "most_valuable_agent": most_valuable,
        "total_value": total_value,
        "shares": shares,
        "efficiency_passes": efficiency_passes,
        "efficiency_gap": efficiency_gap,
    }


def print_shapley_summary(analysis: Dict[str, Any]):
    """Print Shapley analysis summary."""
    print("\n" + "=" * 80)
    print("PHASE 3: SHAPLEY VALUE ATTRIBUTION")
    print("=" * 80)

    print("\nShapley values (agent contribution to swarm performance):")
    print(f"{'Agent':<15} {'Shapley':<12} {'Share':<10} {'Ablation':<12}")
    print("-" * 49)

    sorted_agents = sorted(
        analysis["shapley"].items(),
        key=lambda x: x[1],
        reverse=True
    )

    for agent, shapley in sorted_agents:
        share = analysis["shares"][agent]
        ablation = analysis["ablation"][agent]
        print(f"{agent:<15} {shapley:<12.6f} {share*100:<10.1f}% {ablation:<12.6f}")

    print()
    print(f"Most valuable agent: {analysis['most_valuable_agent']}")
    print(f"Total swarm value (improvement over base rate): {analysis['total_value']:.6f}")
    print(f"Efficiency check (sum = total): {analysis['efficiency_passes']} (gap: {analysis['efficiency_gap']:.9f})")

    print("\nNote: Shapley values fairly distribute credit when agents share information.")
    print("Ablation shows impact of removing each agent from the full ensemble.")
    print("=" * 80)
