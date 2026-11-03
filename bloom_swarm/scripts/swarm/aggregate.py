#!/usr/bin/env python3
"""
Aggregate agent outputs using leave-one-out weighted scoring.

This is the core swarm aggregation logic:
1. Load all week inputs and agent outputs
2. For each week, compute agent weights based on their Brier scores on OTHER weeks
3. Combine probabilities using weighted logit averaging
4. Compare against simple average and base rate baselines
5. Output frontend/data.json and print a scoreboard

Usage:
    python scripts/swarm/aggregate.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

AGENT_NAMES = ["Weather", "Runoff", "Satellite", "Historian", "Skeptic"]
MIN_WEEKS_FOR_WEIGHTS = 3


def clip_p(p: float) -> float:
    """Clip probability to [0.03, 0.97]."""
    return float(np.clip(p, 0.03, 0.97))


def logit(p: float) -> float:
    """Logit transform: log(p / (1 - p))."""
    p = clip_p(p)
    return float(np.log(p / (1 - p)))


def sigmoid(x: float) -> float:
    """Sigmoid transform: 1 / (1 + exp(-x))."""
    return float(1 / (1 + np.exp(-x)))


def brier_score(p: float, actual: int) -> float:
    """Compute Brier score: (p - actual)^2."""
    return (p - actual) ** 2


def load_weeks(weeks_dir: Path) -> dict[str, dict]:
    """Load all week input files."""
    weeks = {}
    for path in sorted(weeks_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
            weeks[data["week"]] = data
    return weeks


def load_agent_outputs(outputs_dir: Path) -> dict[str, dict[str, dict]]:
    """
    Load all agent output files.
    Returns: {week: {agent_name: output_dict}}
    """
    outputs = {}
    for path in sorted(outputs_dir.glob("*.json")):
        # Parse filename: YYYY-MM-DD_AgentName.json
        parts = path.stem.rsplit("_", 1)
        if len(parts) != 2:
            print(f"Warning: Skipping malformed filename {path}")
            continue
        week_str, agent_name = parts
        if agent_name not in AGENT_NAMES:
            print(f"Warning: Unknown agent {agent_name} in {path}")
            continue

        with open(path) as f:
            data = json.load(f)

        # Clip p to valid range
        data["p"] = clip_p(data["p"])

        if week_str not in outputs:
            outputs[week_str] = {}
        outputs[week_str][agent_name] = data

    return outputs


def compute_leave_one_out_weights(
    week_str: str,
    all_weeks: dict[str, dict],
    all_outputs: dict[str, dict[str, dict]],
    agent_names: list[str],
) -> dict[str, float]:
    """
    Compute agent weights for a given week using leave-one-out Brier scores.

    For each agent, compute their Brier score on all OTHER weeks,
    then weight = 1 / brier, normalized to sum to 1.
    """
    other_weeks = [w for w in all_weeks.keys() if w != week_str]

    if len(other_weeks) < MIN_WEEKS_FOR_WEIGHTS:
        # Not enough data: use equal weights
        n_agents = len(agent_names)
        return {name: 1.0 / n_agents for name in agent_names}

    agent_briers = {}
    for agent_name in agent_names:
        scores = []
        for other_week in other_weeks:
            if other_week in all_outputs and agent_name in all_outputs[other_week]:
                agent_p = all_outputs[other_week][agent_name]["p"]
                actual = all_weeks[other_week]["actual"]
                scores.append(brier_score(agent_p, actual))

        if scores:
            # Average Brier score across other weeks
            agent_briers[agent_name] = np.mean(scores)
        else:
            # No data for this agent: assign high Brier (low weight)
            agent_briers[agent_name] = 0.5

    # Convert Brier scores to weights: lower Brier = higher weight
    # weight = 1 / brier, but avoid division by zero
    raw_weights = {}
    for agent_name, brier in agent_briers.items():
        # Add small epsilon to avoid division by zero
        raw_weights[agent_name] = 1.0 / (brier + 0.01)

    # Normalize to sum to 1
    total = sum(raw_weights.values())
    weights = {name: w / total for name, w in raw_weights.items()}

    return weights


def compute_base_rate(week_str: str, all_weeks: dict[str, dict]) -> float:
    """Compute base rate from all OTHER weeks (leave-one-out)."""
    other_actuals = [w["actual"] for w_str, w in all_weeks.items() if w_str != week_str]
    if not other_actuals:
        return 0.5
    return np.mean(other_actuals)


def aggregate_week(
    week_str: str,
    week_data: dict,
    agent_outputs: dict[str, dict],
    weights: dict[str, float],
) -> dict:
    """
    Aggregate agent outputs for a single week.

    Returns dict with agents (including weights), swarm_p, simple_avg_p, base_rate_p, actual.
    """
    # Get available agents for this week
    available_agents = [name for name in AGENT_NAMES if name in agent_outputs]

    if not available_agents:
        # No agent outputs for this week
        return None

    # Renormalize weights for available agents
    total_weight = sum(weights.get(name, 0) for name in available_agents)
    if total_weight == 0:
        total_weight = len(available_agents)
        renorm_weights = {name: 1.0 / total_weight for name in available_agents}
    else:
        renorm_weights = {name: weights.get(name, 0) / total_weight for name in available_agents}

    # Build agents list with weights
    agents = []
    logits = []
    for name in available_agents:
        output = agent_outputs[name]
        agent_weight = renorm_weights[name]
        agents.append({
            "name": output["name"],
            "p": output["p"],
            "reason": output["reason"],
            "evidence": output["evidence"],
            "weight": round(agent_weight, 4),
        })
        logits.append(logit(output["p"]))

    # Compute swarm_p: sigmoid(weighted sum of logits)
    weighted_logit_sum = sum(
        renorm_weights[name] * logit(agent_outputs[name]["p"])
        for name in available_agents
    )
    swarm_p = sigmoid(weighted_logit_sum)

    # Compute simple_avg_p: sigmoid(mean of logits)
    mean_logit = np.mean(logits)
    simple_avg_p = sigmoid(mean_logit)

    return {
        "agents": agents,
        "swarm_p": round(swarm_p, 4),
        "simple_avg_p": round(simple_avg_p, 4),
        "actual": week_data["actual"],
    }


def compute_overall_scores(
    aggregated_weeks: list[dict],
    all_weeks: dict[str, dict],
    all_outputs: dict[str, dict[str, dict]],
) -> dict[str, float]:
    """Compute overall Brier scores for agents, swarm, simple_avg, and base_rate."""
    scores = {}

    # Agent scores
    for agent_name in AGENT_NAMES:
        agent_scores = []
        for week_str, week_data in all_weeks.items():
            if week_str in all_outputs and agent_name in all_outputs[week_str]:
                p = all_outputs[week_str][agent_name]["p"]
                actual = week_data["actual"]
                agent_scores.append(brier_score(p, actual))
        if agent_scores:
            scores[agent_name] = round(np.mean(agent_scores), 4)
        else:
            scores[agent_name] = None

    # Swarm and simple_avg scores
    swarm_scores = []
    simple_avg_scores = []
    base_rate_scores = []

    for week_result in aggregated_weeks:
        actual = week_result["actual"]
        swarm_scores.append(brier_score(week_result["swarm_p"], actual))
        simple_avg_scores.append(brier_score(week_result["simple_avg_p"], actual))
        base_rate_scores.append(brier_score(week_result["base_rate_p"], actual))

    scores["swarm"] = round(np.mean(swarm_scores), 4) if swarm_scores else None
    scores["simple_avg"] = round(np.mean(simple_avg_scores), 4) if simple_avg_scores else None
    scores["base_rate"] = round(np.mean(base_rate_scores), 4) if base_rate_scores else None

    return scores


def print_scoreboard(scores: dict[str, float]):
    """Print a sorted scoreboard table to terminal."""
    print("\n" + "=" * 50)
    print("SCOREBOARD (Brier Score - lower is better)")
    print("=" * 50)

    # Sort by score (lower is better)
    sorted_scores = sorted(
        [(name, score) for name, score in scores.items() if score is not None],
        key=lambda x: x[1],
    )

    print(f"{'Rank':<6}{'Name':<15}{'Brier Score':<12}")
    print("-" * 33)
    for rank, (name, score) in enumerate(sorted_scores, 1):
        print(f"{rank:<6}{name:<15}{score:<12.4f}")

    print("=" * 50)


def main():
    # Load environment
    load_dotenv()
    lake_name = os.getenv("LAKE_NAME", "Lake Erie")
    lake_lat = float(os.getenv("LAKE_LAT", "41.6833"))
    lake_lon = float(os.getenv("LAKE_LON", "-82.3833"))

    # Setup paths
    weeks_dir = Path("data/weeks")
    outputs_dir = Path("outputs")
    frontend_dir = Path("frontend")
    frontend_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading week data...")
    all_weeks = load_weeks(weeks_dir)
    if not all_weeks:
        print("Error: No week files found in data/weeks/")
        sys.exit(1)
    print(f"  Loaded {len(all_weeks)} weeks")

    print("Loading agent outputs...")
    all_outputs = load_agent_outputs(outputs_dir)
    if not all_outputs:
        print("Error: No agent output files found in outputs/")
        sys.exit(1)
    print(f"  Loaded outputs for {len(all_outputs)} weeks")

    # Check for missing agent outputs
    for week_str in all_weeks:
        if week_str not in all_outputs:
            print(f"  Warning: No agent outputs for week {week_str}")
        else:
            missing = [name for name in AGENT_NAMES if name not in all_outputs[week_str]]
            if missing:
                print(f"  Warning: Week {week_str} missing agents: {missing}")

    # Aggregate each week
    print("\nAggregating swarm predictions...")
    aggregated_weeks = []

    for week_str in sorted(all_weeks.keys()):
        week_data = all_weeks[week_str]
        agent_outputs = all_outputs.get(week_str, {})

        if not agent_outputs:
            print(f"  Skipping {week_str}: no agent outputs")
            continue

        # Compute weights (leave-one-out on other weeks)
        available_agents = [name for name in AGENT_NAMES if name in agent_outputs]
        weights = compute_leave_one_out_weights(
            week_str, all_weeks, all_outputs, available_agents
        )

        # Aggregate
        result = aggregate_week(week_str, week_data, agent_outputs, weights)
        if result is None:
            continue

        # Add base rate (leave-one-out)
        result["base_rate_p"] = round(compute_base_rate(week_str, all_weeks), 4)
        result["week"] = week_str

        aggregated_weeks.append(result)

    print(f"  Aggregated {len(aggregated_weeks)} weeks")

    # Compute overall scores
    print("\nComputing overall scores...")
    scores = compute_overall_scores(aggregated_weeks, all_weeks, all_outputs)

    # Build final output
    output = {
        "lake": {
            "name": lake_name,
            "lat": lake_lat,
            "lon": lake_lon,
        },
        "weeks": aggregated_weeks,
        "scores": scores,
    }

    # Write to frontend/data.json
    output_path = frontend_dir / "data.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {output_path}")

    # Print scoreboard
    print_scoreboard(scores)


if __name__ == "__main__":
    main()
