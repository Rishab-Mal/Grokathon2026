#!/usr/bin/env python3
"""
Aggregate agent outputs and run full evaluation pipeline.

This is the thin entry point that calls all evaluation modules in order.
Maintains backward compatibility with frontend/data.json format.

Usage:
    python scripts/swarm/aggregate.py [--phase N] [--figures]

Phases:
    1: Aggregators and protocols
    2: Metrics and significance
    3: Diversity and Shapley
    4: Decision value and robustness
    5: Reporting and figures
    all: Run all phases (default)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.core import load_data, set_seed, CONFIG, brier
from swarm.aggregators import METHODS, PRIMARY_METHOD
from swarm.protocols import (
    evaluate_all_methods,
    evaluate_agents,
    print_leaderboard,
)
from swarm.metrics import compute_metrics_for_all, print_metrics_summary
from swarm.significance import compare_to_baselines, print_significance_table
from swarm.diversity import compute_diversity_analysis, print_diversity_summary
from swarm.shapley import compute_shapley_analysis, print_shapley_summary
from swarm.decision import compute_economic_analysis, print_economic_summary
from swarm.robustness import compute_robustness_analysis, print_robustness_summary
from swarm.report import (
    update_frontend_data,
    write_numbers_md,
    write_appendix_md,
    generate_all_figures,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def run_phase1(weeks, agents, P, y, meta):
    """Phase 1: Aggregators and protocols."""
    print("\n" + "=" * 80)
    print("PHASE 1: AGGREGATORS AND PROTOCOLS")
    print("=" * 80)

    set_seed()

    print("\nEvaluating all methods under LOO and walk-forward protocols...")
    method_results = evaluate_all_methods(P, y, agents)
    agent_results = evaluate_agents(P, y, agents)
    print_leaderboard(method_results, agent_results)

    return method_results, agent_results


def run_phase2(method_results, agent_results, P, y, agents):
    """Phase 2: Metrics and significance."""
    print("\n" + "=" * 80)
    print("PHASE 2: METRICS AND SIGNIFICANCE")
    print("=" * 80)

    # Build agent predictions dict
    agent_preds = {agent: P[:, k] for k, agent in enumerate(agents)}

    # Compute metrics
    print("\nComputing detailed metrics...")
    all_metrics = compute_metrics_for_all(method_results, agent_preds, y)
    print_metrics_summary(all_metrics, PRIMARY_METHOD)

    # Significance testing
    print("\nRunning significance tests...")
    primary_preds = method_results[PRIMARY_METHOD]["loo"]["preds"]
    baselines = {
        "base_rate": method_results["base_rate"]["loo"]["preds"],
        "equal_logodds": method_results["equal_logodds"]["loo"]["preds"],
        "best_agent_selected": method_results["best_agent_selected"]["loo"]["preds"],
    }
    # Add agents
    for k, agent in enumerate(agents):
        valid = ~np.isnan(P[:, k])
        if valid.all():
            baselines[agent] = P[:, k]

    significance = compare_to_baselines(primary_preds, baselines, y)
    print_significance_table(significance)

    return all_metrics, significance


def run_phase3(P, y, agents, method_results):
    """Phase 3: Diversity and Shapley."""
    print("\n" + "=" * 80)
    print("PHASE 3: DIVERSITY AND SHAPLEY")
    print("=" * 80)

    # Get primary weights for weighted analysis
    primary_weights = method_results[PRIMARY_METHOD]["loo"]["weights"][0]
    if primary_weights:
        weights_array = np.array([primary_weights.get(a, 0.0) for a in agents])
    else:
        weights_array = None

    # Diversity
    print("\nRunning diversity analysis...")
    diversity = compute_diversity_analysis(P, y, agents, weights_array)
    print_diversity_summary(diversity)

    # Shapley
    base_rate_brier = method_results["base_rate"]["loo"]["brier"]
    print("\nRunning Shapley analysis...")
    shapley = compute_shapley_analysis(P, y, agents, base_rate_brier)
    print_shapley_summary(shapley)

    return diversity, shapley


def run_phase4(P, y, agents, method_results):
    """Phase 4: Decision value and robustness."""
    print("\n" + "=" * 80)
    print("PHASE 4: DECISION VALUE AND ROBUSTNESS")
    print("=" * 80)

    # Economic analysis
    print("\nRunning economic analysis...")
    forecasts = {
        PRIMARY_METHOD: method_results[PRIMARY_METHOD]["loo"]["preds"],
        "equal_logodds": method_results["equal_logodds"]["loo"]["preds"],
        "best_agent_selected": method_results["best_agent_selected"]["loo"]["preds"],
        "base_rate": method_results["base_rate"]["loo"]["preds"],
    }
    # Add agents
    for k, agent in enumerate(agents):
        valid = ~np.isnan(P[:, k])
        if valid.all():
            forecasts[agent] = P[:, k]

    economic = compute_economic_analysis(forecasts, y)
    print_economic_summary(economic)

    # Robustness
    print("\nRunning robustness analysis...")
    robustness = compute_robustness_analysis(P, y, agents)
    print_robustness_summary(robustness)

    return economic, robustness


def run_phase5(
    weeks, agents, P, y, meta,
    method_results, agent_results, metrics, significance,
    diversity, shapley, economic, robustness,
    figures_only: bool = False,
):
    """Phase 5: Reporting and figures."""
    print("\n" + "=" * 80)
    print("PHASE 5: REPORTING AND FIGURES")
    print("=" * 80)

    frontend_dir = Path("frontend")
    pitch_dir = Path("pitch")
    figures_dir = pitch_dir / "figures"

    frontend_dir.mkdir(exist_ok=True)
    pitch_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # Build base frontend data first
    existing_data = build_base_frontend_data(weeks, agents, P, y, meta, method_results)

    # Update with evaluation
    print("\nUpdating frontend/data.json...")
    updated_data = update_frontend_data(
        existing_data, CONFIG, method_results, agent_results,
        metrics, significance, diversity, shapley, economic, robustness
    )

    frontend_path = frontend_dir / "data.json"
    with open(frontend_path, "w") as f:
        json.dump(updated_data, f, indent=2)
    print(f"  Wrote {frontend_path}")

    # Write pitch files
    print("\nWriting pitch files...")
    write_numbers_md(pitch_dir / "numbers.md", method_results, agent_results,
                     significance, diversity, shapley, economic, robustness)
    print(f"  Wrote {pitch_dir / 'numbers.md'}")

    write_appendix_md(pitch_dir / "appendix.md")
    print(f"  Wrote {pitch_dir / 'appendix.md'}")

    # Generate figures
    print("\nGenerating figures...")
    generate_all_figures(metrics, diversity, shapley, economic, robustness, figures_dir)

    return updated_data


def build_base_frontend_data(weeks, agents, P, y, meta, method_results):
    """Build base frontend data structure."""
    lake_name = os.getenv("LAKE_NAME", "Lake Erie")
    lake_lat = float(os.getenv("LAKE_LAT", "41.6833"))
    lake_lon = float(os.getenv("LAKE_LON", "-82.3833"))

    T, K = P.shape
    primary_preds = method_results[PRIMARY_METHOD]["loo"]["preds"]
    primary_weights = method_results[PRIMARY_METHOD]["loo"]["weights"]
    simple_avg_preds = method_results["equal_logodds"]["loo"]["preds"]
    base_rate_preds = method_results["base_rate"]["loo"]["preds"]

    weeks_data = []
    for t, week_str in enumerate(weeks):
        agents_data = []
        week_weights = primary_weights[t] if primary_weights[t] else {}

        for k, agent_name in enumerate(agents):
            if np.isnan(P[t, k]):
                continue
            agent_meta = meta.get(week_str, {}).get(agent_name, {})
            agents_data.append({
                "name": agent_name,
                "p": round(float(P[t, k]), 4),
                "reason": agent_meta.get("reason", ""),
                "evidence": agent_meta.get("evidence", []),
                "weight": round(week_weights.get(agent_name, 1.0 / len(agents)), 4),
            })

        methods_data = {}
        for method_name, result in method_results.items():
            methods_data[method_name] = {
                "loo": round(float(result["loo"]["preds"][t]), 4),
                "wf": round(float(result["wf"]["preds"][t]), 4),
            }

        weeks_data.append({
            "week": week_str,
            "agents": agents_data,
            "swarm_p": round(float(primary_preds[t]), 4),
            "simple_avg_p": round(float(simple_avg_preds[t]), 4),
            "base_rate_p": round(float(base_rate_preds[t]), 4),
            "actual": int(y[t]),
            "methods": methods_data,
        })

    scores = {}
    for k, agent_name in enumerate(agents):
        valid = ~np.isnan(P[:, k])
        if valid.sum() > 0:
            scores[agent_name] = round(float(np.mean(brier(P[valid, k], y[valid]))), 4)

    scores["swarm"] = round(method_results[PRIMARY_METHOD]["loo"]["brier"], 4)
    scores["simple_avg"] = round(method_results["equal_logodds"]["loo"]["brier"], 4)
    scores["base_rate"] = round(method_results["base_rate"]["loo"]["brier"], 4)

    return {
        "lake": {"name": lake_name, "lat": lake_lat, "lon": lake_lon},
        "weeks": weeks_data,
        "scores": scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Swarm aggregation and evaluation")
    parser.add_argument("--phase", type=str, default="all",
                        help="Phase to run: 1, 2, 3, 4, 5, or all")
    parser.add_argument("--figures", action="store_true",
                        help="Only generate figures (assumes data exists)")
    args = parser.parse_args()

    start_time = time.time()

    # Setup paths
    weeks_dir = Path("data/weeks")
    outputs_dir = Path("outputs")
    frontend_dir = Path("frontend")
    frontend_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading data...")
    weeks, agents, P, y, meta = load_data(weeks_dir, outputs_dir)

    if len(weeks) == 0:
        print("Error: No week files found in data/weeks/")
        sys.exit(1)

    print(f"  Loaded {len(weeks)} weeks, {len(agents)} agents")
    print(f"  Bloom rate: {y.mean():.1%}")

    # Run phases
    phase = args.phase.lower()

    # Phase 1 is always needed
    method_results, agent_results = run_phase1(weeks, agents, P, y, meta)

    if phase in ["2", "3", "4", "5", "all"]:
        metrics, significance = run_phase2(method_results, agent_results, P, y, agents)
    else:
        metrics, significance = {}, []

    if phase in ["3", "4", "5", "all"]:
        diversity, shapley = run_phase3(P, y, agents, method_results)
    else:
        diversity, shapley = {}, {}

    if phase in ["4", "5", "all"]:
        economic, robustness = run_phase4(P, y, agents, method_results)
    else:
        economic, robustness = {}, {}

    if phase in ["5", "all"] or args.figures:
        run_phase5(
            weeks, agents, P, y, meta,
            method_results, agent_results, metrics, significance,
            diversity, shapley, economic, robustness,
            figures_only=args.figures,
        )

    elapsed = time.time() - start_time
    print(f"\n{'=' * 50}")
    print(f"COMPLETE in {elapsed:.1f}s")
    print(f"{'=' * 50}")

    # Final summary
    print(f"\nPrimary method: {PRIMARY_METHOD}")
    print(f"  LOO Brier: {method_results[PRIMARY_METHOD]['loo']['brier']:.4f}")
    print(f"  WF Brier:  {method_results[PRIMARY_METHOD]['wf']['brier']:.4f}")

    best_agent = min(agent_results.items(), key=lambda x: x[1]["brier"])
    print(f"Best agent: {best_agent[0]} with Brier {best_agent[1]['brier']:.4f}")

    if diversity:
        print(f"Effective N: {diversity.get('effective_n', 0):.2f}")
    if shapley:
        print(f"Most valuable agent: {shapley.get('most_valuable_agent', 'N/A')}")

    return method_results, agent_results


if __name__ == "__main__":
    main()
