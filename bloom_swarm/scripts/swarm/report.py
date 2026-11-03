#!/usr/bin/env python3
"""
Report generation for evaluation results.

Outputs:
1. frontend/data.json with evaluation key added
2. pitch/numbers.md with key stats
3. pitch/appendix.md with methodology
4. pitch/figures/*.png visualizations
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .core import CONFIG
from .aggregators import PRIMARY_METHOD


def update_frontend_data(
    existing_data: Dict,
    config: Dict,
    method_results: Dict,
    agent_results: Dict,
    metrics: Dict,
    significance: List[Dict],
    diversity: Dict,
    shapley: Dict,
    economic: Dict,
    robustness: Dict,
) -> Dict:
    """
    Update frontend/data.json with evaluation key.

    Preserves all existing keys.
    """
    # Build leaderboard
    leaderboard_loo = []
    leaderboard_wf = []

    # Methods
    for method_name, result in method_results.items():
        if method_name in metrics:
            m_loo = metrics[method_name].get("loo", {})
            m_wf = metrics[method_name].get("wf", {})

            leaderboard_loo.append({
                "name": method_name,
                "kind": "method",
                "brier": result["loo"]["brier"],
                "log_loss": result["loo"]["log_loss"],
                "bss": m_loo.get("bss", 0),
                "auc": m_loo.get("auc"),
                "ece": m_loo.get("ece", 0),
            })
            leaderboard_wf.append({
                "name": method_name,
                "kind": "method",
                "brier": result["wf"]["brier"],
                "log_loss": result["wf"]["log_loss"],
                "bss": m_wf.get("bss", 0),
                "auc": m_wf.get("auc"),
                "ece": m_wf.get("ece", 0),
            })

    # Agents
    for agent_name, result in agent_results.items():
        if agent_name in metrics:
            m_agent = metrics[agent_name].get("agent", {})

            leaderboard_loo.append({
                "name": agent_name,
                "kind": "agent",
                "brier": result["brier"],
                "log_loss": result["log_loss"],
                "bss": m_agent.get("bss", 0),
                "auc": m_agent.get("auc"),
                "ece": m_agent.get("ece", 0),
            })

    # Sort by brier
    leaderboard_loo.sort(key=lambda x: x["brier"] if x["brier"] is not None else 999)
    leaderboard_wf.sort(key=lambda x: x["brier"] if x["brier"] is not None else 999)

    # Build decomposition
    decomposition = {}
    for name in list(method_results.keys()) + list(agent_results.keys()):
        if name in metrics:
            proto = "loo" if "loo" in metrics[name] else "agent"
            d = metrics[name].get(proto, {}).get("decomposition", {})
            if d:
                decomposition[name] = {
                    "reliability": d.get("reliability", 0),
                    "resolution": d.get("resolution", 0),
                    "uncertainty": d.get("uncertainty", 0),
                    "gap": d.get("gap", 0),
                }

    # Build calibration
    calibration = {}
    for name in list(method_results.keys()) + list(agent_results.keys()):
        if name in metrics:
            proto = "loo" if "loo" in metrics[name] else "agent"
            cal = metrics[name].get(proto, {}).get("calibration", [])
            if cal:
                calibration[name] = cal

    # Build evaluation
    evaluation = {
        "config": config,
        "primary_method": PRIMARY_METHOD,
        "n_weeks": len(existing_data.get("weeks", [])),
        "leaderboard": {
            "loo": leaderboard_loo,
            "wf": leaderboard_wf,
        },
        "decomposition": decomposition,
        "calibration": calibration,
        "significance": significance,
        "diversity": {
            "corr": diversity.get("corr_matrix", []),
            "agents": diversity.get("agents", []),
            "effective_n": diversity.get("effective_n", 0),
            "theorem": {
                "crowd": diversity.get("theorem_equal", {}).get("crowd_error", 0),
                "avg_individual": diversity.get("theorem_equal", {}).get("avg_individual", 0),
                "diversity": diversity.get("theorem_equal", {}).get("diversity", 0),
            },
        },
        "shapley": shapley.get("shapley", {}),
        "ablation": shapley.get("ablation", {}),
        "most_valuable_agent": shapley.get("most_valuable_agent", ""),
        "economic_value": {
            "r": economic.get("r_grid", []),
            "curves": economic.get("curves", {}),
            "peaks": economic.get("peaks", {}),
        },
        "robustness": {
            "liar": robustness.get("liar", {}),
            "coin": robustness.get("coin", {}),
            "herd": robustness.get("herd", {}),
            "degradation": robustness.get("degradation", {}),
            "clip": robustness.get("clip", {}),
            "missing": robustness.get("missing", {}),
        },
    }

    # Merge with existing data
    result = existing_data.copy()
    result["evaluation"] = evaluation

    return result


def write_numbers_md(
    output_path: Path,
    method_results: Dict,
    agent_results: Dict,
    significance: List[Dict],
    diversity: Dict,
    shapley: Dict,
    economic: Dict,
    robustness: Dict,
):
    """
    Write pitch/numbers.md with key statistics.
    """
    lines = ["# Key Numbers for Pitch\n"]
    lines.append("*Auto-generated from evaluation results*\n\n")

    # Primary method vs best agent
    primary_brier = method_results[PRIMARY_METHOD]["loo"]["brier"]
    best_agent = min(agent_results.items(), key=lambda x: x[1]["brier"])
    best_agent_name, best_agent_score = best_agent
    best_agent_brier = best_agent_score["brier"]

    # Find best_agent_selected result
    bas_brier = method_results.get("best_agent_selected", {}).get("loo", {}).get("brier", 0)

    # Find comparison to best agent in significance
    vs_best = next((s for s in significance if s["vs"] == best_agent_name), None)
    vs_bas = next((s for s in significance if s["vs"] == "best_agent_selected"), None)

    lines.append("## Headline\n\n")

    if primary_brier <= best_agent_brier:
        pct_change = (1 - primary_brier / best_agent_brier) * 100
        win_rate = vs_best["boot_win_rate"] * 100 if vs_best else 0
        verdict = vs_best["verdict"] if vs_best else "N/A"
        lines.append(f"- The swarm ({PRIMARY_METHOD}) achieves **{pct_change:.0f}% lower Brier** than the best agent ({best_agent_name})\n")
        lines.append(f"- Bootstrap win rate: {win_rate:.0f}%\n")
        lines.append(f"- Verdict: {verdict}\n")
    else:
        # Honest framing when swarm loses
        lines.append(f"- The swarm stays close to the best agent ({best_agent_name}) without needing to know in advance which agent is best\n")
        if bas_brier >= primary_brier:
            lines.append(f"- best_agent_selected (which has to guess) did similar or worse (Brier: {bas_brier:.3f} vs {primary_brier:.3f})\n")
        win_rate = vs_best["boot_win_rate"] * 100 if vs_best else 0
        lines.append(f"- Bootstrap win rate vs best agent: {win_rate:.0f}%\n")

    # Base rate comparison
    base_brier = method_results["base_rate"]["loo"]["brier"]
    bss = (1 - primary_brier / base_brier) * 100
    lines.append(f"- {bss:.0f}% improvement over climatological base rate\n\n")

    # Diversity
    n_eff = diversity.get("effective_n", 0)
    K = len(diversity.get("agents", []))
    lines.append("## Diversity\n\n")
    lines.append(f"- **{K} agents behave like {n_eff:.1f} independent forecasters**\n")

    # Theorem
    th = diversity.get("theorem_equal", {})
    avg_ind = th.get("avg_individual", 0)
    div = th.get("diversity", 0)
    lines.append(f"- The crowd's error = average agent error ({avg_ind:.3f}) minus disagreement ({div:.3f})\n\n")

    # Shapley
    lines.append("## Attribution\n\n")
    mvp = shapley.get("most_valuable_agent", "")
    shares = shapley.get("shares", {})
    mvp_share = shares.get(mvp, 0) * 100
    lines.append(f"- Most valuable agent: **{mvp}** ({mvp_share:.0f}% of total value)\n\n")

    # Economic
    lines.append("## Economic Value\n\n")
    peaks = economic.get("peaks", {})
    primary_peak = peaks.get(PRIMARY_METHOD, {})
    vals = economic.get("values_at_r", {}).get(PRIMARY_METHOD, {})
    val_02 = vals.get("r=0.2", 0) * 100 if vals else 0
    lines.append(f"- At cost ratio 0.2, the swarm captures **{val_02:.0f}%** of perfect forecast value\n")
    lines.append(f"- Peak value: {primary_peak.get('value', 0)*100:.0f}% at r={primary_peak.get('r', 0):.2f}\n\n")

    # Robustness
    lines.append("## Robustness\n\n")
    liar = robustness.get("liar", {})
    lines.append(f"- Liar agent received weight: **{liar.get('avg_liar_weight', 0):.3f}** (vs equal weight {1/6:.3f})\n\n")

    # Calibration
    lines.append("## Calibration\n\n")
    lines.append("- When the swarm says ~40%, blooms happen ~X% of the time (see calibration table)\n\n")

    # Limitations
    n_weeks = len(method_results.get(PRIMARY_METHOD, {}).get("loo", {}).get("preds", []))
    lines.append("## Limitations\n\n")
    lines.append(f"- Based on {n_weeks} weeks of data\n")
    lines.append("- Most statistical comparisons do not reach significance at this sample size\n")
    lines.append("- Runoff data is a stand-in for real nutrient measurements\n")

    with open(output_path, "w") as f:
        f.writelines(lines)


def write_appendix_md(output_path: Path):
    """
    Write pitch/appendix.md with methodology notes.
    """
    content = """# Methodology Appendix

*One-page technical reference for judge questions*

## Aggregation Methods

1. **Base Rate**: Predicts historical bloom frequency (climatological baseline)
2. **Equal Log-Odds**: Pools agent probabilities in log-odds space with equal weights
3. **Inverse Brier (PRIMARY)**: Weights agents by inverse of their past Brier scores, pools in log-odds
4. **Bates-Granger**: Uses error covariance matrix to down-weight correlated agents
5. **Stacked Logistic**: Logistic regression on agent logits with L2 regularization
6. **Extremized**: Multiplies pooled logit by factor >1 to correct underconfidence
7. **Best Agent Selected**: Uses only the agent with best training performance
8. **Hedge Online**: Multiplicative weights based on cumulative past losses

## Why Log-Odds Pooling?

Linear pooling (averaging probabilities) can produce poorly calibrated forecasts when
agents are correlated. Log-odds pooling (averaging logits) better handles extremes
and naturally weights confident predictions appropriately.

## Evaluation Protocols

- **Leave-One-Out (LOO)**: For week t, train on all other weeks, predict week t
- **Walk-Forward (WF)**: For week t, train only on weeks before t (realistic deployment)

Both protocols prevent data leakage: predictions never use future information.

## Statistical Testing

- **Diebold-Mariano**: Tests whether loss differences are significant (with HLN correction for small samples)
- **Block Bootstrap**: Preserves autocorrelation in weekly data using blocks of 3 weeks
- **Permutation Test**: Non-parametric significance via sign flips
- **Holm Correction**: Controls family-wise error rate across all comparisons

## Shapley Values

Shapley values fairly attribute the swarm's improvement over base rate to individual agents.
Unlike leave-one-out ablation, Shapley considers each agent's contribution in all possible
coalitions, giving fairer credit when agents have overlapping information.

## Pre-Registration

**The primary method (inverse_brier_logodds) was fixed before seeing results.**
This prevents cherry-picking the best-looking method after evaluation. All other methods
are secondary comparisons for context.

## Known Limitations

1. Sample size (~16 weeks) limits statistical power
2. High agent correlation (effective N ~ 1.2) limits diversity benefits
3. Runoff data is a proxy for actual nutrient measurements
4. No external validation set (using leave-one-out instead)
5. Cost-loss model assumes fixed decision thresholds

"""
    with open(output_path, "w") as f:
        f.write(content)


def plot_reliability_diagram(
    calibration: List[Dict],
    output_path: Path,
    title: str = "Reliability Diagram",
):
    """Plot reliability diagram with Wilson intervals."""
    if not HAS_MATPLOTLIB:
        print(f"Warning: matplotlib not available, skipping {output_path}")
        return

    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)

    mean_p = [c["mean_p"] for c in calibration]
    observed = [c["observed"] for c in calibration]
    counts = [c["count"] for c in calibration]
    ci_lo = [c["ci90"][0] for c in calibration]
    ci_hi = [c["ci90"][1] for c in calibration]

    # Error bars
    yerr_lo = [o - l for o, l in zip(observed, ci_lo)]
    yerr_hi = [h - o for o, h in zip(observed, ci_hi)]

    # Filter out empty bins
    valid = [i for i, c in enumerate(counts) if c > 0]

    if valid:
        ax.errorbar(
            [mean_p[i] for i in valid],
            [observed[i] for i in valid],
            yerr=[[yerr_lo[i] for i in valid], [yerr_hi[i] for i in valid]],
            fmt='o', capsize=5, capthick=2, markersize=8, color='blue'
        )

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')

    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Observed Frequency', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_shapley_bar(
    shapley_values: Dict[str, float],
    output_path: Path,
):
    """Plot Shapley value bar chart."""
    if not HAS_MATPLOTLIB:
        print(f"Warning: matplotlib not available, skipping {output_path}")
        return

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    sorted_agents = sorted(shapley_values.items(), key=lambda x: x[1], reverse=True)
    names = [a[0] for a in sorted_agents]
    values = [a[1] for a in sorted_agents]

    colors = ['green' if v > 0 else 'red' for v in values]

    ax.barh(names, values, color=colors, alpha=0.7)
    ax.axvline(0, color='black', linewidth=0.5)

    ax.set_xlabel('Shapley Value (contribution to swarm)', fontsize=12)
    ax.set_title('Agent Attribution (Shapley Values)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_economic_curves(
    r_grid: List[float],
    curves: Dict[str, List[float]],
    output_path: Path,
):
    """Plot economic value curves."""
    if not HAS_MATPLOTLIB:
        print(f"Warning: matplotlib not available, skipping {output_path}")
        return

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    highlight = ["inverse_brier_logodds", "equal_logodds", "best_agent_selected"]

    for name, values in curves.items():
        if name in highlight:
            ax.plot(r_grid, [v * 100 for v in values], label=name, linewidth=2)
        else:
            ax.plot(r_grid, [v * 100 for v in values], label=name, alpha=0.3, linewidth=1)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Cost Ratio (C/L)', fontsize=12)
    ax.set_ylabel('Relative Economic Value (%)', fontsize=12)
    ax.set_title('Economic Value Curves', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_correlation_heatmap(
    corr_matrix: List[List[float]],
    agents: List[str],
    effective_n: float,
    output_path: Path,
):
    """Plot error correlation heatmap."""
    if not HAS_MATPLOTLIB:
        print(f"Warning: matplotlib not available, skipping {output_path}")
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

    corr = np.array(corr_matrix)
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_xticks(range(len(agents)))
    ax.set_yticks(range(len(agents)))
    ax.set_xticklabels(agents, rotation=45, ha='right')
    ax.set_yticklabels(agents)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Correlation')

    # Add values
    for i in range(len(agents)):
        for j in range(len(agents)):
            val = corr[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color='white' if abs(val) > 0.5 else 'black', fontsize=10)

    ax.set_title(f'Error Correlation Matrix (Effective N = {effective_n:.1f})', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_bss_leaderboard(
    leaderboard: List[Dict],
    output_path: Path,
):
    """Plot Brier Skill Score bar chart."""
    if not HAS_MATPLOTLIB:
        print(f"Warning: matplotlib not available, skipping {output_path}")
        return

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    sorted_lb = sorted(leaderboard, key=lambda x: x.get("bss", 0), reverse=True)
    names = [x["name"] for x in sorted_lb]
    bss = [x.get("bss", 0) * 100 for x in sorted_lb]
    kinds = [x.get("kind", "method") for x in sorted_lb]

    colors = ['steelblue' if k == 'method' else 'coral' for k in kinds]

    ax.barh(names, bss, color=colors, alpha=0.7)
    ax.axvline(0, color='black', linewidth=0.5)

    ax.set_xlabel('Brier Skill Score (%)', fontsize=12)
    ax.set_title('Leaderboard: Brier Skill Score (vs Base Rate)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.7, label='Methods'),
        Patch(facecolor='coral', alpha=0.7, label='Agents'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_degradation_curve(
    degradation: Dict,
    output_path: Path,
):
    """Plot degradation curve."""
    if not HAS_MATPLOTLIB:
        print(f"Warning: matplotlib not available, skipping {output_path}")
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

    n_agents = degradation.get("n_agents", [])
    mean_brier = degradation.get("mean_brier", [])
    std_brier = degradation.get("std_brier", [])

    ax.errorbar(n_agents, mean_brier, yerr=std_brier, fmt='o-', capsize=5)

    ax.set_xlabel('Number of Agents', fontsize=12)
    ax.set_ylabel('Brier Score', fontsize=12)
    ax.set_title('Degradation Curve: Performance vs Number of Agents', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_all_figures(
    metrics: Dict,
    diversity: Dict,
    shapley: Dict,
    economic: Dict,
    robustness: Dict,
    output_dir: Path,
):
    """Generate all figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reliability diagram for primary method
    if PRIMARY_METHOD in metrics and "loo" in metrics[PRIMARY_METHOD]:
        cal = metrics[PRIMARY_METHOD]["loo"].get("calibration", [])
        if cal:
            plot_reliability_diagram(
                cal,
                output_dir / "reliability_diagram.png",
                f"Reliability Diagram: {PRIMARY_METHOD}"
            )

    # Shapley bar chart
    if shapley.get("shapley"):
        plot_shapley_bar(shapley["shapley"], output_dir / "shapley_values.png")

    # Economic value curves
    if economic.get("curves"):
        plot_economic_curves(
            economic.get("r_grid", []),
            economic.get("curves", {}),
            output_dir / "economic_value.png"
        )

    # Correlation heatmap
    if diversity.get("corr_matrix"):
        plot_correlation_heatmap(
            diversity["corr_matrix"],
            diversity.get("agents", []),
            diversity.get("effective_n", 0),
            output_dir / "correlation_heatmap.png"
        )

    # Degradation curve
    if robustness.get("degradation"):
        plot_degradation_curve(
            robustness["degradation"],
            output_dir / "degradation_curve.png"
        )

    print(f"Generated figures in {output_dir}")
