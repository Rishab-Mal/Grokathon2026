#!/usr/bin/env python3
"""
Bankroll race and blind-vs-debate comparison for frontend/bankroll.html.

Every bettor starts with $100 and each week stakes its forecast against a fair coin:
its bankroll is multiplied by 2p if a bloom happened, or 2(1 - p) if not. Above $100 means
it beat a coin flip. This is the log scoring rule, so ranking by final bankroll is ranking
by log score. The swarm bets the swarm_p from scripts/swarm/aggregate.py (same leave-one-out
weights as the scoreboard); the base-rate bettor bets the leave-one-out bloom rate.

Usage:
    python scripts/agents/bankroll.py        # writes frontend/bankroll.json
"""

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from swarm.aggregate import (
    AGENT_NAMES,
    aggregate_week,
    clip_p,
    compute_base_rate,
    compute_leave_one_out_weights,
    load_agent_outputs,
    load_weeks,
)

ROOT = Path(__file__).resolve().parents[2]
START = 100.0


def score_round(all_weeks: dict, outputs_dir: Path) -> dict | None:
    """Swarm forecasts, bankroll paths and Brier scores for one round of outputs."""
    if not outputs_dir.is_dir():
        return None
    all_outputs = load_agent_outputs(outputs_dir)
    weeks = [w for w in sorted(all_weeks) if len(all_outputs.get(w, {})) == len(AGENT_NAMES)]
    if not weeks:
        return None

    forecasts = {name: [] for name in [*AGENT_NAMES, "Swarm", "Simple avg", "Base rate"]}
    for w in weeks:
        outs = all_outputs[w]
        weights = compute_leave_one_out_weights(w, all_weeks, all_outputs, AGENT_NAMES)
        agg = aggregate_week(w, all_weeks[w], outs, weights)
        for name in AGENT_NAMES:
            forecasts[name].append(outs[name]["p"])
        forecasts["Swarm"].append(agg["swarm_p"])
        forecasts["Simple avg"].append(agg["simple_avg_p"])
        forecasts["Base rate"].append(compute_base_rate(w, all_weeks))

    actual = [all_weeks[w]["actual"] for w in weeks]
    bankroll, brier = {}, {}
    for name, ps in forecasts.items():
        path = [START]
        for p, a in zip(ps, actual):
            p = clip_p(p)
            path.append(path[-1] * 2 * (p if a == 1 else 1 - p))
        bankroll[name] = [round(b, 2) for b in path]
        brier[name] = round(st.mean((p - a) ** 2 for p, a in zip(ps, actual)), 4)

    fake = any("model" not in all_outputs[w][n] for w in weeks for n in AGENT_NAMES)
    spread = st.mean(st.pstdev(all_outputs[w][n]["p"] for n in AGENT_NAMES) for w in weeks)
    moves = [abs(o["p"] - o["p_blind"]) for w in weeks for o in all_outputs[w].values() if "p_blind" in o]
    return {
        "weeks": weeks,
        "actual": actual,
        "swarm_p": [round(p, 4) for p in forecasts["Swarm"]],
        "bankroll": {n: bankroll[n] for n in [*AGENT_NAMES, "Swarm", "Base rate"]},
        "brier": brier,
        "spread": round(spread, 4),
        "avg_move": round(st.mean(moves), 4) if moves else None,
        "fake_outputs": fake,
    }


def main():
    all_weeks = load_weeks(ROOT / "data" / "weeks")
    if not all_weeks:
        sys.exit("No week files in data/weeks/")
    blind = score_round(all_weeks, ROOT / "outputs")
    if blind is None:
        sys.exit("No complete weeks in outputs/; run the agents first")
    debate = score_round(all_weeks, ROOT / "outputs_debate")

    out = {"start": START, "blind": blind, "debate": debate}
    path = ROOT / "frontend" / "bankroll.json"
    path.write_text(json.dumps(out, indent=2) + "\n")

    final = sorted(((b[-1], n) for n, b in blind["bankroll"].items()), reverse=True)
    print(f"Wrote {path.relative_to(ROOT)}  ({len(blind['weeks'])} weeks"
          f"{', fake outputs' if blind['fake_outputs'] else ''})\n\nFinal bankroll, blind round:")
    for rank, (b, n) in enumerate(final, 1):
        print(f"  {rank}. {n:<10} ${b:>12,.2f}   Brier {blind['brier'][n]:.4f}")
    if debate:
        print(f"\nDebate round: swarm Brier {blind['brier']['Swarm']:.4f} -> {debate['brier']['Swarm']:.4f}, "
              f"agent disagreement {blind['spread']:.3f} -> {debate['spread']:.3f}")
    else:
        print("\nNo debate round yet (outputs_debate/ empty).")


if __name__ == "__main__":
    main()
