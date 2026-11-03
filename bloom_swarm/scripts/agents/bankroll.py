#!/usr/bin/env python3
"""
Bankroll race and blind-vs-debate comparison for frontend/bankroll.html.

Every bettor starts with $100 and each week stakes its forecast against a fair coin:
its bankroll is multiplied by 2p if a bloom happened, or 2(1 - p) if not. Above $100 means
it beat a coin flip. This is the log scoring rule, so ranking by final bankroll is ranking
by log score. The swarm, simple average and base rate use the swarm package's leave-one-out
predictions (the primary method from scripts/swarm/config.json), the same numbers as the
scoreboard in frontend/data.json.

Usage:
    python scripts/agents/bankroll.py        # writes frontend/bankroll.json
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from swarm.aggregators import PRIMARY_METHOD
from swarm.core import clip, load_data
from swarm.protocols import leave_one_out

ROOT = Path(__file__).resolve().parents[2]
START = 100.0
AGENT_NAMES = ["Weather", "Runoff", "Satellite", "Historian", "Skeptic"]
POOLED = {"Swarm": PRIMARY_METHOD, "Simple avg": "equal_logodds", "Base rate": "base_rate"}


def score_round(weeks_dir: Path, outputs_dir: Path) -> dict | None:
    """Swarm forecasts, bankroll paths and Brier scores for one round of outputs."""
    if not outputs_dir.is_dir():
        return None
    weeks, agents, P, y, _ = load_data(weeks_dir, outputs_dir)
    if not set(AGENT_NAMES) <= set(agents):
        return None
    P = P[:, [agents.index(n) for n in AGENT_NAMES]]
    complete = ~np.isnan(P).any(axis=1)
    if not complete.any():
        return None
    weeks = [w for w, ok in zip(weeks, complete) if ok]
    P, y = P[complete], y[complete]

    forecasts = {name: P[:, k] for k, name in enumerate(AGENT_NAMES)}
    for label, method in POOLED.items():
        forecasts[label], _ = leave_one_out(method, P, y, AGENT_NAMES)

    bankroll, brier = {}, {}
    for name, ps in forecasts.items():
        ps = clip(np.asarray(ps, dtype=float))
        stakes = 2 * np.where(y == 1, ps, 1 - ps)
        bankroll[name] = [round(float(b), 2) for b in START * np.concatenate([[1.0], np.cumprod(stakes)])]
        brier[name] = round(float(np.mean((ps - y) ** 2)), 4)

    raw = [json.loads((outputs_dir / f"{w}_{n}.json").read_text()) for w in weeks for n in AGENT_NAMES]
    moves = [abs(o["p"] - o["p_blind"]) for o in raw if "p_blind" in o]
    return {
        "weeks": weeks,
        "actual": [int(a) for a in y],
        "swarm_p": [round(float(p), 4) for p in forecasts["Swarm"]],
        "bankroll": {n: bankroll[n] for n in [*AGENT_NAMES, "Swarm", "Base rate"]},
        "brier": brier,
        "spread": round(float(np.std(P, axis=1).mean()), 4),
        "avg_move": round(float(np.mean(moves)), 4) if moves else None,
        "fake_outputs": any("model" not in o for o in raw),
    }


def main():
    weeks_dir = ROOT / "data" / "weeks"
    if not any(weeks_dir.glob("*.json")):
        sys.exit("No week files in data/weeks/")
    blind = score_round(weeks_dir, ROOT / "outputs")
    if blind is None:
        sys.exit("No complete weeks in outputs/; run the agents first")
    debate = score_round(weeks_dir, ROOT / "outputs_debate")

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
