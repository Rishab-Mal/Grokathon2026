#!/usr/bin/env python3
"""
Sanity-check agent outputs before aggregation.

Flags missing files, leftover fake outputs, agents that never move, and a Skeptic stuck at
the floor. The Brier scores are a quick look only; official scoring is scripts/swarm/aggregate.py.

Usage:
    python scripts/agents/check_outputs.py            # data/weeks + outputs/
    python scripts/agents/check_outputs.py --debate   # also compare outputs_debate/ with the blind round
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.personas import get_all_agent_names
from agents.run_agents import P_MIN, ROOT


def load(out_dir: Path, weeks: list[str]) -> dict[str, dict[str, dict]]:
    """{agent: {week: output}} for every saved output."""
    found = {}
    for name in get_all_agent_names():
        found[name] = {}
        for w in weeks:
            f = out_dir / f"{w}_{name}.json"
            if f.exists():
                found[name][w] = json.loads(f.read_text())
    return found


def report(title: str, found: dict, weeks: list[str], actual: dict) -> list[str]:
    print(f"\n{title}\n{'agent':<11}{'n':>4}{'mean':>7}{'min':>7}{'max':>7}{'std':>7}{'brier':>8}")
    problems = []
    for name, outs in found.items():
        missing = [w for w in weeks if w not in outs]
        fake = [w for w, o in outs.items() if "model" not in o]
        if missing:
            problems.append(f"{name}: missing {len(missing)} weeks ({', '.join(missing[:4])}...)")
        if fake:
            problems.append(f"{name}: {len(fake)} outputs are fake (no model field); run the agents")
        ps = [o["p"] for o in outs.values()]
        if not ps:
            continue
        scored = [(o["p"] - actual[w]) ** 2 for w, o in outs.items() if actual.get(w) is not None]
        brier = f"{st.mean(scored):.3f}" if scored else "-"
        sd = st.pstdev(ps)
        print(f"{name:<11}{len(ps):>4}{st.mean(ps):>7.2f}{min(ps):>7.2f}{max(ps):>7.2f}{sd:>7.2f}{brier:>8}")
        if len(ps) >= 4 and sd < 0.03:
            problems.append(f"{name}: barely moves (std {sd:.3f})")
        if sum(p <= P_MIN + 1e-9 for p in ps) > len(ps) / 2:
            problems.append(f"{name}: stuck at the {P_MIN} floor most weeks")
    return problems


def spread(found: dict, week: str) -> float | None:
    ps = [outs[week]["p"] for outs in found.values() if week in outs]
    return st.pstdev(ps) if len(ps) > 1 else None


def main():
    parser = argparse.ArgumentParser(description="Sanity-check agent outputs")
    parser.add_argument("--debate", action="store_true", help="Also check outputs_debate/")
    args = parser.parse_args()

    actual = {}
    for f in sorted((ROOT / "data" / "weeks").glob("*.json")):
        data = json.loads(f.read_text())
        actual[data.get("week", f.stem)] = data.get("actual")
    weeks = list(actual)
    if not weeks:
        sys.exit("No week files in data/weeks/")
    print(f"{len(weeks)} weeks, {sum(a == 1 for a in actual.values())} with a bloom")

    blind = load(ROOT / "outputs", weeks)
    problems = report("Blind round (outputs/)", blind, weeks, actual)

    if args.debate:
        debate = load(ROOT / "outputs_debate", weeks)
        problems += [f"debate {p}" for p in report("Debate round (outputs_debate/)", debate, weeks, actual)]
        moved = [abs(o["p"] - o["p_blind"]) for outs in debate.values() for o in outs.values() if "p_blind" in o]
        both = [w for w in weeks if spread(blind, w) is not None and spread(debate, w) is not None]
        if moved and both:
            print(f"\nAverage move after the group chat: {st.mean(moved):.3f}")
            print(f"Disagreement between agents (mean per-week std): blind {st.mean(spread(blind, w) for w in both):.3f}"
                  f" -> debate {st.mean(spread(debate, w) for w in both):.3f}  (a big drop suggests groupthink)")

    if problems:
        print("\nCheck these:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("\nLooks good.")


if __name__ == "__main__":
    main()
