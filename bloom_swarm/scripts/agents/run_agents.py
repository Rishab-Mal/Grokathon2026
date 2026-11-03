#!/usr/bin/env python3
"""
Run all agents against week input files using the Claude API.

Reads   data/weeks/{week}.json
Writes  outputs/{week}_{Agent}.json          blind round: agents forecast independently
        outputs_debate/{week}_{Agent}.json   --debate: agents read each other's blind bets, then re-bet

Usage:
    python scripts/agents/run_agents.py                    # Run all weeks
    python scripts/agents/run_agents.py --week 2025-07-14  # Run single week
    python scripts/agents/run_agents.py --force            # Overwrite existing outputs
    python scripts/agents/run_agents.py --dry_run          # Print prompts without API calls
    python scripts/agents/run_agents.py --debate           # Also run the group-chat round

An output is skipped only if it was made from exactly the same prompt and model, so a crashed run
can simply be re-run, and new data or prompt changes redo just the weeks they affect.
Fake outputs (from make_fake_data.py) have no prompt_hash and are always replaced.
"""

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.personas import DEBATE_PROMPT, PERSONAS, get_agent_input, get_all_agent_names, get_system_prompt

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "claude-sonnet-5-5"  # or claude-haiku-5-5 via ANTHROPIC_MODEL / --model
P_MIN, P_MAX = 0.03, 0.97

# Keys that would leak the answer or the date. Never sent to an agent, at any depth.
LEAKY_KEYS = {"actual", "label", "week", "date", "year", "week_start", "week_end"}
YEAR = re.compile(r"\b(19|20)\d{2}\b")

# Structured output: the API guarantees a JSON object of this shape.
FORECAST_SCHEMA = {
    "type": "object",
    "properties": {
        "p": {"type": "number"},
        "reason": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["p", "reason", "evidence"],
    "additionalProperties": False,
}

# Errors that will fail every call the same way: stop the run instead of logging 80 failures.
FATAL_ERRORS = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
    anthropic.BadRequestError,
)


def scrub(x):
    """Drop label/date keys and redact years in strings."""
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items()
                if k.lower() not in LEAKY_KEYS and not k.lower().endswith("_date")}
    if isinstance(x, list):
        return [scrub(v) for v in x]
    if isinstance(x, str):
        return YEAR.sub("[year]", x)
    return x


def time_of_year(week: str) -> str | None:
    try:
        d = dt.date.fromisoformat(week)
    except ValueError:
        return None
    part = "early" if d.day <= 10 else "mid" if d.day <= 20 else "late"
    return f"{part} {d.strftime('%B')}"


def build_user_message(agent_name: str, week_data: dict) -> str:
    when = time_of_year(week_data["week"])
    agent_input = scrub(get_agent_input(agent_name, week_data))
    return (f"Time of year: {when}\n" if when else "") + (
        f"Your data for this week:\n{json.dumps(agent_input, indent=2)}")


def debate_message(agent_name: str, user_message: str, blind: dict[str, dict]) -> str:
    chat = "\n".join(f"- {name}: p={out['p']:.2f}. {out['reason']}"
                     for name, out in blind.items() if name != agent_name)
    own = blind[agent_name]
    return user_message + "\n\n" + DEBATE_PROMPT.format(
        chat=chat, own_p=own["p"], own_reason=own["reason"],
        debate_note=PERSONAS[agent_name]["debate_note"])


def parse_forecast(text: str, agent_name: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON object in reply: {text[:120]!r}")
    data = json.loads(match.group(0))
    p = float(data["p"])
    if p > 1:  # answered in percent
        p /= 100
    if math.isnan(p):
        raise ValueError("p is NaN")
    reason = str(data.get("reason", "")).strip()
    if not reason:
        raise ValueError("empty reason")
    evidence = data.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = [evidence]
    return {
        "name": agent_name,
        "p": round(min(max(p, P_MIN), P_MAX), 3),
        "reason": reason,
        "evidence": [str(e) for e in evidence],
    }


def request_options(model: str) -> dict:
    """Low effort keeps one-sentence forecasts fast and cheap."""
    return {"max_tokens": 1024}


async def forecast(ctx, agent_name: str, user_message: str) -> tuple[dict, str]:
    """One Claude call, retried once if the reply is unusable. Returns (forecast, serving model)."""
    err = None
    for _ in range(2):
        async with ctx.sem:
            response = await ctx.client.messages.create(
                model=ctx.model,
                system=get_system_prompt(agent_name),
                messages=[{"role": "user", "content": user_message}],
                **request_options(ctx.model),
            )
        if response.stop_reason in ("refusal", "max_tokens"):
            err = f"stop_reason={response.stop_reason}"
            continue
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return parse_forecast(text, agent_name), response.model
        except (ValueError, KeyError, TypeError) as e:
            err = e
    raise RuntimeError(f"bad reply twice: {err}")


def prompt_hash(model: str, agent_name: str, user_message: str) -> str:
    text = f"{model}\n{get_system_prompt(agent_name)}\n{user_message}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def saved_hash(path: Path) -> str | None:
    try:
        return json.loads(path.read_text()).get("prompt_hash")
    except (OSError, json.JSONDecodeError):
        return None


async def run_round(ctx, week_data: dict, out_dir: Path, mode: str, blind=None):
    """Run every agent for one week in parallel. Returns ({agent: output}, [errors])."""
    week = week_data["week"]
    results, errors = {}, []

    async def one(agent_name):
        path = out_dir / f"{week}_{agent_name}.json"
        user_message = build_user_message(agent_name, week_data)
        if mode == "debate":
            user_message = debate_message(agent_name, user_message, blind)
        digest = prompt_hash(ctx.model, agent_name, user_message)
        if not ctx.force and saved_hash(path) == digest:
            results[agent_name] = json.loads(path.read_text())
            return
        try:
            output, served_by = await forecast(ctx, agent_name, user_message)
        except FATAL_ERRORS:
            raise
        except Exception as e:  # keep going; a re-run fills the gap
            errors.append(f"{week} {agent_name} ({mode}): {e}")
            return
        output |= {"model": served_by, "mode": mode, "prompt_hash": digest}
        if mode == "debate":
            output["p_blind"] = blind[agent_name]["p"]
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2) + "\n")
        results[agent_name] = output
        print(f"  [{week}] {mode:<6} {agent_name:<10} p={output['p']:.2f}  {output['reason'][:90]}")

    await asyncio.gather(*(one(name) for name in get_all_agent_names()))
    return results, errors


async def run_week(ctx, week_data: dict) -> list[str]:
    blind, errors = await run_round(ctx, week_data, ctx.outputs_dir, "blind")
    if ctx.debate:
        if len(blind) == len(PERSONAS):
            _, debate_errors = await run_round(ctx, week_data, ctx.debate_dir, "debate", blind)
            errors += debate_errors
        else:
            errors.append(f"{week_data['week']}: debate round skipped, blind round incomplete")
    return errors


def load_weeks(weeks_dir: Path, only_week: str | None) -> list[dict]:
    week_files = sorted(weeks_dir.glob("*.json"))
    if only_week:
        week_files = [f for f in week_files if f.stem == only_week]
    if not week_files:
        print(f"Error: no week files found in {weeks_dir}" + (f" for {only_week}" if only_week else ""))
        sys.exit(1)
    weeks = []
    for f in week_files:
        data = json.loads(f.read_text())
        data.setdefault("week", f.stem)
        weeks.append(data)
    return weeks


async def main_async(args):
    load_dotenv(ROOT / ".env")
    weeks = load_weeks(ROOT / "data" / "weeks", args.week)

    if args.dry_run:
        print("DRY RUN: not calling the API")
        for agent_name in get_all_agent_names():
            print(f"\n===== {agent_name} ({weeks[0]['week']}) =====")
            print(get_system_prompt(agent_name))
            print("--- user ---\n" + build_user_message(agent_name, weeks[0]))
        return

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set. Copy .env.example to .env and add the key.")
        sys.exit(1)

    ctx = argparse.Namespace(
        model=args.model or os.getenv("ANTHROPIC_MODEL") or DEFAULT_MODEL,
        force=args.force,
        debate=args.debate,
        outputs_dir=ROOT / "outputs",
        debate_dir=ROOT / "outputs_debate",
        client=anthropic.AsyncAnthropic(max_retries=4, timeout=120),
        sem=asyncio.Semaphore(args.concurrency),
    )
    print(f"Model: {ctx.model} | {len(weeks)} weeks x {len(PERSONAS)} agents"
          f"{' x 2 rounds (blind + debate)' if ctx.debate else ''}")

    try:
        results = await asyncio.gather(*(run_week(ctx, w) for w in weeks))
    except anthropic.APIStatusError as e:
        print(f"\nClaude API error {e.status_code}: {e.message}")
        sys.exit(1)

    errors = [e for errs in results for e in errs]
    if errors:
        print(f"\n{len(errors)} failed (re-run to fill them in):\n  " + "\n  ".join(errors))
        sys.exit(1)
    print("\nDone. All outputs saved.")


def main():
    parser = argparse.ArgumentParser(description="Run bloom prediction agents (Claude API)")
    parser.add_argument("--week", type=str, help="Run only this week (e.g., 2025-07-14)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--dry_run", action="store_true", help="Print prompts without calling API")
    parser.add_argument("--debate", action="store_true", help="Also run the group-chat round into outputs_debate/")
    parser.add_argument("--model", type=str, help=f"Claude model (default: $ANTHROPIC_MODEL or {DEFAULT_MODEL})")
    parser.add_argument("--concurrency", type=int, default=8, help="Max parallel API calls")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
