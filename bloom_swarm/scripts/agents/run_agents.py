#!/usr/bin/env python3
"""
Run all agents against week input files using the xAI API (Grok).

Usage:
    python scripts/agents/run_agents.py                    # Run all weeks
    python scripts/agents/run_agents.py --week 2025-07-14  # Run single week
    python scripts/agents/run_agents.py --force            # Overwrite existing outputs
    python scripts/agents/run_agents.py --dry_run          # Print prompts without API calls
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.personas import PERSONAS, get_agent_input, get_system_prompt, get_all_agent_names


def clip_p(p: float) -> float:
    """Clip probability to [0.03, 0.97]."""
    return float(np.clip(p, 0.03, 0.97))


def prepare_agent_input(agent_name: str, week_data: dict) -> dict:
    """
    Prepare input for an agent, removing week date and actual outcome.

    This prevents the model from looking up information or leaking the answer.
    """
    # Remove fields the agent shouldn't see
    filtered = {k: v for k, v in week_data.items() if k not in ("week", "actual")}

    # Get only the keys this agent should see
    return get_agent_input(agent_name, filtered)


def parse_agent_response(response_text: str, agent_name: str) -> dict | None:
    """
    Parse JSON response from agent.

    Returns parsed dict or None if parsing fails.
    """
    try:
        # Try to extract JSON from the response
        text = response_text.strip()

        # Handle potential markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.startswith("```") and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            text = "\n".join(json_lines)

        data = json.loads(text)

        # Validate required fields
        required = ["name", "p", "reason", "evidence"]
        for field in required:
            if field not in data:
                print(f"    Warning: Missing field '{field}' in response")
                return None

        # Validate and clip p
        if not isinstance(data["p"], (int, float)):
            print(f"    Warning: 'p' is not a number: {data['p']}")
            return None

        data["p"] = clip_p(float(data["p"]))

        # Ensure name matches
        data["name"] = agent_name

        # Ensure evidence is a list
        if not isinstance(data["evidence"], list):
            data["evidence"] = [str(data["evidence"])]

        return data

    except json.JSONDecodeError as e:
        print(f"    Warning: JSON parse error: {e}")
        return None


async def run_single_agent(
    client: AsyncOpenAI,
    model: str,
    agent_name: str,
    week_data: dict,
    dry_run: bool = False,
) -> dict | None:
    """
    Run a single agent against week data.

    Returns parsed output dict or None on failure.
    """
    system_prompt = get_system_prompt(agent_name)
    agent_input = prepare_agent_input(agent_name, week_data)

    user_message = f"Analyze this data and predict bloom probability:\n\n{json.dumps(agent_input, indent=2)}"

    if dry_run:
        print(f"\n--- {agent_name} Agent ---")
        print(f"System prompt ({len(system_prompt)} chars):")
        print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)
        print(f"\nUser message:\n{user_message}")
        return None

    # Try up to 2 times (initial + 1 retry)
    for attempt in range(2):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
                max_tokens=500,
            )

            response_text = response.choices[0].message.content
            parsed = parse_agent_response(response_text, agent_name)

            if parsed:
                return parsed
            elif attempt == 0:
                print(f"    Retrying {agent_name} due to parse failure...")

        except Exception as e:
            print(f"    Error calling API for {agent_name}: {e}")
            if attempt == 0:
                print(f"    Retrying {agent_name}...")
                await asyncio.sleep(1)

    return None


async def run_week(
    client: AsyncOpenAI,
    model: str,
    week_str: str,
    week_data: dict,
    outputs_dir: Path,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, dict]:
    """
    Run all agents for a single week concurrently.

    Returns dict of {agent_name: output_dict} for successful runs.
    """
    agent_names = get_all_agent_names()
    results = {}

    # Check which agents need to run
    agents_to_run = []
    for agent_name in agent_names:
        output_path = outputs_dir / f"{week_str}_{agent_name}.json"
        if output_path.exists() and not force and not dry_run:
            print(f"  Skipping {agent_name} (output exists)")
            continue
        agents_to_run.append(agent_name)

    if not agents_to_run:
        print(f"  All agents already complete for {week_str}")
        return results

    # Run agents concurrently
    tasks = [
        run_single_agent(client, model, agent_name, week_data, dry_run)
        for agent_name in agents_to_run
    ]

    outputs = await asyncio.gather(*tasks)

    # Save results
    for agent_name, output in zip(agents_to_run, outputs):
        if output is None:
            print(f"  Failed: {agent_name}")
            continue

        results[agent_name] = output

        if not dry_run:
            output_path = outputs_dir / f"{week_str}_{agent_name}.json"
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"  Saved: {output_path.name} (p={output['p']:.2f})")

    return results


async def main_async(args):
    """Async main function."""
    # Load environment
    load_dotenv()

    api_key = os.getenv("XAI_API_KEY")
    model = os.getenv("XAI_MODEL", "grok-beta")

    if not api_key and not args.dry_run:
        print("Error: XAI_API_KEY not set in environment")
        print("Copy .env.example to .env and add your API key")
        sys.exit(1)

    # Setup paths
    weeks_dir = Path("data/weeks")
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    # Load weeks
    week_files = sorted(weeks_dir.glob("*.json"))
    if not week_files:
        print("Error: No week files found in data/weeks/")
        print("Run 'make fake' first to generate test data")
        sys.exit(1)

    # Filter to single week if specified
    if args.week:
        week_files = [f for f in week_files if f.stem == args.week]
        if not week_files:
            print(f"Error: Week {args.week} not found")
            sys.exit(1)

    print(f"Processing {len(week_files)} week(s)...")
    print(f"Model: {model}")
    if args.dry_run:
        print("DRY RUN: Not calling API\n")

    # Create client
    client = AsyncOpenAI(
        api_key=api_key or "dummy-key-for-dry-run",
        base_url="https://api.x.ai/v1",
    )

    # Process each week
    for week_file in week_files:
        with open(week_file) as f:
            week_data = json.load(f)

        week_str = week_data["week"]
        print(f"\nWeek: {week_str}")

        await run_week(
            client=client,
            model=model,
            week_str=week_str,
            week_data=week_data,
            outputs_dir=outputs_dir,
            force=args.force,
            dry_run=args.dry_run,
        )

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Run bloom prediction agents")
    parser.add_argument(
        "--week",
        type=str,
        help="Run only this week (e.g., 2025-07-14)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print prompts without calling API",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
