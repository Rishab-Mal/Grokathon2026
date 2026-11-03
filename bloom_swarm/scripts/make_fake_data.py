#!/usr/bin/env python3
"""
Generate fake weekly input data and agent outputs for testing.

Usage:
    python scripts/make_fake_data.py              # Generate weeks + agent outputs
    python scripts/make_fake_data.py --weeks_only # Generate only week input files
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

SEED = 42
AGENT_NAMES = ["Weather", "Runoff", "Satellite", "Historian", "Skeptic"]

# Fake reasons templates
WEATHER_REASONS = [
    "Six days above 30C favour cyanobacteria growth.",
    "High average temperatures accelerate algal metabolism.",
    "Warm conditions with minimal rain create stagnant water.",
    "Sustained heat promotes surface stratification.",
    "Hot spell combined with low precipitation increases bloom risk.",
]

RUNOFF_REASONS = [
    "Heavy recent rainfall washed nutrients into the lake.",
    "Moderate runoff in past two weeks adds phosphorus load.",
    "Low rainfall means reduced nutrient input from land.",
    "Multiple heavy rain days increased agricultural runoff.",
    "Dry period limits nutrient transport to water body.",
]

SATELLITE_REASONS = [
    "Satellite detects elevated cyanobacteria concentrations.",
    "Cyano index rising sharply from last week.",
    "Current satellite readings show moderate algae presence.",
    "Index stable but already at concerning levels.",
    "Low cyano index suggests minimal bloom activity.",
]

HISTORIAN_REASONS = [
    "Historical data shows blooms common this month.",
    "Past years rarely had blooms in early summer.",
    "This period historically sees peak bloom frequency.",
    "Historical bloom rate for this month is moderate.",
    "Records indicate low bloom probability for this season.",
]

SKEPTIC_REASONS = [
    "Wind and currents may disperse any forming blooms.",
    "Cooler nights could slow algal growth rates.",
    "Recent water mixing reduces surface accumulation.",
    "Natural predators may control cyanobacteria population.",
    "Lake depth provides thermal refuge against blooms.",
]

EVIDENCE_POOL = {
    "Weather": [
        "Max temp averaged {temp:.1f}C this week.",
        "{hot} days exceeded 30C threshold.",
        "Only {rain:.1f}mm of rain recorded.",
        "UV index remained high throughout week.",
        "No significant cooling events observed.",
    ],
    "Runoff": [
        "{rain14:.0f}mm fell in the past 14 days.",
        "{heavy} days had heavy precipitation (>10mm).",
        "Nearby rivers showed elevated turbidity.",
        "Agricultural fields are in active growing season.",
        "Storm drains discharged into tributaries.",
    ],
    "Satellite": [
        "Cyano index at {cyano:.2f}.",
        "Index changed by {change:+.2f} vs last week.",
        "Surface reflectance indicates algal pigments.",
        "Thermal bands show warm surface layer.",
        "Chlorophyll-a estimates support reading.",
    ],
    "Historian": [
        "Bloom rate this month in past years: {rate:.0%}.",
        "Similar conditions caused blooms in 3 of 5 prior years.",
        "Last year had {'a bloom' if rate > 0.5 else 'no bloom'} this week.",
        "Long-term trend shows {'increasing' if rate > 0.4 else 'stable'} frequency.",
        "Seasonal pattern peaks in {'August' if rate > 0.5 else 'July'}.",
    ],
    "Skeptic": [
        "Wind speeds averaged 15 km/h, promoting mixing.",
        "Lake level is higher than usual, diluting nutrients.",
        "Water temperature still below optimal for blooms.",
        "Zebra mussels may filter some algae.",
        "Recent cold front disrupted stratification.",
    ],
}


def clip_p(p: float) -> float:
    """Clip probability to [0.03, 0.97]."""
    return float(np.clip(p, 0.03, 0.97))


def generate_weeks(rng: np.random.Generator, n_weeks: int = 16) -> list[dict]:
    """Generate n_weeks consecutive weekly input data starting June 2025."""
    start_date = datetime(2025, 6, 2)  # First Monday in June 2025
    weeks = []

    # Monthly base rates
    month_base_rates = {6: 0.35, 7: 0.55, 8: 0.65, 9: 0.45}

    for i in range(n_weeks):
        week_date = start_date + timedelta(weeks=i)
        week_str = week_date.strftime("%Y-%m-%d")
        month = week_date.month

        # Weather data
        base_temp = 26 + (month - 6) * 2  # Warmer in summer
        max_temp_avg = base_temp + rng.normal(0, 3)
        max_temp_avg = float(np.clip(max_temp_avg, 18, 38))

        rain_total_mm = float(max(0, rng.exponential(15)))
        hot_days = int(np.clip(rng.poisson(max(0, (max_temp_avg - 26) / 2)), 0, 7))

        weather = {
            "max_temp_avg": round(max_temp_avg, 1),
            "rain_total_mm": round(rain_total_mm, 1),
            "hot_days": hot_days,
        }

        # Runoff data (correlated with recent rain)
        rain_last_14d_mm = float(rain_total_mm + max(0, rng.exponential(30)))
        heavy_rain_days = int(np.clip(rng.poisson(rain_last_14d_mm / 25), 0, 7))

        runoff = {
            "rain_last_14d_mm": round(rain_last_14d_mm, 0),
            "heavy_rain_days": heavy_rain_days,
        }

        # Satellite data (occasionally null, ~10% missing)
        if rng.random() < 0.1:
            satellite = None
        else:
            # Cyano index correlated with heat and runoff
            base_cyano = 0.2 + (max_temp_avg - 25) * 0.02 + rain_last_14d_mm * 0.002
            cyano_index = float(np.clip(base_cyano + rng.normal(0, 0.1), 0, 1))
            change = float(np.clip(rng.normal(0, 0.08), -0.3, 0.3))

            satellite = {
                "cyano_index": round(cyano_index, 2),
                "change_vs_last_week": round(change, 2),
            }

        # Historical base rate
        bloom_rate = month_base_rates.get(month, 0.4) + rng.normal(0, 0.05)
        bloom_rate = float(np.clip(bloom_rate, 0.1, 0.9))

        history = {"bloom_rate_this_month_past_years": round(bloom_rate, 2)}

        # Actual outcome: correlate with heat, rain, cyano
        # Higher temp, more runoff, higher cyano = higher bloom chance
        bloom_score = 0.0
        bloom_score += (max_temp_avg - 25) * 0.08  # Heat contribution
        bloom_score += rain_last_14d_mm * 0.005  # Nutrient runoff
        if satellite:
            bloom_score += satellite["cyano_index"] * 0.6  # Satellite is most informative
        bloom_score += bloom_rate * 0.3  # Historical tendency
        bloom_score += rng.normal(0, 0.15)  # Random noise

        bloom_prob = 1 / (1 + np.exp(-bloom_score + 1))  # Sigmoid with offset
        actual = int(rng.random() < bloom_prob)

        week_data = {
            "week": week_str,
            "weather": weather,
            "runoff": runoff,
            "satellite": satellite,
            "history": history,
            "actual": actual,
        }
        weeks.append(week_data)

    return weeks


def generate_agent_output(
    agent_name: str, week_data: dict, rng: np.random.Generator
) -> dict:
    """Generate fake agent output based on week data."""
    weather = week_data["weather"]
    runoff = week_data["runoff"]
    satellite = week_data["satellite"]
    history = week_data["history"]
    actual = week_data["actual"]

    # Base probability on relevant data + noise + correlation with actual
    # Agent accuracy varies: Satellite best, Weather/Runoff moderate, Historian close to base, Skeptic low

    if agent_name == "Weather":
        # Moderately informative
        base_p = 0.3 + (weather["max_temp_avg"] - 25) * 0.03 + weather["hot_days"] * 0.04
        base_p += actual * 0.15 + rng.normal(0, 0.12)
        reasons = WEATHER_REASONS
        evidence_templates = EVIDENCE_POOL["Weather"]

    elif agent_name == "Runoff":
        # Moderately informative
        base_p = 0.25 + runoff["rain_last_14d_mm"] * 0.004 + runoff["heavy_rain_days"] * 0.05
        base_p += actual * 0.15 + rng.normal(0, 0.12)
        reasons = RUNOFF_REASONS
        evidence_templates = EVIDENCE_POOL["Runoff"]

    elif agent_name == "Satellite":
        # Most informative
        if satellite:
            base_p = 0.2 + satellite["cyano_index"] * 0.7 + satellite["change_vs_last_week"] * 0.3
            base_p += actual * 0.2 + rng.normal(0, 0.08)
        else:
            # No satellite data: fall back to base rate
            base_p = history["bloom_rate_this_month_past_years"] + rng.normal(0, 0.1)
        reasons = SATELLITE_REASONS
        evidence_templates = EVIDENCE_POOL["Satellite"]

    elif agent_name == "Historian":
        # Close to base rate
        base_p = history["bloom_rate_this_month_past_years"] + rng.normal(0, 0.08)
        base_p += actual * 0.05  # Slight correlation
        reasons = HISTORIAN_REASONS
        evidence_templates = EVIDENCE_POOL["Historian"]

    elif agent_name == "Skeptic":
        # Biased low
        base_p = 0.25 + rng.normal(0, 0.1)
        base_p += actual * 0.1  # Weak correlation
        reasons = SKEPTIC_REASONS
        evidence_templates = EVIDENCE_POOL["Skeptic"]

    else:
        base_p = 0.5
        reasons = ["Unknown agent."]
        evidence_templates = ["No data available."]

    p = clip_p(base_p)
    reason = rng.choice(reasons)

    # Format evidence with actual values
    evidence = []
    selected_templates = rng.choice(evidence_templates, size=min(3, len(evidence_templates)), replace=False)
    for tmpl in selected_templates:
        try:
            formatted = tmpl.format(
                temp=weather["max_temp_avg"],
                hot=weather["hot_days"],
                rain=weather["rain_total_mm"],
                rain14=runoff["rain_last_14d_mm"],
                heavy=runoff["heavy_rain_days"],
                cyano=satellite["cyano_index"] if satellite else 0,
                change=satellite["change_vs_last_week"] if satellite else 0,
                rate=history["bloom_rate_this_month_past_years"],
            )
            evidence.append(formatted)
        except (KeyError, TypeError):
            evidence.append(tmpl.split("{")[0].strip(".") + ".")

    return {
        "name": agent_name,
        "p": round(p, 2),
        "reason": reason,
        "evidence": evidence,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate fake bloom data")
    parser.add_argument(
        "--weeks_only",
        action="store_true",
        help="Only generate week input files, not agent outputs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Ensure directories exist
    weeks_dir = Path("data/weeks")
    outputs_dir = Path("outputs")
    weeks_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Generate weeks
    weeks = generate_weeks(rng)
    print(f"Generating {len(weeks)} weeks of data...")

    for week_data in weeks:
        week_str = week_data["week"]
        week_path = weeks_dir / f"{week_str}.json"
        with open(week_path, "w") as f:
            json.dump(week_data, f, indent=2)
        print(f"  Created {week_path}")

    if args.weeks_only:
        print("Done (weeks only).")
        return

    # Generate agent outputs
    print(f"\nGenerating agent outputs for {len(AGENT_NAMES)} agents...")
    for week_data in weeks:
        week_str = week_data["week"]
        for agent_name in AGENT_NAMES:
            output = generate_agent_output(agent_name, week_data, rng)
            output_path = outputs_dir / f"{week_str}_{agent_name}.json"
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)

    print(f"  Created {len(weeks) * len(AGENT_NAMES)} agent output files")
    print("Done.")


if __name__ == "__main__":
    main()
