#!/usr/bin/env python3
"""
Build weekly input files from weather, satellite, and historical data.

This script combines multiple data sources into the input_week schema format
and writes one JSON file per week to data/weeks/.

Usage:
    python scripts/data/build_weeks.py

TODO (Person 2): Complete the implementation:
1. Load weather data (from fetch_weather.py output)
2. Load or mock satellite data
3. Load or compute historical bloom rates
4. Add actual bloom labels (manual or from another source)
5. Validate and write week files
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# Required keys per schema section
SCHEMA_KEYS = {
    "weather": ["max_temp_avg", "rain_total_mm", "hot_days"],
    "runoff": ["rain_last_14d_mm", "heavy_rain_days"],
    "satellite": ["cyano_index", "change_vs_last_week"],  # Can be null
    "history": ["bloom_rate_this_month_past_years"],
}


def validate_week(week_data: dict) -> list[str]:
    """
    Validate a week data dict against the input_week schema.

    Args:
        week_data: Dict to validate

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check top-level required keys
    required_top = ["week", "weather", "runoff", "history", "actual"]
    for key in required_top:
        if key not in week_data:
            errors.append(f"Missing required key: {key}")

    # Validate week format
    if "week" in week_data:
        try:
            datetime.strptime(week_data["week"], "%Y-%m-%d")
        except ValueError:
            errors.append(f"Invalid week format: {week_data['week']} (expected YYYY-MM-DD)")

    # Validate nested sections
    for section, keys in SCHEMA_KEYS.items():
        if section not in week_data:
            continue

        # satellite can be null
        if section == "satellite" and week_data[section] is None:
            continue

        if not isinstance(week_data[section], dict):
            errors.append(f"{section} must be a dict or null")
            continue

        for key in keys:
            if key not in week_data[section]:
                errors.append(f"Missing {section}.{key}")

    # Validate actual
    if "actual" in week_data:
        if week_data["actual"] not in (0, 1):
            errors.append(f"actual must be 0 or 1, got {week_data['actual']}")

    return errors


def load_weather_data(weather_csv: Path) -> dict[str, dict]:
    """
    Load weather data from CSV and return dict keyed by week date.

    TODO (Person 2): Implement this function:
    1. Read the CSV file (date, max_temp, precipitation columns)
    2. Group by ISO week
    3. Compute: max_temp_avg, rain_total_mm, hot_days (>= 30C)
    4. Return {week_date_str: weather_dict}
    """
    raise NotImplementedError("TODO: Load and aggregate weather CSV")


def load_satellite_data(satellite_path: Path) -> dict[str, dict | None]:
    """
    Load satellite data, returning dict keyed by week date.

    TODO (Person 2): Implement this function:
    1. Load satellite data (format TBD - could be CSV, JSON, or API)
    2. Return {week_date_str: {"cyano_index": float, "change_vs_last_week": float}}
    3. Return None for weeks with no satellite data
    """
    raise NotImplementedError("TODO: Load satellite data")


def compute_historical_rates(month: int) -> float:
    """
    Return historical bloom rate for a given month.

    TODO (Person 2): Implement this function:
    Could be:
    - Hard-coded rates from literature
    - Computed from historical bloom records
    - Loaded from a data file
    """
    # Placeholder values based on typical Great Lakes patterns
    rates = {
        6: 0.35,  # June
        7: 0.55,  # July
        8: 0.65,  # August
        9: 0.45,  # September
    }
    return rates.get(month, 0.4)


def compute_runoff(weather_data: dict[str, dict], week_str: str) -> dict:
    """
    Compute runoff metrics from weather data.

    For runoff, we need rainfall over the past 14 days, which spans
    the current week plus the previous week.

    TODO (Person 2): Implement this function:
    1. Sum precipitation over the past 14 days
    2. Count days with > 10mm precipitation
    3. Return {"rain_last_14d_mm": float, "heavy_rain_days": int}
    """
    raise NotImplementedError("TODO: Compute runoff from weather data")


def build_week_file(
    week_str: str,
    weather: dict,
    runoff: dict,
    satellite: dict | None,
    history: dict,
    actual: int,
) -> dict:
    """
    Assemble a complete week data dict.

    Args:
        week_str: Week date string (YYYY-MM-DD, should be a Monday)
        weather: Weather data dict
        runoff: Runoff data dict
        satellite: Satellite data dict or None
        history: Historical data dict
        actual: 1 if bloom occurred, 0 otherwise

    Returns:
        Complete week data dict matching input_week schema
    """
    week_data = {
        "week": week_str,
        "weather": weather,
        "runoff": runoff,
        "satellite": satellite,
        "history": history,
        "actual": actual,
    }

    # Validate before returning
    errors = validate_week(week_data)
    if errors:
        raise ValueError(f"Invalid week data for {week_str}: {errors}")

    return week_data


def main():
    parser = argparse.ArgumentParser(description="Build weekly input files")
    parser.add_argument(
        "--weather",
        type=str,
        default="data/weather.csv",
        help="Path to weather CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/weeks",
        help="Output directory for week JSON files",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only validate existing week files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        # Validate existing files
        print("Validating existing week files...")
        week_files = sorted(output_dir.glob("*.json"))

        if not week_files:
            print("No week files found.")
            return

        all_valid = True
        for path in week_files:
            with open(path) as f:
                data = json.load(f)

            errors = validate_week(data)
            if errors:
                print(f"INVALID: {path.name}")
                for err in errors:
                    print(f"  - {err}")
                all_valid = False
            else:
                print(f"VALID: {path.name}")

        if all_valid:
            print(f"\nAll {len(week_files)} files are valid.")
        else:
            print("\nSome files have validation errors.")
            sys.exit(1)
        return

    # The real pipeline lives in the repo-root `pipeline` package (NOAA bulletin labels,
    # Open-Meteo weather, EPA CyAN satellite) and writes data/weeks/ in this schema.
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from pipeline import build_inputs
    build_inputs.main()


if __name__ == "__main__":
    main()
