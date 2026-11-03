#!/usr/bin/env python3
"""
Fetch historical weather data from Open-Meteo archive API.

This script retrieves daily max temperature and precipitation data for the lake location.

API Documentation: https://open-meteo.com/en/docs/historical-weather-api

Usage:
    python scripts/data/fetch_weather.py --start 2025-06-01 --end 2025-09-30

TODO (Person 2): Complete the implementation:
1. Add error handling for API failures
2. Implement rate limiting if needed
3. Add caching to avoid repeated API calls
4. Parse the JSON response and save as CSV
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv


def build_api_url(lat: float, lon: float, start_date: str, end_date: str) -> str:
    """
    Build the Open-Meteo archive API URL.

    Args:
        lat: Latitude of the location
        lon: Longitude of the location
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        Full API URL string
    """
    base_url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,precipitation_sum",
        "timezone": "auto",
    }

    # Build query string
    query_parts = [f"{k}={v}" for k, v in params.items()]
    return f"{base_url}?{'&'.join(query_parts)}"


def fetch_weather(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """
    Fetch weather data from Open-Meteo API.

    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD

    Returns:
        API response as dict

    TODO (Person 2): Implement this function:
    1. Call the API using requests.get()
    2. Check response status code
    3. Return parsed JSON
    """
    url = build_api_url(lat, lon, start_date, end_date)
    print(f"API URL: {url}")

    # TODO: Implement the actual API call
    # response = requests.get(url)
    # response.raise_for_status()
    # return response.json()

    raise NotImplementedError("TODO: Implement API call and response parsing")


def save_to_csv(data: dict, output_path: Path):
    """
    Save weather data to CSV.

    Expected API response structure:
    {
        "daily": {
            "time": ["2025-06-01", "2025-06-02", ...],
            "temperature_2m_max": [28.5, 30.1, ...],
            "precipitation_sum": [0.0, 2.5, ...]
        }
    }

    TODO (Person 2): Implement this function:
    1. Extract time, temperature, and precipitation arrays
    2. Write to CSV with columns: date, max_temp, precipitation
    3. Handle missing data (nulls in the API response)
    """
    raise NotImplementedError("TODO: Implement CSV writing")


def aggregate_to_weeks(csv_path: Path) -> list[dict]:
    """
    Aggregate daily data to weekly summaries.

    For each week (Monday-Sunday):
    - max_temp_avg: mean of daily max temperatures
    - rain_total_mm: sum of daily precipitation
    - hot_days: count of days with max temp >= 30

    TODO (Person 2): Implement this function:
    1. Load the CSV
    2. Group by ISO week
    3. Compute aggregates
    4. Return list of dicts with week start date and aggregates
    """
    raise NotImplementedError("TODO: Implement weekly aggregation")


def main():
    parser = argparse.ArgumentParser(description="Fetch weather data from Open-Meteo")
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/weather.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    # Load environment for lake coordinates
    load_dotenv()
    lat = float(os.getenv("LAKE_LAT", "41.6833"))
    lon = float(os.getenv("LAKE_LON", "-82.3833"))

    print(f"Fetching weather for ({lat}, {lon})")
    print(f"Date range: {args.start} to {args.end}")

    # Show the URL that would be called
    url = build_api_url(lat, lon, args.start, args.end)
    print(f"\nAPI URL:\n{url}")

    print("\n" + "=" * 60)
    print("TODO (Person 2): Implement the following:")
    print("1. fetch_weather() - call the API")
    print("2. save_to_csv() - save daily data")
    print("3. aggregate_to_weeks() - compute weekly summaries")
    print("=" * 60)


if __name__ == "__main__":
    main()
