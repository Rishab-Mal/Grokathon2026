"""Daily max temperature and precipitation from the Open-Meteo archive (no API key)."""
import pandas as pd
import requests

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def daily_weather(lat, lon, start, end):
    r = requests.get(
        ARCHIVE,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": str(start),
            "end_date": str(end),
            "daily": "temperature_2m_max,precipitation_sum",
            "timezone": "America/New_York",
        },
        timeout=60,
    )
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(d["time"]),
        "temp_max_c": d["temperature_2m_max"],
        "precip_mm": d["precipitation_sum"],
    })
    return df.set_index("date")
