"""Build one JSON input file per week for western Lake Erie, with the actual bloom label.

    python -m pipeline.build_inputs

Writes data/inputs/week_YYYY-MM-DD.json and data/labels.csv.
"""
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from pipeline import bulletins, cyan, weather

LAKE = "Lake Erie (western basin)"
LAT, LON = 41.70, -83.25          # open water off Maumee Bay / Toledo
TARGET_YEAR = 2024
SEASON_START = date(2024, 6, 3)   # first Monday of June
SEASON_END = date(2024, 11, 3)    # last Sunday covered
HISTORY_YEARS = range(2018, 2024) # every archived season before the target year
HISTORY_SAMPLES_PER_MONTH = 4
LABEL_SAMPLES_PER_WEEK = 3        # bulletins spread across the week
HEAVY_RAIN_MM = 10.0

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "inputs"


def fetch_texts(items):
    with ThreadPoolExecutor(3) as ex:
        return list(ex.map(lambda kv: bulletins.bulletin_text(*kv), items))


def spread(seq, n):
    if len(seq) <= n:
        return list(seq)
    step = (len(seq) - 1) / (n - 1)
    return [seq[round(i * step)] for i in range(n)]


def build_history():
    """{month: {year: 0/1}} - a month is a bloom month if any sampled bulletin in it is a bloom day.
    Months with no bulletins (season not started / over) count as no bloom."""
    per_month = defaultdict(dict)
    for year in HISTORY_YEARS:
        listing = bulletins.list_bulletins(year)
        by_month = defaultdict(list)
        for d in listing:
            by_month[d.month].append(d)
        picks = [d for m in by_month for d in spread(by_month[m], HISTORY_SAMPLES_PER_MONTH)]
        texts = fetch_texts([(d, listing[d]) for d in picks])
        flags = defaultdict(int)
        for d, t in zip(picks, texts):
            if bulletins.classify(t)[0] == 1:
                flags[d.month] = 1
        for m in range(1, 13):
            per_month[m][year] = flags.get(m, 0)
    return per_month


def week_label(week_start, listing):
    days = [week_start + timedelta(i) for i in range(7)]
    items = spread([(d, listing[d]) for d in days if d in listing], LABEL_SAMPLES_PER_WEEK)
    used = []
    for (d, _), t in zip(items, fetch_texts(items)):
        bloom, area, reason = bulletins.classify(t)
        if bloom is not None:
            used.append({"date": d.isoformat(), "bloom": bloom,
                         "area_sq_mi": area if area is not None else "not stated", "reason": reason})
    if not used:
        return None
    areas = [u["area_sq_mi"] for u in used if u["area_sq_mi"] != "not stated"]
    return {
        "bloom": int(any(u["bloom"] for u in used)),
        "max_reported_area_sq_mi": max(areas) if areas else "not stated",
        "rule": (f"1 if any NOAA bulletin issued this week reports a bloom >= {bulletins.AREA_THRESHOLD:g} sq mi, "
                 "or (no area given) describes a Microcystis bloom as present/continuing; else 0. See LABELS.md"),
        "source": "NOAA NCCOS Lake Erie HAB Forecast bulletins (NCEI archive)",
        "bulletins_used": used,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("history ...")
    history = build_history()

    print("weather ...")
    wx = weather.daily_weather(LAT, LON, SEASON_START - timedelta(14), SEASON_END)

    print("satellite (EPA CyAN) ...")
    try:
        sat = cyan.daily_histograms(cyan.LAKE_ERIE_OBJECTID, SEASON_START - timedelta(7), SEASON_END)
    except Exception as e:  # satellite is optional by design
        print("  CyAN unavailable:", e)
        sat = {}

    print("labels ...")
    listing = bulletins.list_bulletins(TARGET_YEAR)

    rows = []
    week = SEASON_START
    while week + timedelta(6) <= SEASON_END:
        end = week + timedelta(6)
        label = week_label(week, listing)
        if label is None:
            print(f"  {week}: no classifiable bulletin, skipped")
            week += timedelta(7)
            continue

        w = wx.loc[str(week):str(end)]
        r14 = wx.loc[str(end - timedelta(13)):str(end)]

        sat_window = [week - timedelta(i) for i in range(7, 0, -1)]
        s = cyan.summarize([sat[d] for d in sat_window if d in sat]) if sat else None
        satellite = {
            "available": s is not None,
            "source": f"EPA CyAN, Sentinel-3 OLCI, whole Lake Erie (objectid {cyan.LAKE_ERIE_OBJECTID})",
            "window": f"{sat_window[0]} to {sat_window[-1]} (7 days before week_start, so it is not the label's own imagery)",
            **(s or {}),
            "note": ("Whole-lake pixel histogram; blooms sit in the western basin, so percentages are small but move with the bloom."
                     if s else "Satellite data unavailable for this window (clouds or CyAN API down). Use weather + history only."),
        }

        hist = history[week.month]
        rec = {
            "lake": LAKE,
            "latitude": LAT,
            "longitude": LON,
            "week_start": week.isoformat(),
            "week_end": end.isoformat(),
            "month": week.month,
            "weather": {
                "source": "Open-Meteo archive API (ERA5)",
                "temp_max_mean_c": round(float(w.temp_max_c.mean()), 1),
                "temp_max_peak_c": round(float(w.temp_max_c.max()), 1),
                "days_above_25c": int((w.temp_max_c > 25).sum()),
                "precip_week_mm": round(float(w.precip_mm.sum()), 1),
                "daily": [
                    {"date": d.date().isoformat(), "temp_max_c": float(t), "precip_mm": float(p)}
                    for d, t, p in zip(w.index, w.temp_max_c, w.precip_mm)
                ],
            },
            "runoff_proxy": {
                "rain_past_14d_mm": round(float(r14.precip_mm.sum()), 1),
                "heavy_rain_days_past_14d": int((r14.precip_mm >= HEAVY_RAIN_MM).sum()),
                "heavy_rain_threshold_mm": HEAVY_RAIN_MM,
                "note": "Stand-in for nutrient runoff: rainfall only, no streamflow or phosphorus data.",
            },
            "satellite": satellite,
            "history": {
                "month": week.month,
                "bloom_fraction_past_years": round(sum(hist.values()) / len(hist), 3),
                "years": {str(y): v for y, v in hist.items()},
                "note": f"Fraction of {min(HISTORY_YEARS)}-{max(HISTORY_YEARS)} seasons with a bloom in this calendar month (same label rule).",
            },
            "actual": label,
        }
        path = OUT / f"week_{week.isoformat()}.json"
        path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        rows.append({
            "week_start": rec["week_start"],
            "actual_bloom": label["bloom"],
            "max_area_sq_mi": label["max_reported_area_sq_mi"],
            "n_bulletins": len(label["bulletins_used"]),
            "temp_max_mean_c": rec["weather"]["temp_max_mean_c"],
            "rain_past_14d_mm": rec["runoff_proxy"]["rain_past_14d_mm"],
            "heavy_rain_days_past_14d": rec["runoff_proxy"]["heavy_rain_days_past_14d"],
            "sat_available": satellite["available"],
            "sat_pct_detected": satellite.get("pct_pixels_cyano_detected", ""),
            "history_fraction": rec["history"]["bloom_fraction_past_years"],
        })
        print(f"  {week}: bloom={label['bloom']} area={label['max_reported_area_sq_mi']}")
        week += timedelta(7)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "labels.csv", index=False)
    print(f"\n{len(df)} weekly files, {df.actual_bloom.sum()} bloom / {(df.actual_bloom == 0).sum()} no-bloom")


if __name__ == "__main__":
    main()
