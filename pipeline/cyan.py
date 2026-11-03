"""EPA CyAN satellite cyanobacteria index (Sentinel-3 OLCI) for a whole waterbody.

The API returns, per day, a 256-bin histogram of pixel DN values:
0 = below detection, 1-253 = valid CI, 254 = land, 255 = no data/cloud.
"""
from datetime import date

import pandas as pd
import requests

API = "https://cyan.epa.gov/waterbody/data/"
LAKE_ERIE_OBJECTID = 3012931

# DN -> cyanobacteria index; cells/mL = CI * 1e8
def dn_to_ci(dn):
    return 10 ** (3.0 / 250.0 * dn - 4.2)

DN_100K_CELLS = 100  # dn_to_ci(100) * 1e8 = 100,000 cells/mL (WHO moderate-risk level)


def daily_histograms(objectid, start: date, end: date):
    """{date: [256 pixel counts]}"""
    r = requests.get(
        API,
        params={
            "OBJECTID": objectid,
            "start_year": start.year, "start_day": start.timetuple().tm_yday,
            "end_year": end.year, "end_day": end.timetuple().tm_yday,
        },
        timeout=180,
    )
    r.raise_for_status()
    out = {}
    for key, counts in r.json()["data"].items():
        y, doy = map(int, key.split())
        out[(pd.Timestamp(y, 1, 1) + pd.Timedelta(days=doy - 1)).date()] = counts
    return out


def summarize(hists):
    """Aggregate several days of histograms into one set of satellite features."""
    total = [0.0] * 256
    days = 0
    for counts in hists:
        valid = sum(counts[:254])
        if valid == 0:
            continue
        days += 1
        total = [a + b for a, b in zip(total, counts)]
    valid = sum(total[:254])
    if valid == 0:
        return None
    detected = sum(total[1:254])
    high = sum(total[DN_100K_CELLS:254])
    mean_cells = (
        sum(total[dn] * dn_to_ci(dn) for dn in range(1, 254)) / detected * 1e8 if detected else 0.0
    )
    return {
        "days_with_clear_pixels": days,
        "valid_pixels": int(valid),
        "pct_pixels_cyano_detected": round(100 * detected / valid, 3),
        "pct_pixels_ge_100k_cells_ml": round(100 * high / valid, 3),
        "mean_cells_ml_where_detected": round(mean_cells),
    }
