# Data and label rules (for the pitch)

## Lake: western Lake Erie, not Lake Lanier

Lake Lanier has no usable history of bloom advisories. Georgia EPD only samples when someone reports a suspected
bloom. The public record is a handful of one-off alerts (Nov 2020, Nov 2024, summer 2026), mostly from the
Riverkeeper or the lake association rather than EPD. That is nowhere near 15 labelled weeks.

Western Lake Erie has NOAA's HAB Forecast bulletin, issued several times a week every summer. NCEI archives it for
2018–2024.

## Label rule

A **bulletin** counts as a bloom day if:

- it reports a bloom area of **at least 50 sq mi**, or
- it gives no area (clouds or wind) but describes a Microcystis bloom as present or continuing in the western basin.

A bulletin counts as no bloom if it reports an area under 50 sq mi, or says cyanobacteria are "not currently
detectable", only traces, at low or relatively low concentrations, small patches, an early-season projection only,
or that the bloom has ended.

Bulletins that fit neither case are skipped.

A **week** (Monday to Sunday) is labelled **1** if any of up to 3 bulletins sampled across that week is a bloom day.
Otherwise it is labelled **0**. Sampling keeps the number of downloads down, because NCEI throttles bulk requests.

The 50 sq mi cut-off separates "a bloom is starting or fading" from "an established bloom". Early-season and
late-season bulletins often report 10–30 sq mi.

Each input file lists the bulletins behind its label (`actual.bulletins_used`), with the date, area and matched
phrase, so anyone can spot-check it. The code is in `pipeline/bulletins.py`, in `classify`.

## Inputs

| Field | Source | Notes |
|---|---|---|
| weather | Open-Meteo archive API at 41.70, -83.25 | daily `temperature_2m_max`, `precipitation_sum`, aggregated to the week |
| runoff_proxy | the same rainfall data | **This is a stand-in.** It uses rain over the past 14 days and the number of days with more than 10 mm of rain. There is no streamflow or phosphorus data. |
| satellite | EPA CyAN API, Sentinel-3 OLCI, whole Lake Erie | covers the **7 days before** the week starts, so the model never sees the imagery the label came from. The agents get `cyano_index`, the share of detected pixels divided by the western basin's 12.7% of the lake area: roughly the fraction of the western basin with detectable cyanobacteria (0.003 in early June, about 0.3 at peak). |
| history | the same bulletins, 2018–2023 | for each calendar month, the fraction of past seasons with a bloom that month under the same rule. Months with no bulletins count as no bloom. |

## Run it

```
pip install -r requirements.txt
python -m pipeline.build_inputs
```

The build writes `data/inputs/week_*.json` and a summary in `data/labels.csv`. Bulletin text is cached in
`data/raw/bulletin_text/`.
