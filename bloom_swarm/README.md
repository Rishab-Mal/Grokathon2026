# Bloom Swarm

A swarm of LLM agents (Claude via the Anthropic API) that predict harmful algal blooms on lakes. Each agent analyzes different data sources, and their predictions are combined using accuracy-weighted aggregation. The system includes backtesting against historical outcomes and a web frontend for visualization.

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env to add your ANTHROPIC_API_KEY (optional for demo)

# 4. Run demo with fake data
make demo
# Opens http://localhost:8000
```

## Real data run

`data/weeks/` holds 21 real weeks of western Lake Erie data (June to October 2024). The labels come from NOAA
HAB bulletins; see `../LABELS.md` for the rule. To run without `make` (for example on Windows):

```bash
python scripts/data/build_weeks.py          # rebuild data/weeks/ (bulletin text is cached)
python scripts/agents/run_agents.py         # needs ANTHROPIC_API_KEY in .env
python scripts/agents/check_outputs.py
python scripts/swarm/aggregate.py
cd frontend && python -m http.server 8000
```

`make fake` / `make demo` overwrite `data/weeks/` with synthetic data. Run `make data` to restore the real weeks.

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make data` | Build real week files from NOAA bulletins, Open-Meteo and EPA CyAN |
| `make fake` | Generate 16 weeks of fake test data + agent outputs |
| `make fake-weeks` | Generate only week input files (no agent outputs) |
| `make agents` | Run LLM agents against week data (requires ANTHROPIC_API_KEY); replaces fake outputs |
| `make agents-debate` | Blind round, then a group-chat round where agents read each other's bets (`outputs_debate/`) |
| `make agents-check` | Flag missing or fake outputs and agents stuck at one value |
| `make agents-force` | Run agents, overwriting existing outputs |
| `make agents-dry` | Print prompts without calling the API |
| `make swarm` | Aggregate outputs into `frontend/data.json` |
| `make serve` | Start local server at http://localhost:8000 |
| `make demo` | Full pipeline: fake + swarm + serve |
| `make clean` | Remove all generated files |
| `make validate` | Validate week data files against schema |

## Schemas

### Schema 1: Input Week (`data/weeks/{week}.json`)

```json
{
  "week": "2025-07-14",
  "weather": {"max_temp_avg": 31.2, "rain_total_mm": 4.1, "hot_days": 6},
  "runoff": {"rain_last_14d_mm": 62, "heavy_rain_days": 2},
  "satellite": {"cyano_index": 0.41, "change_vs_last_week": 0.09},
  "history": {"bloom_rate_this_month_past_years": 0.55},
  "actual": 1
}
```

Notes:
- `week` is the Monday of the week (ISO format)
- `satellite` may be `null` if data unavailable
- `actual` is 1 if bloom occurred, 0 otherwise

### Schema 2: Agent Output (`outputs/{week}_{name}.json`)

```json
{
  "name": "Weather",
  "p": 0.62,
  "reason": "Six days above 30C favour cyanobacteria growth.",
  "evidence": ["Max temp averaged 31.2C", "Only 4.1mm rain recorded"]
}
```

Agent names: `Weather`, `Runoff`, `Satellite`, `Historian`, `Skeptic`

Real agent runs also write `model`, `mode` (`blind` or `debate`; debate files add `p_blind`) and `prompt_hash` (a week is re-run when its data or prompt changes). A file without `model` came from `make_fake_data.py`. `outputs_debate/` uses the same format for the group-chat round.

### Schema 3: Final Output (`frontend/data.json`)

```json
{
  "lake": {"name": "Lake Erie", "lat": 41.6833, "lon": -82.3833},
  "weeks": [
    {
      "week": "2025-07-14",
      "agents": [{"name": "Weather", "p": 0.62, "reason": "...", "evidence": [...], "weight": 0.21}],
      "swarm_p": 0.58,
      "simple_avg_p": 0.55,
      "base_rate_p": 0.40,
      "actual": 1
    }
  ],
  "scores": {
    "Weather": 0.21, "Runoff": 0.24, "Satellite": 0.18,
    "Historian": 0.23, "Skeptic": 0.26,
    "swarm": 0.15, "simple_avg": 0.17, "base_rate": 0.24
  }
}
```

Notes:
- `weeks` sorted by date
- `scores` are Brier scores (lower is better)
- Each agent includes its `weight` for that week (computed via leave-one-out)

## Project Structure

```
bloom_swarm/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── Makefile
├── schemas/
│   ├── input_week.schema.json
│   ├── agent_output.schema.json
│   └── final.schema.json
├── data/
│   └── weeks/           # Input files (e.g., 2025-07-14.json)
├── outputs/             # Agent outputs (e.g., 2025-07-14_Weather.json)
├── scripts/
│   ├── make_fake_data.py
│   ├── data/
│   │   ├── fetch_weather.py
│   │   └── build_weeks.py
│   ├── agents/
│   │   ├── personas.py
│   │   └── run_agents.py
│   └── swarm/
│       └── aggregate.py
└── frontend/
    ├── index.html
    └── data.json
```

## Team Ownership

| Person | Area | Files |
|--------|------|-------|
| Person 1 | Frontend | `frontend/` |
| Person 2 | Data Pipeline | `scripts/data/`, `data/weeks/` |
| Person 3 | Agents | `scripts/agents/` |
| Person 4 | Swarm Logic | `scripts/swarm/`, `scripts/make_fake_data.py` |

## How It Works

1. **Data Collection** (Person 2): Weather, satellite, and historical data are collected and formatted into weekly input files.

2. **Agent Prediction** (Person 3): Each agent sees only its relevant data slice and outputs a bloom probability with reasoning.

3. **Swarm Aggregation** (Person 4): Agent predictions are combined using weights based on each agent's historical accuracy (leave-one-out Brier scores).

4. **Visualization** (Person 1): The frontend displays predictions, agent cards, and backtesting results.

## Aggregation Method

For each week t:
1. Compute each agent's Brier score on all OTHER weeks (leave-one-out)
2. Weight = 1 / Brier score (normalized to sum to 1)
3. `swarm_p` = sigmoid(Σ weight × logit(p))
4. Compare against `simple_avg_p` and `base_rate_p` baselines

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key | (required for agents) |
| `ANTHROPIC_MODEL` | Claude model | `claude-sonnet-5-5` (or `claude-haiku-5-5`) |
| `LAKE_NAME` | Lake name for display | `Lake Erie` |
| `LAKE_LAT` | Lake latitude | `41.6833` |
| `LAKE_LON` | Lake longitude | `-82.3833` |
