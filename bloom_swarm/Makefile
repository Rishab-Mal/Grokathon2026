.PHONY: data fake agents swarm serve demo clean help check test figures final bankroll

# Build real week files (NOAA labels + weather + satellite) into data/weeks/
data:
	python scripts/data/build_weeks.py

# Generate fake test data (weeks + agent outputs). Overwrites data/weeks/ - run `make data` to restore.
fake:
	python scripts/make_fake_data.py

# Generate only week input files (no agent outputs)
fake-weeks:
	python scripts/make_fake_data.py --weeks_only

# Run all agents against week data (requires ANTHROPIC_API_KEY)
agents:
	python scripts/agents/run_agents.py

# Run agents with force overwrite
agents-force:
	python scripts/agents/run_agents.py --force

# Dry run agents (print prompts without API calls)
agents-dry:
	python scripts/agents/run_agents.py --dry_run

# Blind round + group-chat round (outputs_debate/)
agents-debate:
	python scripts/agents/run_agents.py --debate

# Check agent outputs: missing, fake, stuck agents
agents-check:
	python scripts/agents/check_outputs.py --debate

# Bankroll race + blind-vs-debate data for frontend/bankroll.html
bankroll:
	python scripts/agents/bankroll.py

# Aggregate agent outputs into frontend/data.json (full evaluation)
swarm:
	python scripts/swarm/aggregate.py

# Start local web server for frontend
serve:
	cd frontend && python -m http.server 8000

# Full demo: generate fake data, aggregate, and serve
demo: fake swarm bankroll serve

# Clean generated files
clean:
	rm -rf data/weeks/*.json
	rm -rf outputs/*.json
	rm -f frontend/data.json
	rm -rf pitch/

# Validate week data files
validate:
	python scripts/data/build_weeks.py --validate_only

# Run integration checks (Person 4)
check:
	python scripts/swarm/check.py

# Run tests (Person 4)
test:
	python scripts/swarm/tests/test_all.py

# Generate figures only (Person 4)
figures:
	python scripts/swarm/aggregate.py --figures
	python scripts/agents/bankroll.py

# Full final run: check + swarm + figures (Person 4)
final: check swarm bankroll

# Demo without serve: fake + swarm + figures (Person 4)
demo-no-serve: fake swarm bankroll

# Show help
help:
	@echo "Bloom Swarm - Makefile targets"
	@echo ""
	@echo "Data Generation:"
	@echo "  make fake        Generate fake test data (16 weeks + agent outputs)"
	@echo "  make fake-weeks  Generate only week input files"
	@echo "  make clean       Remove generated files"
	@echo ""
	@echo "Agent Pipeline:"
	@echo "  make agents      Run LLM agents (requires ANTHROPIC_API_KEY in .env)"
	@echo "  make agents-force  Run agents, overwriting existing outputs"
	@echo "  make agents-dry  Print prompts without calling API"
	@echo "  make bankroll    Bankroll race data for frontend/bankroll.html"
	@echo ""
	@echo "Swarm Evaluation (Person 4):"
	@echo "  make swarm       Full evaluation: aggregation + metrics + figures"
	@echo "  make check       Run integration validation checks"
	@echo "  make test        Run unit tests"
	@echo "  make figures     Generate figures only"
	@echo "  make final       Full pipeline: check + swarm + bankroll"
	@echo ""
	@echo "Frontend:"
	@echo "  make serve       Start local server at http://localhost:8000"
	@echo "  make demo        Full pipeline: fake + swarm + bankroll + serve"
	@echo "  make validate    Validate week data files"
	@echo ""
	@echo "Quick start:"
	@echo "  1. cp .env.example .env"
	@echo "  2. make demo"
	@echo "  3. Open http://localhost:8000"
