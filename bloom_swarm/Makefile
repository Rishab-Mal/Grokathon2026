.PHONY: fake agents swarm serve demo clean help

# Generate fake test data (weeks + agent outputs)
fake:
	python scripts/make_fake_data.py

# Generate only week input files (no agent outputs)
fake-weeks:
	python scripts/make_fake_data.py --weeks_only

# Run all agents against week data (requires XAI_API_KEY)
agents:
	python scripts/agents/run_agents.py

# Run agents with force overwrite
agents-force:
	python scripts/agents/run_agents.py --force

# Dry run agents (print prompts without API calls)
agents-dry:
	python scripts/agents/run_agents.py --dry_run

# Aggregate agent outputs into frontend/data.json
swarm:
	python scripts/swarm/aggregate.py

# Start local web server for frontend
serve:
	cd frontend && python -m http.server 8000

# Full demo: generate fake data, aggregate, and serve
demo: fake swarm serve

# Clean generated files
clean:
	rm -rf data/weeks/*.json
	rm -rf outputs/*.json
	rm -f frontend/data.json

# Validate week data files
validate:
	python scripts/data/build_weeks.py --validate_only

# Show help
help:
	@echo "Bloom Swarm - Makefile targets"
	@echo ""
	@echo "  make fake        Generate fake test data (16 weeks + agent outputs)"
	@echo "  make fake-weeks  Generate only week input files"
	@echo "  make agents      Run LLM agents (requires XAI_API_KEY in .env)"
	@echo "  make agents-force  Run agents, overwriting existing outputs"
	@echo "  make agents-dry  Print prompts without calling API"
	@echo "  make swarm       Aggregate outputs to frontend/data.json"
	@echo "  make serve       Start local server at http://localhost:8000"
	@echo "  make demo        Full pipeline: fake + swarm + serve"
	@echo "  make clean       Remove generated files"
	@echo "  make validate    Validate week data files"
	@echo ""
	@echo "Quick start:"
	@echo "  1. cp .env.example .env"
	@echo "  2. make demo"
	@echo "  3. Open http://localhost:8000"
