# Simulation Query (SQ)

A multi-agent TTRPG simulation harness: several actor models and one orchestrator
play a 7-day season, with every event written to a SQLite WAL ledger and an
Auto-Judge grading social posts against the ledger.

Architecture, data policy and the run commands are in [CLAUDE.md](CLAUDE.md).

## Setup

Requires Python 3.11 or newer (`.python-version` pins 3.14 for development).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
```

The editable install puts `src/` modules on the path, so `run_season.py` and
`scripts/` need no path setup.

## Tests and lint

```bash
python -m pytest
ruff check .
```

Tests write their ledgers under pytest's `tmp_path`; they do not touch `data/`.
The Auto-Judge HTTP test stubs `requests.post`, so no Ollama server is needed.
Simulation runs need a reachable Ollama endpoint (see `src/inference.py`).
