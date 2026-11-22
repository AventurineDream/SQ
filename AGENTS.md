# AGENTS.md

The authoritative guide for this repository is [CLAUDE.md](CLAUDE.md). Read it before
editing: it defines the module responsibilities, the data policy (never commit
`.db`, `.db-shm`, `.db-wal`), and the run commands.

## Commands

| Gate | Command |
|---|---|
| Install | `pip install -e '.[dev]'` |
| Tests | `python -m pytest` |
| Lint | `ruff check .` |
| Compile | `python -m compileall -q src tests scripts run_season.py` |

CI (`.github/workflows/ci.yml`) runs the same gates on Python 3.11 and 3.14.

## Conventions

- Package modules live flat in `src/` and import each other by bare name (`from schema import ...`);
  do not add `sys.path` manipulation.
- Tests use `tmp_path` for ledgers; never point a test at `data/`.
