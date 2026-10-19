# Simulation Query (SQ) - Project Architecture & Guidelines

## 1. Project Vision & Scope
Simulation Query is an autonomous, production-grade, multi-agent simulation platform and research harness. It models 4 actors (7–13B) and 1 Orchestrator (20–35B) across a 7-day, 3-run/day TTRPG framework governed by SQLite state durability and automated eval telemetry.

## 2. Repository Structure
* **`src/`**: Core execution engine. All executable logic, schema definitions, database adapters, context builders, and inference routers live here.
* **`tests/`**: Automated unit and integration tests (e.g., mock loops, context verification, auto-judge pipelines).
* **`data/`**: Local SQLite databases (`sq_ledger.db`) and telemetry export files. Never commit production databases to git.
* **`docs/`**: Non-executable documentation, including the PRD, TTRPG Rulebook, and Worldbook templates.
* **`notebooks/`**: Strictly reserved for exploratory data analysis and post-season evaluation. Never used to run simulation loops.

## 3. Core Architectural Rules
1. **Strict Separation of Concerns:** Inference logic (`inference.py`), state storage (`db.py`), context assembly (`context.py`), and rules validation (`rules.py`) are fully decoupled.
2. **Immutable State & Provenance:** All state is persisted in SQLite via Pydantic models. Every event carries a deterministic UUID. Diary claims map directly to `event_id` keys via relational tables (`diary_citations`, `diary_evaluations`).
3. **Token-Aware Context Assembly:** Actors receive literal, chronological transcripts of the shared canon up to token safety margins, avoiding lossy text summarization.
4. **Deterministic Mechanics:** All skill checks, HP tracking, and inventory validation are executed in deterministic Python (`rules.py`). LLMs are strictly used for intent parsing, dialogue generation, and atmospheric narration.

## 4. Key Execution Commands
* **Run Mock Pipeline Test:** `python tests/test_pipeline.py`
* **Run End-to-End Mock Engine Loop:** `python tests/test_mock_loop.py`
* **Run Auto-Judge Telemetry Test:** `python tests/test_evals.py`
* **Run Production Season:** `python run_season.py` (Requires local `ollama serve` active).