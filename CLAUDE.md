# Simulation Query (SQ) — Project Architecture & Guidelines (v3.0)

## 1. Project Vision & Scope
Simulation Query is an autonomous, production-grade multi-agent simulation platform and research harness. It orchestrates 4 distinct actor models (7–13B) and 1 Orchestrator model (20–35B) across a 7-day, 3-slot/day, 16-turn/slot TTRPG framework governed by SQLite WAL state durability and automated evaluation telemetry.

## 2. Repository Structure
* **`src/schema.py`**: Single source of truth for Pydantic data contracts (`Event`, `DiaryEntry`, `DiaryCitation`, `DiaryEvalRecord`, `ModelCallDiagnostic`, `FailureCode`, and structured LLM output schemas).
* **`src/db.py`**: Thread-safe SQLite WAL ledger (`LedgerDB`), crash-recovery state queries, and relational CSV telemetry export (`export_season_telemetry`).
* **`src/context.py`**: Token-aware `ContextAssembler` featuring immutable `[event_id]` prefixing and second-person focal framing (`YOU (Actor)`).
* **`src/rules.py`**: Deterministic Python d20 skill checks (`TTRPGRulebook`) and inventory verification seeded via SHA-256 per turn.
* **`src/prompts.py`**: Centralized system prompts enforcing the Action-Result boundary, two-step Orchestrator narration, and `SQ_FAILURE_CODEBOOK_V2` Auto-Judge grading.
* **`src/inference.py`**: Self-healing `LLMAdapter` and `OllamaAdapter` with native JSON Schema (GBNF) enforcement, `<think>` tag sanitization, and retry-with-feedback.
* **`src/engine.py`**: Crash-resumable `SimulationEngine` executing the 2-step Orchestrator validation and stochastic tweet-diary loop ($N \ge 0.34$).
* **`src/evals.py`**: Witness-aware `AutoJudge` comparing diary claims against full-slot ground truth (`[WITNESSED]` vs. `[UNWITNESSED]`).
* **`tests/`**: Automated unit and integration tests (`test_pipeline.py`, `test_mock_loop.py`, `test_evals.py`).
* **`data/`**: Local SQLite ledgers and exported CSV snapshots. **Never commit `.db`, `.db-shm`, or `.db-wal` files to git.**

## 3. Key Execution Commands
* **Run All Unit & Integration Tests:**
  `python tests/test_pipeline.py && python tests/test_mock_loop.py && python tests/test_evals.py`
* **Run Pre-Season Model Audition (3 Genres):**
  `python audition.py --model qwen2.5:7b`
* **Run Production Simulation (Crash-Resumable + Auto CSV Export):**
  `python run_season.py --season-id s2-scifi-horror --days 1`