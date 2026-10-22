import argparse
import logging
import os
import sys

# Ensure imports inside src/ (like `from schema import ...`) resolve cleanly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from engine import SimulationEngine
from evals import AutoJudge
from inference import OllamaAdapter
from rules import Skill
from schema import Event, EventType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_WORLDBOOK = {
    "opening_narration": (
        "Emergency klaxons pulse across Deck 4 of the orbital rig. "
        "The primary blast door is sealed, and pressure in the corridor is dropping."
    ),
    "characters": {
        "Sage": {
            "hp": 18,
            "skills": {Skill.ATHLETICS: 3, Skill.PERCEPTION: 1},
            "inventory": ["Pulse Rifle", "Combat Knife"],
            "bio": "You are Sage, a pragmatic space marine. You speak in clipped, tactical sentences.",
        },
        "Rory": {
            "hp": 12,
            "skills": {Skill.TECH: 4, Skill.STEALTH: 2},
            "inventory": ["Plasma Torch", "Datapad"],
            "bio": "You are Rory, the rig's chief systems engineer. You are analytical, dry-witted, and protective of your gear.",
        },
        "Kaelen": {
            "hp": 14,
            "skills": {Skill.PERCEPTION: 3, Skill.PERSUASION: 2},
            "inventory": ["Medkit", "Bio-Scanner"],
            "bio": "You are Kaelen, the crew's xenobiologist and medic. You observe details others miss.",
        },
        "Jace": {
            "hp": 13,
            "skills": {Skill.ACROBATICS: 3, Skill.STEALTH: 3},
            "inventory": ["Sidearm", "Magnetic Grappling Hook"],
            "bio": "You are Jace, a salvage reconnaissance pilot. You use humor to deflect tension.",
        },
    },
}


def main():
    parser = argparse.ArgumentParser(description="Run Simulation Query Season 2")
    parser.add_argument("--season-id", default="s2-scifi-horror", help="Unique season identifier")
    parser.add_argument("--days", type=int, default=1, help="Number of simulated days to run (1-7)")
    parser.add_argument("--db-path", default="data/sq_ledger.db", help="SQLite ledger path")
    args = parser.parse_args()

    logger.info(f"[*] Initializing Simulation Query ({args.season_id}) for {args.days} day(s)...")

    engine = SimulationEngine(
        season_id=args.season_id,
        worldbook_data=DEFAULT_WORLDBOOK,
        db_path=args.db_path,
    )

    # 1. Configure Orchestrator (20-35B) & Cast of 4 Actors (7-13B / 20B)
    engine.orchestrator_model = OllamaAdapter(model_name="qwen3:30b-instruct")
    engine.actors = {
        "Sage": OllamaAdapter(model_name="gpt-oss:20b", think_mode="low"),
        "Rory": OllamaAdapter(model_name="qwen2.5:7b"),
        "Kaelen": OllamaAdapter(model_name="llama3.1:8b"),
        "Jace": OllamaAdapter(model_name="mistral:7b"),
    }

    # 2. Attach Cross-Family Auto-Judge
    judge_adapter = OllamaAdapter(model_name="mistral-small3.2:24b")
    engine.auto_judge = AutoJudge(db=engine.db, judge_model=judge_adapter)

    # 3. Seed Opening Narration on Day 1, Slot 1 if ledger is fresh
    if engine.db.get_last_completed_turn(args.season_id) is None:
        engine.db.record_event(
            Event(
                season_id=args.season_id,
                day=1,
                slot=1,
                turn_number=0,
                actor_id="Orchestrator",
                event_type=EventType.NARRATION,
                content=DEFAULT_WORLDBOOK["opening_narration"],
                visible_to=["all"],
            )
        )

    # 4. Run Autonomous Loop (Crash-Resumable)
    for day in range(1, args.days + 1):
        logger.info(f"=== Commencing Day {day}/{args.days} ===")
        engine.run_day(day=day)

    # 5. Export Relational CSV Telemetry
    export_dir = engine.db.export_season_telemetry(args.season_id)
    logger.info(f"[*] Run complete! Relational CSV snapshots exported to: {export_dir}")


if __name__ == "__main__":
    main()