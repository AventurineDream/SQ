import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from engine import SimulationEngine
from evals import AutoJudge
from inference import OllamaAdapter
from rules import Skill
from schema import Event, EventType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# The 4-actor pod for the Sci-Fi Horror Dry Run
# Swap the model_names below with the exact tags you pulled (e.g., gemma4:12b, qwen3:8b)
DRY_RUN_CAST = {
    "Vance": "qwen3.5:9b",             # he/him, Breacher
    "Lin": "gemma4:12b",               # she/her, Medic
    "Soren": "hermes3:8b",             # they/them, Tech
    "Dax": "Rocinante-X-12B-v1:Q4_K_M" # he/him, Auditor
}

DRY_RUN_WORLDBOOK = {
    "characters": {
        "Vance": {
            "hp": 18, 
            "skills": {Skill.ATHLETICS: 4, Skill.TECH: 1}, 
            "inventory": ["Plasma Torch", "Combat Knife"],
            "bio": "You are Corbin Vance (he/him), a cynical, clipped hull-welder on a derelict orbital rig. Never use customer-service apologies or therapy-speak."
        },
        "Lin": {
            "hp": 14, 
            "skills": {Skill.PERCEPTION: 3}, 
            "inventory": ["Bio-Scanner"],
            "bio": "You are Lin (she/her), the rig's cryo-medic. You are pragmatic and protective of the crew's physical safety."
        },
        "Soren": {
            "hp": 12, 
            "skills": {Skill.TECH: 4, Skill.STEALTH: 2}, 
            "inventory": ["Datapad"],
            "bio": "You are Soren (they/them), the comms tech. You are highly analytical and rely on data over emotion."
        },
        "Dax": {
            "hp": 10, 
            "skills": {Skill.PERSUASION: 3}, 
            "inventory": ["Encrypted Drive"],
            "bio": "You are Dax (he/him), a corporate auditor. You are cowardly, bureaucratic, and easily panicked by physical danger."
        },
    }
}

def inject_director_beat(engine: SimulationEngine, day: int, slot: int, turn: int, content: str, visible_to: list = None):
    """Bypasses the LLM to directly inject a scripted environmental or NPC event."""
    engine.db.record_event(Event(
        season_id=engine.season_id,
        day=day,
        slot=slot,
        turn_number=turn,
        actor_id="Orchestrator",
        event_type=EventType.NARRATION,
        content=content,
        visible_to=visible_to or ["all"]
    ))
    logger.info(f"[DIRECTOR BEAT] {content}")

def main():
    season_id = "s2-dryrun-01"
    
    # Clear out previous runs of the same ID for a clean test
    db_path = "data/sq_ledger.db"
    engine = SimulationEngine(season_id=season_id, worldbook_data=DRY_RUN_WORLDBOOK, db_path=db_path)

    # Attach Adapters
    # Using a fast 8B/9B for the Orchestrator here to save time on the dry run, 
    # but in production this should be qwen3:30b-instruct
    engine.orchestrator_model = OllamaAdapter(model_name="qwen3.5:9b") 
    engine.actors = {
        actor: OllamaAdapter(model_name=model, think_mode="low" if "gpt-oss" in model else None)
        for actor, model in DRY_RUN_CAST.items()
    }
    
    # We leave auto_judge out of this script so we can grade it later with the calibrated frontier judges

    logger.info("=== STARTING 1-DAY DRY RUN POD ===")
    
    # ---------------------------------------------------------
    # SLOT 1: Group Establishment & Epistemic Whisper
    # ---------------------------------------------------------
    logger.info("--- SLOT 1 ---")
    inject_director_beat(engine, 1, 1, 0, "Emergency klaxons pulse across Deck 4. The primary blast door is sealed, and pressure is dropping.")
    
    for turn in range(1, 17):
        # Inject Beat 1: Object Transfer Prompting
        if turn == 4:
            inject_director_beat(engine, 1, 1, turn, "The air cycler groans. Vance notices the vent above Soren is rusted shut and requires thermal cutting.")
        
        # Inject Beat 2: Epistemic Whisper (Only Dax and Lin hear this)
        elif turn == 12:
            inject_director_beat(
                engine, 1, 1, turn, 
                "Lin pulls Dax aside into the alcove. 'I saw the captain's logs. The depressurization wasn't an accident.'",
                visible_to=["Lin", "Dax", "Orchestrator"]
            )
            
        else:
            engine.run_turn(day=1, slot=1, turn_number=turn)
            
    engine._trigger_diaries(day=1, slot=1)
    
    # ---------------------------------------------------------
    # SLOT 2: Contested Action & The Action-Result Trap
    # ---------------------------------------------------------
    logger.info("--- SLOT 2 ---")
    inject_director_beat(engine, 1, 2, 0, "The group reaches the lower manual release lever. It is heavily rusted and sparks dangerously.")

    for turn in range(1, 17):
        # Inject Beat 3: Forced Failure & Assistant Bait
        if turn == 8:
            inject_director_beat(engine, 1, 2, turn, "[SYSTEM] Vance rolled athletics (2 + 4 = 6) vs DC 20. Result: failure. The lever refuses to budge and Vance's grip slips.")
            inject_director_beat(engine, 1, 2, turn, "Dax begins hyperventilating. 'Vance, you're useless! Please, someone give me a step-by-step list of how we survive this!'")
            
        else:
            engine.run_turn(day=1, slot=2, turn_number=turn)
            
    engine._trigger_diaries(day=1, slot=2)

    # ---------------------------------------------------------
    # SLOT 3: Multi-Agent Climax
    # ---------------------------------------------------------
    logger.info("--- SLOT 3 ---")
    inject_director_beat(engine, 1, 3, 0, "The air is growing noticeably thin. The only remaining option is the compromised blast door panel.")

    for turn in range(1, 17):
        engine.run_turn(day=1, slot=3, turn_number=turn)
            
    engine._trigger_diaries(day=1, slot=3)

    # ---------------------------------------------------------
    # EXPORT RESULTS
    # ---------------------------------------------------------
    export_dir = engine.db.export_season_telemetry(season_id)
    logger.info(f"[*] Dry Run Complete! Telemetry exported to: {export_dir}")

if __name__ == "__main__":
    main()