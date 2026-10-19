import os
import sys
from datetime import datetime, timezone

# Ensure the script can find the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from schema import EventType, ActorTurnOutput
from engine import SimulationEngine
from inference import LLMAdapter
from db import LedgerDB
from rules import Skill

class DummyActorAdapter(LLMAdapter):
    """Mocks an LLM returning perfectly formatted JSON."""
    
    def generate_turn(self, prompt_payload):
        return ActorTurnOutput(
            internal_thought="I need to get through this door.",
            action_type=EventType.ACTION,
            public_content="I try to pry the blast door open with my bare hands.",
            target_actors=["all"]
        )
        
    def generate_diary(self, prompt_payload):
        return "Everything is broken and the parasite is definitely in the vents."


def test_end_to_end_loop():
    """Runs the orchestration engine without making actual HTTP requests."""
    
    test_db_path = "data/mock_engine_ledger.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
        
    print("[*] Initializing Season 2 Mock Environment...")
    
    # 1. Initialize the Worldbook based on our template
    wb = {
        "characters": {
            "Sage": {"hp": 18, "skills": {Skill.ATHLETICS: 3, Skill.TECH: 0}, "inventory": ["Pulse Rifle"]},
            "Rory": {"hp": 12, "skills": {Skill.ATHLETICS: -1, Skill.TECH: 4}, "inventory": ["Datapad"]}
        }
    }
    
    # 2. Instantiate the Engine
    engine = SimulationEngine(season_id="s2-mock", worldbook_data=wb)
    
    # 3. Safely override the production database with our test database
    engine.db = LedgerDB(db_path=test_db_path)
    engine.context_assembler.db = engine.db
    
    # 4. Inject the Dummy Adapters to bypass Ollama
    engine.actors = {
        "Sage": DummyActorAdapter(model_name="dummy-sage"),
        "Rory": DummyActorAdapter(model_name="dummy-rory")
    }
    
    # 5. Run a 3-turn sequence
    print("[*] Executing Command Center Loop (3 Turns)...\n")
    for turn in range(1, 4):
        engine.run_turn(day=1, slot=1, turn_number=turn)
        
    # 6. Verify the Output
    print("=== FINAL LEDGER STATE ===")
    events = engine.db.get_visible_events_for_actor("s2-mock", "Sage", 1, 1)
    
    for e in events:
        if e.event_type == EventType.SYSTEM:
            print(f"\033[93m[Turn {e.turn_number}] {e.actor_id}: {e.content}\033[0m") # Yellow for system
        else:
            print(f"[Turn {e.turn_number}] {e.actor_id} ({e.event_type.value}): {e.content}")
            
    print("==========================")
    
    # Cleanup
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    print("\n[*] Engine Test Complete.")

if __name__ == "__main__":
    test_end_to_end_loop()