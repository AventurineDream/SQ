import os
import sys
import json
from datetime import datetime, timezone

# Ensure the script can find the src directory when run from the root folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from schema import Event, EventType
from db import LedgerDB
from context import ContextAssembler

def test_context_assembly():
    """Proves the context assembler pulls literal transcripts and formats them correctly."""
    
    # 1. Initialize a temporary test database
    test_db_path = "data/test_ledger.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
        
    db = LedgerDB(db_path=test_db_path)
    print("[*] Test database initialized.")
    
    # 2. Create mock events mimicking early Season 2 TTRPG gameplay
    events = [
        Event(
            season_id="s2-test",
            day=1,
            slot=1,
            turn_number=1,
            actor_id="Orchestrator",
            event_type=EventType.NARRATION,
            content="The sci-fi airlock cycles with a heavy hiss. Red emergency lights bathe the corridor.",
            visible_to=["all"]
        ),
        Event(
            season_id="s2-test",
            day=1,
            slot=1,
            turn_number=2,
            actor_id="Sage",
            event_type=EventType.ACTION,
            content="draws her plasma cutter and steps into the corridor.",
            visible_to=["all"]
        ),
        Event(
            season_id="s2-test",
            day=1,
            slot=1,
            turn_number=3,
            actor_id="Rory",
            event_type=EventType.DIALOGUE,
            content="Wait, check the atmospheric readings first. We don't know if the hull breached.",
            visible_to=["all"]
        ),
        Event(
            season_id="s2-test",
            day=1,
            slot=1,
            turn_number=4,
            actor_id="SYSTEM",
            event_type=EventType.SYSTEM,
            content="[Mechanic] Rory rolls Perception (14) vs DC 12. Success. The air is breathable but laced with toxic spores.",
            visible_to=["Rory", "Orchestrator"] # Sage doesn't inherently know the roll result unless Rory says it
        )
    ]
    
    # 3. Record events to the ledger
    for event in events:
        db.record_event(event)
    print(f"[*] {len(events)} events written to ledger.")
        
    # 4. Initialize Context Assembler (simulate a small 500 token window)
    assembler = ContextAssembler(db=db, max_tokens=500)
    
    # 5. Build the prompt for Sage
    print("[*] Assembling context for Sage's next turn...\n")
    sage_prompt = assembler.build_actor_prompt(
        season_id="s2-test",
        actor_id="Sage",
        day=1,
        slot=1,
        system_prompt="You are Sage, an impulsive space marine. Keep your responses under 50 words."
    )
    
    # 6. Output the payload exactly as it will be sent to the LLM API
    print("=== GENERATED PROMPT PAYLOAD ===")
    print(json.dumps(sage_prompt, indent=2))
    print("================================\n")
    
    # 7. Cleanup
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    print("[*] Test complete and cleaned up.")

if __name__ == "__main__":
    test_context_assembly()