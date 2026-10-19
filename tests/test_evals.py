import os
import sys
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from schema import Event, EventType, DiaryEntry, ClaimEvaluation
from db import LedgerDB
from evals import AutoJudge
from inference import LLMAdapter

class MockJudgeAdapter(LLMAdapter):
    """Mocks an LLM Judge returning a structured evaluation."""
    
    def generate_turn(self, prompt_payload):
        pass
        
    def generate_diary(self, prompt_payload):
        pass
        
    def generate_generic_json(self, prompt_payload) -> str:
        """Simulates the Mistral judge catching a misattribution."""
        return json.dumps({
            "claim": "I fixed the airlock pressure system.",
            "evaluation": "claimed_others_event" 
        })

def test_auto_judge_pipeline():
    """Proves the Auto-Judge intercepts diaries and logs evaluations."""
    
    test_db_path = "data/test_eval_ledger.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
        
    db = LedgerDB(db_path=test_db_path)
    print("[*] Evaluation database initialized.")
    
    # 1. Setup Ground Truth Events (Rory fixes the airlock, Sage just watches)
    events = [
        Event(season_id="s2-eval", day=1, slot=1, turn_number=1, actor_id="Orchestrator", event_type=EventType.NARRATION, content="The airlock depressurizes rapidly.", visible_to=["all"]),
        Event(season_id="s2-eval", day=1, slot=1, turn_number=2, actor_id="Rory", event_type=EventType.ACTION, content="reroutes the pressure valves using his datapad.", visible_to=["all"]),
        Event(season_id="s2-eval", day=1, slot=1, turn_number=3, actor_id="SYSTEM", event_type=EventType.SYSTEM, content="[SYSTEM] Rory rolled tech (12 + 4 = 16) vs DC 15. Result: success.", visible_to=["all"]),
        Event(season_id="s2-eval", day=1, slot=1, turn_number=4, actor_id="Sage", event_type=EventType.DIALOGUE, content="Good work, Rory. I thought we were dead.", visible_to=["all"])
    ]
    for e in events:
        db.record_event(e)

    # 2. Simulate Sage writing a misattributed diary entry
    misattributed_diary = DiaryEntry(
        actor_id="Sage",
        day=1,
        slot=1,
        roll_value=0.85, # Passed the stochastic check
        content="Close call today. The airlock blew, but I managed to reroute the pressure valves just in time. Rory was panicking."
    )
    db.record_diary(misattributed_diary)
    print(f"[*] Recorded Diary: '{misattributed_diary.content}'")

    # 3. Run the Auto-Judge
    print("[*] Running Auto-Judge...")
    mock_judge_model = MockJudgeAdapter(model_name="mock-mistral-judge")
    auto_judge = AutoJudge(db=db, judge_model=mock_judge_model)
    
    eval_records = auto_judge.evaluate_diary(season_id="s2-eval", diary=misattributed_diary)

    # 4. Verify Results
    print("\n=== EVALUATION RESULT ===")
    if eval_records:
        record = eval_records[0]
        print(f"Claim Extracted: {record.claim_text}")
        if record.evaluation == ClaimEvaluation.CLAIMED:
            print(f"\033[91mVerdict: {record.evaluation.value.upper()} (Misattribution Caught!)\033[0m")
        else:
            print(f"Verdict: {record.evaluation.value}")
        print(f"Logged to DB ID: {record.eval_id}")
    else:
        print("Failed to generate evaluation.")
    print("=========================\n")

    # Cleanup
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    print("[*] Eval Test Complete.")

if __name__ == "__main__":
    test_auto_judge_pipeline()