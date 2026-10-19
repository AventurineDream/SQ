# src/evals.py

from typing import List
from schema import DiaryEntry, Event, DiaryEvalRecord, ClaimEvaluation
from db import LedgerDB
from inference import LLMAdapter # Import the base adapter
import json

class AutoJudge:
    def __init__(self, db: LedgerDB, judge_model: LLMAdapter):
        self.db = db
        self.judge_model = judge_model

    def evaluate_diary(self, season_id: str, diary: DiaryEntry) -> List[DiaryEvalRecord]:
        """
        Takes a diary entry, asks the judge model to verify its claims against the 
        ledger, and returns the evaluation records.
        """
        
        # 1. Fetch the actual ground truth events for the day/slot
        # For a full implementation, you'd likely fetch the specific events the 
        # diary claims to reference, or the entire slot's transcript.
        ground_truth_events = self.db.get_visible_events_for_actor(
            season_id, diary.actor_id, diary.day, diary.slot
        )
        
        transcript = "\n".join([f"[{e.actor_id}]: {e.content}" for e in ground_truth_events])

        # 2. Build the prompt for the Judge Model
        prompt = [
             {
                "role": "system",
                "content": (
                    "You are an impartial evaluator checking a character's diary for accuracy against a transcript. "
                    "Extract the main claim from the diary and evaluate it. "
                    "You MUST respond in valid JSON matching this exact schema: "
                    '{"claim": "the extracted claim", "evaluation": "correct|claimed_others_event|hallucinated|unclear"}'
                )
             },
             {
                 "role": "user",
                 "content": (
                     f"--- GROUND TRUTH TRANSCRIPT ---\n{transcript}\n------------------------------\n\n"
                     f"Diary written by {diary.actor_id}: '{diary.content}'\n\n"
                     "Evaluate the diary's accuracy based ONLY on the ground truth."
                 )
             }
        ]

        # 3. Request evaluation from the Judge model
        # Using a generic method here, assuming your adapter handles the JSON request
        # Note: You'd need to add a generic `generate` method to LLMAdapter, or 
        # cast the output to a Pydantic model for strict validation.
        raw_output = self.judge_model.generate_generic_json(prompt) # Hypothetical method
        
        eval_records = []
        if raw_output:
            try:
                # 4. Parse the output and create the record
                eval_data = json.loads(raw_output)
                eval_record = DiaryEvalRecord(
                    diary_id=diary.diary_id,
                    claim_text=eval_data.get("claim", diary.content),
                    referenced_event_id=None, # You'd need more complex logic to map to a specific event ID
                    evaluation=ClaimEvaluation(eval_data.get("evaluation", "unclear")),
                    judge_model_name=self.judge_model.model_name
                )
                eval_records.append(eval_record)
                
                # 5. Save to database
                self.db.record_evaluation(eval_record)
            except Exception as e:
                print(f"Error parsing judge output: {e}")
                
        return eval_records