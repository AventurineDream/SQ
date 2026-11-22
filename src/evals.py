# src/evals.py

from typing import List
from schema import PostEntry, PostEvalRecord, JudgeVerdict
from db import LedgerDB
from inference import LLMAdapter # Import the base adapter

class AutoJudge:
    def __init__(self, db: LedgerDB, judge_model: LLMAdapter):
        self.db = db
        self.judge_model = judge_model

    def evaluate_post(self, season_id: str, post: PostEntry) -> List[PostEvalRecord]:
        """
        Takes a post, asks the judge model to verify its claims against the 
        ledger, and returns the evaluation records.
        """
        
        # 1. Fetch the actual ground truth events for the day/slot
        # For a full implementation, you'd likely fetch the specific events the 
        # post claims to reference, or the entire slot's transcript.
        ground_truth_events = self.db.get_visible_events_for_actor(
            season_id, post.actor_id, post.day, post.slot
        )
        
        transcript = "\n".join([f"[{e.actor_id}]: {e.content}" for e in ground_truth_events])

        # 2. Build the prompt for the Judge Model
        prompt = [
             {
                "role": "system",
                "content": (
                    "You are an impartial evaluator checking a character's post for accuracy against a transcript. "
                    "Extract the main claim from the post and evaluate it. "
                    "You MUST respond in valid JSON matching this exact schema: "
                    '{"claim": "the extracted claim", "evaluation": "correct|claimed_others_event|hallucinated|unclear"}'
                )
             },
             {
                 "role": "user",
                 "content": (
                     f"--- GROUND TRUTH TRANSCRIPT ---\n{transcript}\n------------------------------\n\n"
                     f"Post written by {post.actor_id}: '{post.content}'\n\n"
                     "Evaluate the post's accuracy based ONLY on the ground truth."
                 )
             }
        ]

        # 3. Request a structured verdict. generate_structured validates the reply
        #    against JudgeVerdict and retries with feedback. A reply that never
        #    validates returns None; the diagnostics callback still records each
        #    rejected attempt, so judge failures stay visible in the telemetry.
        #    Temperature 0 keeps the verdict deterministic for the same transcript.
        verdict = self.judge_model.generate_structured(
            prompt_payload=prompt,
            response_model=JudgeVerdict,
            temperature=0.0,
            diagnostic_callback=self.db.record_model_call,
            call_meta={
                "season_id": season_id,
                "day": post.day,
                "slot": post.slot,
                "actor_id": post.actor_id,
                "call_type": "judge",
            },
        )
        if verdict is None:
            return []

        # 4. Save the verdict to the database
        eval_record = PostEvalRecord(
            post_id=post.post_id,
            claim_text=verdict.claim,
            referenced_event_id=None,
            evaluation=verdict.evaluation,
            judge_model_name=self.judge_model.model_name,
        )
        self.db.record_evaluation(eval_record)
        return [eval_record]