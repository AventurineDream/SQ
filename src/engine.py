import hashlib
import logging
import random
from typing import Dict, List, Optional, Tuple

from schema import (
    ActorTurnOutput,
    DiaryCitation,
    DiaryEntry,
    DiaryGenerationOutput,
    Event,
    EventType,
    FailureCode,
    OrchestratorIntentOutput,
    OrchestratorNarrationOutput,
)
from db import LedgerDB
from context import ContextAssembler
from inference import LLMAdapter, OllamaAdapter
from rules import CharacterSheet, Skill, TTRPGRulebook
from prompts import (
    ORCHESTRATOR_INTENT_SYSTEM_PROMPT,
    ORCHESTRATOR_NARRATION_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

class SimulationEngine:
    """The Command Center Orchestrator."""

    def __init__(
        self,
        season_id: str,
        worldbook_data: Dict,
        db_path: str = "data/sq_ledger.db",
    ):
        self.season_id = season_id
        self.worldbook_data = worldbook_data

        # Pass the db_path down to the Ledger
        self.db = LedgerDB(db_path=db_path)
        self.context_assembler = ContextAssembler(db=self.db)
        self.rules = self._initialize_rules(worldbook_data)

        self.orchestrator_model: Optional[LLMAdapter] = OllamaAdapter(
            model_name="qwen3:30b-instruct"
        )
        self.actors: Dict[str, LLMAdapter] = {}
        self.auto_judge = None

    def _initialize_rules(self, worldbook_data: Dict) -> TTRPGRulebook:
        sheets = {}
        for char_name, data in worldbook_data.get("characters", {}).items():
            sheets[char_name] = CharacterSheet(
                actor_id=char_name,
                hp=data.get("hp", 10),
                max_hp=data.get("hp", 10),
                skills=data.get("skills", {}),
                inventory=data.get("inventory", []),
            )
        return TTRPGRulebook(character_sheets=sheets)

    def _get_turn_rng(self, day: int, slot: int, turn_number: int, actor_id: str) -> random.Random:
        seed_str = f"{self.season_id}:{day}:{slot}:{turn_number}:{actor_id}"
        seed_int = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:16], 16)
        return random.Random(seed_int)

    def run_turn(self, day: int, slot: int, turn_number: int):
        # We process one actor per turn sequentially based on turn order
        actor_ids = list(self.actors.keys())
        actor_id = actor_ids[(turn_number - 1) % len(actor_ids)]
        actor_model = self.actors[actor_id]

        rng = self._get_turn_rng(day, slot, turn_number, actor_id)

        prompt = self.context_assembler.build_actor_prompt(
            season_id=self.season_id,
            actor_id=actor_id,
            day=day,
            slot=slot,
            system_prompt=self.worldbook_data["characters"][actor_id].get("bio", "")
        )

        def validate_turn(output: ActorTurnOutput) -> Tuple[bool, Optional[str], FailureCode]:
            if not self.orchestrator_model:
                return True, None, FailureCode.NONE
                
            intent_prompt = [
                {"role": "system", "content": ORCHESTRATOR_INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Actor: {actor_id}\nProposed Action: {output.public_content}"}
            ]
            intent_eval = self.orchestrator_model.generate_structured(
                prompt_payload=intent_prompt,
                response_model=OrchestratorIntentOutput,
                temperature=0.0,
                max_tokens=150
            )
            
            if not intent_eval:
                return False, "Orchestrator failed to parse intent.", FailureCode.FORMAT_VIOLATION
                
            if not intent_eval.is_valid_turn:
                return False, intent_eval.rejection_reason, FailureCode.GOD_MODE_OUTCOME
                
            if output.item_used and not self.rules.check_inventory(actor_id, output.item_used):
                return False, f"You do not possess the item: {output.item_used}", FailureCode.FORMAT_VIOLATION
                
            output._intent_cache = intent_eval
            return True, None, FailureCode.NONE

        action_output = actor_model.generate_structured(
            prompt_payload=prompt,
            response_model=ActorTurnOutput,
            validator_fn=validate_turn,
            diagnostic_callback=self.db.record_model_call,
            call_meta={"season_id": self.season_id, "day": day, "slot": slot, "turn_number": turn_number, "call_type": "turn"}
        )

        if not action_output:
            logger.error(f"[{actor_id}] Turn {turn_number} completely failed after retries.")
            return

        target_visibility = action_output.target_actors
        if "all" not in [t.lower() for t in target_visibility]:
            if actor_id not in target_visibility:
                target_visibility.append(actor_id)
            if "Orchestrator" not in target_visibility:
                target_visibility.append("Orchestrator")

        self.db.record_event(Event(
            season_id=self.season_id,
            day=day,
            slot=slot,
            turn_number=turn_number,
            actor_id=actor_id,
            event_type=action_output.action_type,
            content=action_output.public_content,
            visible_to=target_visibility
        ))

        intent_eval = getattr(action_output, "_intent_cache", None)
        if intent_eval and intent_eval.requires_check and intent_eval.skill:
            try:
                skill_enum = Skill(intent_eval.skill.lower())
                dc = intent_eval.dc or 15
                
                roll_result, total, summary = self.rules.roll_skill_check(
                    actor_id=actor_id, 
                    skill=skill_enum, 
                    dc=dc,
                    rng=rng
                )
                
                self.db.record_event(Event(
                    season_id=self.season_id,
                    day=day,
                    slot=slot,
                    turn_number=turn_number,
                    actor_id="SYSTEM",
                    event_type=EventType.SYSTEM,
                    content=summary,
                    visible_to=["all"]
                ))
                
                narration_prompt = [
                    {"role": "system", "content": ORCHESTRATOR_NARRATION_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Action: {action_output.public_content}\nResult: {summary}"}
                ]
                
                narration_eval = self.orchestrator_model.generate_structured(
                    prompt_payload=narration_prompt,
                    response_model=OrchestratorNarrationOutput,
                    temperature=0.7,
                    max_tokens=150
                )
                
                if narration_eval:
                    self.db.record_event(Event(
                        season_id=self.season_id,
                        day=day,
                        slot=slot,
                        turn_number=turn_number,
                        actor_id="Orchestrator",
                        event_type=EventType.NARRATION,
                        content=narration_eval.narration,
                        visible_to=["all"]
                    ))

            except ValueError:
                logger.error(f"Invalid skill returned by orchestrator: {intent_eval.skill}")

    def _trigger_diaries(self, day: int, slot: int, force_write: bool = False):
        for actor_id, actor_model in self.actors.items():
            rng = self._get_turn_rng(day, slot, 0, actor_id)
            roll_val = 1.0 if force_write else rng.uniform(0, 1)
            if roll_val >= 0.34:
                char_bio = (
                    self.worldbook_data.get("characters", {})
                    .get(actor_id, {})
                    .get("bio", f"You are {actor_id}.")
                )
                prompt = self.context_assembler.build_diary_prompt(
                    season_id=self.season_id,
                    actor_id=actor_id,
                    day=day,
                    slot=slot,
                    character_card=char_bio,
                )
                res = actor_model.generate_structured(
                    prompt_payload=prompt,
                    response_model=DiaryGenerationOutput,
                    temperature=0.9,
                    max_tokens=180,
                    diagnostic_callback=self.db.record_model_call,
                    call_meta={
                        "season_id": self.season_id,
                        "day": day,
                        "slot": slot,
                        "turn_number": 16,
                        "actor_id": actor_id,
                        "call_type": "diary",
                    },
                )
                if res:
                    diary = DiaryEntry(
                        season_id=self.season_id,
                        actor_id=actor_id,
                        day=day,
                        slot=slot,
                        roll_value=round(roll_val, 4),
                        content=res.content[:280],
                    )
                    self.db.record_diary(diary)
                    for cite_id in res.cited_event_ids:
                        if hasattr(self.db, "record_citation"):
                            self.db.record_citation(
                                DiaryCitation(
                                    diary_id=diary.diary_id,
                                    referenced_event_id=cite_id,
                                )
                            )
                        elif hasattr(self.db, "record_diary_citation"):
                            self.db.record_diary_citation(
                                DiaryCitation(
                                    diary_id=diary.diary_id,
                                    referenced_event_id=cite_id,
                                )
                            )
                    if self.auto_judge:
                        self.auto_judge.evaluate_diary(self.season_id, diary)