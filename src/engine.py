import hashlib
import logging
import random
from typing import Dict, List, Optional, Tuple # <-- Added Optional

from schema import (
    ActorTurnOutput,
    DiaryCitation,
    DiaryEntry, # <-- Added DiaryEntry
    DiaryGenerationOutput,
    Event,
    EventType,
    FailureCode,
    OrchestratorIntentOutput,
    OrchestratorNarrationOutput,
)
from db import LedgerDB
from context import ContextAssembler
from inference import LLMAdapter, OllamaAdapter # <-- Added adapters
from rules import CharacterSheet, Skill
class SimulationEngine:
    """The Command Center Orchestrator."""

    def __init__(
        self,
        season_id: str,
        worldbook_data: Dict,
        db_path: str = "data/sq_ledger.db", # <-- Add this line
    ):
        self.season_id = season_id
        self.worldbook_data = worldbook_data

        # Pass the db_path down to the Ledger
        self.db = LedgerDB(db_path=db_path)

        self.orchestrator_model: Optional[LLMAdapter] = OllamaAdapter(
            model_name="qwen3:30b-instruct"
        )
        self.actors: Dict[str, LLMAdapter] = {
            "Sage": OllamaAdapter(model_name="gpt-oss:20b"),
            "Rory": OllamaAdapter(model_name="qwen3.8:27b"),
        }
        self.auto_judge = None
        
    def _initialize_rules(self, data: Dict) -> TTRPGRulebook:
        """Parses the Worldbook config into deterministic CharacterSheet objects."""
        sheets = {}
        for actor_id, stats in data["characters"].items():
            sheets[actor_id] = CharacterSheet(
                actor_id=actor_id,
                hp=stats["hp"],
                max_hp=stats["hp"],
                skills={Skill(k): v for k, v in stats["skills"].items()},
                inventory=stats["inventory"]
            )
        return TTRPGRulebook(character_sheets=sheets)

    def _determine_next_actor(self) -> str:
        """Logic for fluid turn-taking. 
        For a basic loop, it can just be round-robin. 
        For advanced S2, the Orchestrator LLM can dictate the next speaker.
        """
        # Placeholder for round-robin
        return random.choice(list(self.actors.keys()))

    def run_turn(self, day: int, slot: int, turn_number: int):
        """Executes a single turn in the simulation."""
        
        actor_id = self._determine_next_actor()
        actor = self.actors[actor_id]
        
        # 1. Assemble Context
        prompt = self.context_assembler.build_actor_prompt(
            season_id=self.season_id,
            actor_id=actor_id,
            day=day,
            slot=slot,
            system_prompt=f"You are {actor_id}. Play your role."
        )
        
        # 2. Get Actor Intent (with retry loop)
        max_retries = 3
        action_output = None
        
        for attempt in range(max_retries):
            action_output = actor.generate_turn(prompt)
            if action_output is not None:
                break # Valid JSON received
            logger.warning(f"[{actor_id}] Turn failed validation. Retrying ({attempt+1}/{max_retries})...")
            
        if not action_output:
            logger.error(f"[{actor_id}] Failed to generate valid turn after 3 retries. Skipping.")
            return

        # 3. State/Rules Verification
        # If the actor's action requires a check, intercept it with Python
        if "try to" in action_output.public_content.lower() or "attempt" in action_output.public_content.lower():
             # Placeholder: In full implementation, Orchestrator LLM decides the DC and Skill.
             result, total, summary = self.rules.roll_skill_check(actor_id, Skill.ATHLETICS, 15)
             
             # Record the system roll so everyone sees it
             sys_event = Event(
                 season_id=self.season_id, day=day, slot=slot, turn_number=turn_number,
                 actor_id="SYSTEM", event_type=EventType.SYSTEM, content=summary, visible_to=["all"]
             )
             self.db.record_event(sys_event)

        # 4. Record the Actor's Action
        event = Event(
            season_id=self.season_id,
            day=day,
            slot=slot,
            turn_number=turn_number,
            actor_id=actor_id,
            event_type=action_output.action_type,
            content=action_output.public_content,
            visible_to=["all"] # Or action_output.target_actors
        )
        self.db.record_event(event)

    def run_day(self, day: int):
        """Runs the 3 slots for a specific day."""
        for slot in range(1, 4):
            logger.info(f"--- Starting Day {day}, Slot {slot} ---")
            
            # Orchestrator sets the scene
            # scene_init = self.orchestrator_model.generate(...)
            
            for turn in range(1, 17):
                self.run_turn(day, slot, turn)
                
            # Run Stochastic Diaries at end of slot
            self._trigger_diaries(day, slot)

    def _trigger_diaries(self, day: int, slot: int):
        """Rolls stochastic triggers and prompts diaries."""
        for actor_id, actor in self.actors.items():
            roll = random.uniform(0, 1)
            if roll >= 0.34:
                logger.info(f"[{actor_id}] rolled {roll:.2f}. Writing diary...")
                prompt = self.context_assembler.build_diary_prompt(
                    self.season_id, actor_id, day, slot, f"You are {actor_id}."
                )
                diary_content = actor.generate_diary(prompt)
                if diary_content:
                    # Save to DB (Skipping full DiaryEntry instantiation for brevity)
                    pass
                diary_content = actor.generate_diary(prompt)
                if diary_content:
                    diary = DiaryEntry(
                        actor_id=actor_id, day=day, slot=slot, roll_value=roll, content=diary_content
                    )
                    self.db.record_diary(diary)
                    
                    # Run the Auto-Judge immediately
                    self.auto_judge.evaluate_diary(self.season_id, diary)

# Entry point
if __name__ == "__main__":
    # Mock Worldbook
    wb = {
        "characters": {
            "Sage": {"hp": 10, "skills": {Skill.ATHLETICS: 3}, "inventory": ["Plasma Cutter"]},
            "Rory": {"hp": 8, "skills": {Skill.TECH: 4}, "inventory": ["Datapad"]}
        }
    }
    
    engine = SimulationEngine(season_id="s2-alpha", worldbook_data=wb)
    engine.run_day(day=1)