import random
from typing import Dict, Any, Tuple
from enum import Enum

class Skill(str, Enum):
    ATHLETICS = "athletics"
    ACROBATICS = "acrobatics"
    STEALTH = "stealth"
    PERCEPTION = "perception"
    PERSUASION = "persuasion"
    TECH = "tech" # Custom sci-fi skill

class RollResult(str, Enum):
    CRIT_SUCCESS = "critical_success"
    SUCCESS = "success"
    FAILURE = "failure"
    CRIT_FAILURE = "critical_failure"

class CharacterSheet:
    """Tracks the persistent RPG stats for a single actor."""
    def __init__(self, actor_id: str, hp: int, max_hp: int, skills: Dict[Skill, int], inventory: list[str]):
        self.actor_id = actor_id
        self.hp = hp
        self.max_hp = max_hp
        self.skills = skills # e.g., {Skill.ATHLETICS: 3, Skill.PERCEPTION: -1}
        self.inventory = inventory

class TTRPGRulebook:
    """Deterministic RPG engine. Handles math so the LLM doesn't have to."""
    
    def __init__(self, character_sheets: Dict[str, CharacterSheet]):
        self.sheets = character_sheets

    def roll_skill_check(self, actor_id: str, skill: Skill, dc: int) -> Tuple[RollResult, int, str]:
        """
        Rolls a d20, adds the character's modifier, and compares to the DC.
        Returns the categorical result, the final total, and a summary string for the ledger.
        """
        if actor_id not in self.sheets:
            return RollResult.FAILURE, 0, f"Error: No sheet for {actor_id}"

        modifier = self.sheets[actor_id].skills.get(skill, 0)
        d20_roll = random.randint(1, 20)
        total = d20_roll + modifier

        if d20_roll == 20:
            result = RollResult.CRIT_SUCCESS
        elif d20_roll == 1:
            result = RollResult.CRIT_FAILURE
        elif total >= dc:
            result = RollResult.SUCCESS
        else:
            result = RollResult.FAILURE

        summary = f"[SYSTEM] {actor_id} rolled {skill.value} ({d20_roll} + {modifier} = {total}) vs DC {dc}. Result: {result.value}."
        return result, total, summary
        
    def check_inventory(self, actor_id: str, item_name: str) -> bool:
        """Validates if an actor actually possesses an item they are trying to use."""
        if actor_id not in self.sheets:
            return False
        
        # Simple case-insensitive check
        inventory_lower = [item.lower() for item in self.sheets[actor_id].inventory]
        return item_name.lower() in inventory_lower