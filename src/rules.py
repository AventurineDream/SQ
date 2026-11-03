import random
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Skill(str, Enum):
    ATHLETICS = "athletics"
    ACROBATICS = "acrobatics"
    STEALTH = "stealth"
    PERCEPTION = "perception"
    PERSUASION = "persuasion"
    TECH = "tech"


class RollResult(str, Enum):
    CRIT_SUCCESS = "critical_success"
    SUCCESS = "success"
    FAILURE = "failure"
    CRIT_FAILURE = "critical_failure"


class CharacterSheet:
    """Tracks the persistent RPG stats and inventory for a single actor."""

    def __init__(
        self,
        actor_id: str,
        hp: int,
        max_hp: int,
        skills: Dict[Skill, int],
        inventory: List[str],
    ):
        self.actor_id = actor_id
        self.hp = hp
        self.max_hp = max_hp
        self.skills = skills
        self.inventory = inventory


class TTRPGRulebook:
    """Deterministic RPG engine. Handles d20 math and inventory checks in Python."""

    def __init__(self, character_sheets: Dict[str, CharacterSheet]):
        self.sheets = character_sheets

    def roll_skill_check(
        self,
        actor_id: str,
        skill: Skill,
        dc: int,
        rng: Optional[random.Random] = None,
    ) -> Tuple[RollResult, int, str]:
        """Rolls a d20 (using deterministic RNG if provided), adds modifier, and compares to DC."""
        if actor_id not in self.sheets:
            return RollResult.FAILURE, 0, f"[SYSTEM] Error: No character sheet for {actor_id}."

        roller = rng if rng is not None else random
        modifier = self.sheets[actor_id].skills.get(skill, 0)
        d20_roll = roller.randint(1, 20)
        total = d20_roll + modifier

        if d20_roll == 20:
            result = RollResult.CRIT_SUCCESS
        elif d20_roll == 1:
            result = RollResult.CRIT_FAILURE
        elif total >= dc:
            result = RollResult.SUCCESS
        else:
            result = RollResult.FAILURE

        summary = (
            f"[SYSTEM] {actor_id} rolled {skill.value} "
            f"({d20_roll} + {modifier} = {total}) vs DC {dc}. Result: {result.value}."
        )
        return result, total, summary

    def check_inventory(self, actor_id: str, item_name: str) -> bool:
        """Validates if an actor possesses an item they are attempting to use."""
        if not item_name or actor_id not in self.sheets:
            return True
        inventory_lower = [item.lower() for item in self.sheets[actor_id].inventory]
        return item_name.lower() in inventory_lower