"""End-to-end turn loop with stub actors: no HTTP, ledger in tmp_path.

Exercises SimulationEngine.run_turn through prompt assembly, validation and ledger
writes, with deterministic replies from DummyActorAdapter.
"""

from engine import SimulationEngine
from inference import LLMAdapter
from rules import Skill
from schema import EventType


class DummyActorAdapter(LLMAdapter):
    """Returns a well-formed ActorTurnOutput for every call."""

    def _call_raw(self, messages, json_schema=None, temperature=0.7, max_tokens=350) -> str:
        return (
            '{"internal_thought": "I need to get through this door.", '
            '"action_type": "action", '
            '"public_content": "I try to pry the blast door open with my bare hands.", '
            '"target_actors": ["all"], "item_used": null}'
        )


def test_end_to_end_loop(tmp_path):
    worldbook = {
        "characters": {
            "Sage": {"hp": 18, "skills": {Skill.ATHLETICS: 3, Skill.TECH: 0}, "inventory": ["Pulse Rifle"]},
            "Rory": {"hp": 12, "skills": {Skill.ATHLETICS: -1, Skill.TECH: 4}, "inventory": ["Datapad"]},
        }
    }
    engine = SimulationEngine(
        season_id="s2-mock", worldbook_data=worldbook, db_path=tmp_path / "engine.db"
    )
    engine.orchestrator_model = None
    engine.actors = {
        "Sage": DummyActorAdapter(model_name="dummy-sage"),
        "Rory": DummyActorAdapter(model_name="dummy-rory"),
    }

    for turn in range(1, 4):
        engine.run_turn(day=1, slot=1, turn_number=turn)

    events = engine.db.get_visible_events_for_actor("s2-mock", "Sage", 1, 1)
    actions = [e for e in events if e.event_type == EventType.ACTION]
    # Turns 1-3 alternate Sage, Rory, Sage; every turn's action lands in the ledger.
    assert [(e.actor_id, e.turn_number) for e in actions] == [("Sage", 1), ("Rory", 2), ("Sage", 3)]
    assert all("blast door" in e.content for e in actions)
