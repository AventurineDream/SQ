"""Context assembly: an actor's prompt carries exactly the transcript that actor may see.

Uses a per-test SQLite ledger under pytest's tmp_path, so nothing touches data/.
"""

import json

from context import ContextAssembler
from db import LedgerDB
from schema import Event, EventType


def test_context_assembly_respects_visibility(tmp_path):
    db = LedgerDB(db_path=tmp_path / "ledger.db")
    events = [
        Event(season_id="s2-test", day=1, slot=1, turn_number=1, actor_id="Orchestrator",
              event_type=EventType.NARRATION,
              content="The sci-fi airlock cycles with a heavy hiss. Red emergency lights bathe the corridor.",
              visible_to=["all"]),
        Event(season_id="s2-test", day=1, slot=1, turn_number=2, actor_id="Sage",
              event_type=EventType.ACTION,
              content="draws her plasma cutter and steps into the corridor.",
              visible_to=["all"]),
        Event(season_id="s2-test", day=1, slot=1, turn_number=3, actor_id="Rory",
              event_type=EventType.DIALOGUE,
              content="Wait, check the atmospheric readings first. We don't know if the hull breached.",
              visible_to=["all"]),
        Event(season_id="s2-test", day=1, slot=1, turn_number=4, actor_id="SYSTEM",
              event_type=EventType.SYSTEM,
              content="[Mechanic] Rory rolls Perception (14) vs DC 12. Success. The air is breathable but laced with toxic spores.",
              visible_to=["Rory", "Orchestrator"]),
    ]
    for event in events:
        db.record_event(event)

    assembler = ContextAssembler(db=db, max_tokens=500)
    sage_prompt = assembler.build_actor_prompt(
        season_id="s2-test",
        actor_id="Sage",
        day=1,
        slot=1,
        system_prompt="You are Sage, an impulsive space marine. Keep your responses under 50 words.",
    )
    payload = json.dumps(sage_prompt)

    assert "airlock cycles" in payload
    assert "plasma cutter" in payload
    assert "check the atmospheric readings" in payload
    # The roll result is visible only to Rory and the Orchestrator.
    assert "toxic spores" not in payload
