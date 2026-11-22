"""Auto-Judge: verdicts are parsed from the judge model and persisted to the ledger.

Two paths are covered. The mock path stubs the abstract `_call_raw` boundary. The
HTTP path runs the real OllamaAdapter (generate_structured, schema enforcement, JSON
parsing) and stubs only `requests.post`, so no Ollama server is needed.
"""

import json
import sqlite3

import pytest

import inference
from db import LedgerDB
from evals import AutoJudge
from inference import LLMAdapter, OllamaAdapter
from schema import ClaimEvaluation, Event, EventType, PostEntry

JUDGE_REPLY = {
    "claim": "I fixed the airlock pressure system.",
    "evaluation": "claimed_others_event",
}


class MockJudgeAdapter(LLMAdapter):
    """Judge stand-in whose `_call_raw` returns a fixed reply string."""

    def __init__(self, reply: str):
        super().__init__(model_name="mock-mistral-judge")
        self.reply = reply

    def _call_raw(self, messages, json_schema=None, temperature=0.7, max_tokens=350) -> str:
        return self.reply


class FakeResponse:
    """Minimal stand-in for requests.Response as used by OllamaAdapter."""

    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"message": {"content": self._content}}


@pytest.fixture
def seeded_db(tmp_path):
    """Ledger with a ground-truth slot where Rory fixes the airlock and Sage only watches."""
    db = LedgerDB(db_path=tmp_path / "eval.db")
    events = [
        Event(season_id="s2-eval", day=1, slot=1, turn_number=1, actor_id="Orchestrator",
              event_type=EventType.NARRATION, content="The airlock depressurizes rapidly.", visible_to=["all"]),
        Event(season_id="s2-eval", day=1, slot=1, turn_number=2, actor_id="Rory",
              event_type=EventType.ACTION, content="reroutes the pressure valves using his datapad.", visible_to=["all"]),
        Event(season_id="s2-eval", day=1, slot=1, turn_number=3, actor_id="SYSTEM",
              event_type=EventType.SYSTEM,
              content="[SYSTEM] Rory rolled tech (12 + 4 = 16) vs DC 15. Result: success.", visible_to=["all"]),
        Event(season_id="s2-eval", day=1, slot=1, turn_number=4, actor_id="Sage",
              event_type=EventType.DIALOGUE, content="Good work, Rory. I thought we were dead.", visible_to=["all"]),
    ]
    for event in events:
        db.record_event(event)
    post = PostEntry(
        actor_id="Sage",
        day=1,
        slot=1,
        roll_value=0.85,
        content="Close call today. The airlock blew, but I managed to reroute the pressure valves just in time. "
        "Rory was panicking.",
    )
    db.record_post(post)
    return db, post


def _stored_evaluations(db: LedgerDB) -> list:
    with sqlite3.connect(db.db_path) as conn:
        return conn.execute(
            "SELECT post_id, evaluation, judge_model_name FROM post_evaluations"
        ).fetchall()


def test_auto_judge_records_mock_verdict(seeded_db):
    db, post = seeded_db
    judge = MockJudgeAdapter(reply=json.dumps(JUDGE_REPLY))

    records = AutoJudge(db=db, judge_model=judge).evaluate_post("s2-eval", post)

    assert len(records) == 1
    assert records[0].evaluation == ClaimEvaluation.CLAIMED
    assert records[0].claim_text == JUDGE_REPLY["claim"]
    assert _stored_evaluations(db) == [(post.post_id, "claimed_others_event", "mock-mistral-judge")]


def test_auto_judge_through_ollama_http_boundary(seeded_db, monkeypatch):
    db, post = seeded_db
    reply_text = json.dumps(JUDGE_REPLY)
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json})
        return FakeResponse(reply_text)

    monkeypatch.setattr(inference.requests, "post", fake_post)
    judge = OllamaAdapter(model_name="mistral-small3.2:24b", base_url="http://judge.invalid:11434")

    records = AutoJudge(db=db, judge_model=judge).evaluate_post("s2-eval", post)

    assert len(records) == 1
    assert records[0].evaluation == ClaimEvaluation.CLAIMED
    assert records[0].judge_model_name == "mistral-small3.2:24b"
    assert _stored_evaluations(db) == [(post.post_id, "claimed_others_event", "mistral-small3.2:24b")]

    # The request carried the judge schema as the Ollama format and temperature 0.
    assert len(calls) == 1
    request = calls[0]["json"]
    assert calls[0]["url"] == "http://judge.invalid:11434/api/chat"
    assert request["model"] == "mistral-small3.2:24b"
    assert request["options"]["temperature"] == 0.0
    assert set(request["format"]["properties"]) == {"claim", "evaluation"}


def test_auto_judge_records_nothing_for_malformed_reply(seeded_db, monkeypatch):
    db, post = seeded_db
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(json)
        return FakeResponse('{"claim": "x", "evaluation": "maybe"}')

    monkeypatch.setattr(inference.requests, "post", fake_post)
    judge = OllamaAdapter(model_name="mistral-small3.2:24b")

    records = AutoJudge(db=db, judge_model=judge).evaluate_post("s2-eval", post)

    # An out-of-enum verdict is rejected on every attempt, so no record is written,
    # but each rejected attempt is still logged to the model-call telemetry.
    assert records == []
    assert len(calls) == 3
    assert _stored_evaluations(db) == []
    with sqlite3.connect(db.db_path) as conn:
        rejected = conn.execute(
            "SELECT call_type, accepted FROM model_calls WHERE call_type = 'judge'"
        ).fetchall()
    assert rejected == [("judge", 0), ("judge", 0), ("judge", 0)]
