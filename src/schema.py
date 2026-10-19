from datetime import datetime, timezone
from enum import Enum
from typing import List
from uuid import uuid4
from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of events that can occur during a run."""

    DIALOGUE = "dialogue"
    ACTION = "action"
    NARRATION = "narration"
    SYSTEM = "system"


class Event(BaseModel):
    """Represents a single atomic turn or action in the shared ledger."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    season_id: str
    day: int = Field(ge=1, le=7)
    slot: int = Field(ge=1, le=3)
    turn_number: int = Field(ge=1, le=16)
    actor_id: str
    event_type: EventType
    content: str
    visible_to: List[str]  # List of actor_ids who can see this event
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class DiaryEntry(BaseModel):
    """Represents a tweet-style diary post generated at run completion."""

    diary_id: str = Field(default_factory=lambda: str(uuid4()))
    actor_id: str
    day: int = Field(ge=1, le=7)
    slot: int = Field(ge=1, le=3)
    roll_value: float = Field(ge=0.0, le=1.0)
    content: str = Field(max_length=280)  # Capped to tweet-length
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class DiaryCitation(BaseModel):
    """Explicit relational link between a diary claim and an event ID."""

    citation_id: str = Field(default_factory=lambda: str(uuid4()))
    diary_id: str
    referenced_event_id: str


class ActorTurnOutput(BaseModel):
    """The structured JSON output expected from an actor during roleplay."""

    internal_thought: str
    action_type: EventType
    public_content: str
    target_actors: List[str]

    # Add to src/schema.py

class ClaimEvaluation(str, Enum):
    CORRECT = "correct"
    CLAIMED = "claimed_others_event" # Misattribution
    HALLUCINATED = "hallucinated" # Event never happened
    UNCLEAR = "unclear"

class DiaryEvalRecord(BaseModel):
    """Stores the auto-judge evaluation for a specific diary entry."""
    eval_id: str = Field(default_factory=lambda: str(uuid4()))
    diary_id: str
    claim_text: str
    referenced_event_id: Optional[str] # The actual event ID, if found
    evaluation: ClaimEvaluation
    judge_model_name: str # e.g., "mistral-small3.2:24b"
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )