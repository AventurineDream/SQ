import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Union

from schema import (
    DiaryCitation,
    DiaryEntry,
    DiaryEvalRecord,
    Event,
    EventType,
    ModelCallDiagnostic,
)


class LedgerDB:
    """SQLite Database Interface for the Durable State Ledger.

    Provides thread-safe connections, schema creation, and typed Pydantic
    CRUD operations for events, diaries, and citations.
    """

    def __init__(self, db_path: Union[str, Path] = "data/sq_ledger.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection configured with WAL mode and foreign keys enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")  # Write-Ahead Logging for durability/concurrency
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes database tables and indexes if they do not exist."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    season_id TEXT NOT NULL,
                    day INTEGER NOT NULL,
                    slot INTEGER NOT NULL,
                    turn_number INTEGER NOT NULL,
                    actor_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    visible_to TEXT NOT NULL, -- Stored as JSON array
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS diaries (
                    diary_id TEXT PRIMARY KEY,
                    actor_id TEXT NOT NULL,
                    day INTEGER NOT NULL,
                    slot INTEGER NOT NULL,
                    roll_value REAL NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS diary_citations (
                    citation_id TEXT PRIMARY KEY,
                    diary_id TEXT NOT NULL,
                    referenced_event_id TEXT NOT NULL,
                    FOREIGN KEY(diary_id) REFERENCES diaries(diary_id) ON DELETE CASCADE,
                    FOREIGN KEY(referenced_event_id) REFERENCES events(event_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS diary_evaluations (
                    eval_id TEXT PRIMARY KEY,
                    diary_id TEXT NOT NULL,
                    claim_text TEXT NOT NULL,
                    referenced_event_id TEXT,
                    evaluation TEXT NOT NULL,
                    judge_model_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(diary_id) REFERENCES diaries(diary_id) ON DELETE CASCADE,
                    FOREIGN KEY(referenced_event_id) REFERENCES events(event_id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS model_calls (
                    call_id TEXT PRIMARY KEY,
                    season_id TEXT NOT NULL,
                    day INTEGER NOT NULL,
                    slot INTEGER NOT NULL,
                    turn_number INTEGER NOT NULL,
                    actor_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    call_type TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    accepted INTEGER NOT NULL,
                    failure_code TEXT,
                    error_detail TEXT,
                    latency_ms INTEGER,
                    created_at TEXT NOT NULL
                );

                -- Performance Indexes for fast context assembly and evaluation queries
                CREATE INDEX IF NOT EXISTS idx_events_lookup
                ON events(season_id, day, slot, turn_number);

                CREATE INDEX IF NOT EXISTS idx_diaries_lookup
                ON diaries(actor_id, day, slot);
                """
            )

    def record_evaluation(self, eval_record: DiaryEvalRecord) -> None:
        """Inserts an auto-judge evaluation into the ledger."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO diary_evaluations (
                    eval_id, diary_id, claim_text, referenced_event_id,
                    evaluation, judge_model_name, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    eval_record.eval_id,
                    eval_record.diary_id,
                    eval_record.claim_text,
                    eval_record.referenced_event_id,
                    eval_record.evaluation.value,
                    eval_record.judge_model_name,
                    eval_record.created_at.isoformat(),
                ),
            )

    def record_event(self, event: Event) -> None:
        """Inserts an immutable event into the ledger."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO events (
                    event_id, season_id, day, slot, turn_number,
                    actor_id, event_type, content, visible_to, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.season_id,
                    event.day,
                    event.slot,
                    event.turn_number,
                    event.actor_id,
                    event.event_type.value if isinstance(event.event_type, EventType) else event.event_type,
                    event.content,
                    json.dumps(event.visible_to),
                    event.created_at.isoformat(),
                ),
            )

    def record_model_call(self, diagnostic: ModelCallDiagnostic) -> None:
        """Records model call attempt telemetry for audition & retry analysis."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO model_calls (
                    call_id, season_id, day, slot, turn_number, actor_id,
                    model_name, call_type, attempt, accepted, failure_code,
                    error_detail, latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    diagnostic.call_id,
                    diagnostic.season_id,
                    diagnostic.day,
                    diagnostic.slot,
                    diagnostic.turn_number,
                    diagnostic.actor_id,
                    diagnostic.model_name,
                    diagnostic.call_type,
                    diagnostic.attempt,
                    1 if diagnostic.accepted else 0,
                    diagnostic.failure_code.value if hasattr(diagnostic.failure_code, "value") else diagnostic.failure_code,
                    diagnostic.error_detail,
                    diagnostic.latency_ms,
                    diagnostic.created_at.isoformat(),
                ),
            )

    def record_diary(self, diary: DiaryEntry) -> None:
        """Inserts a tweet-style diary entry into the ledger."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO diaries (
                    diary_id, actor_id, day, slot, roll_value, content, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    diary.diary_id,
                    diary.actor_id,
                    diary.day,
                    diary.slot,
                    diary.roll_value,
                    diary.content,
                    diary.created_at.isoformat(),
                ),
            )

    def record_citation(self, citation: DiaryCitation) -> None:
        """Links a diary claim directly to an event ID for explicit provenance."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO diary_citations (
                    citation_id, diary_id, referenced_event_id
                ) VALUES (?, ?, ?)
                """,
                (
                    citation.citation_id,
                    citation.diary_id,
                    citation.referenced_event_id,
                ),
            )

    def get_visible_events_for_actor(
        self, season_id: str, actor_id: str, up_to_day: int, up_to_slot: int
    ) -> List[Event]:
        """Queries all events witnessed by an actor up to a specific day and slot.

        Feeds directly into the ContextAssembler to eliminate context truncation.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM events
                WHERE season_id = ?
                  AND (day < ? OR (day = ? AND slot <= ?))
                ORDER BY day ASC, slot ASC, turn_number ASC
                """,
                (season_id, up_to_day, up_to_day, up_to_slot),
            )

            rows = cursor.fetchall()
            visible_events = []

            for row in rows:
                visible_list = json.loads(row["visible_to"])
                # Filter by actor visibility permissions
                if actor_id in visible_list or "all" in visible_list:
                    visible_events.append(
                        Event(
                            event_id=row["event_id"],
                            season_id=row["season_id"],
                            day=row["day"],
                            slot=row["slot"],
                            turn_number=row["turn_number"],
                            actor_id=row["actor_id"],
                            event_type=EventType(row["event_type"]),
                            content=row["content"],
                            visible_to=visible_list,
                            created_at=datetime.fromisoformat(row["created_at"]),
                        )
                    )
            return visible_events


        