import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

from schema import (
    ClaimEvaluation,
    Event,
    EventType,
    FailureCode,
    ModelCallDiagnostic,
    PostCitation,
    PostEntry,
    PostEvalRecord,
)

class LedgerDB:
    """SQLite Database Interface for the Durable State Ledger."""

    def __init__(self, db_path: Union[str, Path] = "data/sq_ledger.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection configured with WAL mode and foreign keys enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes database tables and indexes if they do not exist."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    season_id TEXT NOT NULL,
                    day INTEGER NOT NULL,
                    slot INTEGER NOT NULL,
                    turn_number INTEGER NOT NULL,
                    actor_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    visible_to TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS posts (
                    post_id TEXT PRIMARY KEY,
                    season_id TEXT NOT NULL DEFAULT 's2-default',
                    actor_id TEXT NOT NULL,
                    day INTEGER NOT NULL,
                    slot INTEGER NOT NULL,
                    roll_value REAL NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS post_citations (
                    citation_id TEXT PRIMARY KEY,
                    post_id TEXT NOT NULL,
                    referenced_event_id TEXT NOT NULL,
                    FOREIGN KEY(post_id) REFERENCES posts(post_id) ON DELETE CASCADE,
                    FOREIGN KEY(referenced_event_id) REFERENCES events(event_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS post_evaluations (
                    eval_id TEXT PRIMARY KEY,
                    post_id TEXT NOT NULL,
                    claim_text TEXT NOT NULL,
                    referenced_event_id TEXT,
                    evaluation TEXT NOT NULL,
                    failure_code TEXT NOT NULL DEFAULT 'none',
                    judge_model_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(post_id) REFERENCES posts(post_id) ON DELETE CASCADE,
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
                    failure_code TEXT NOT NULL,
                    error_detail TEXT,
                    latency_ms INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_events_lookup 
                ON events(season_id, day, slot, turn_number);
                
                CREATE INDEX IF NOT EXISTS idx_posts_lookup 
                ON posts(actor_id, day, slot);

                CREATE INDEX IF NOT EXISTS idx_model_calls_lookup
                ON model_calls(season_id, day, slot, actor_id);
            """)
            # Ensure older databases have season_id on posts and failure_code on post_evaluations
            post_cols = [r["name"] for r in conn.execute("PRAGMA table_info(posts)").fetchall()]
            if "season_id" not in post_cols:
                conn.execute("ALTER TABLE posts ADD COLUMN season_id TEXT NOT NULL DEFAULT 's2-default'")

            eval_cols = [r["name"] for r in conn.execute("PRAGMA table_info(post_evaluations)").fetchall()]
            if "failure_code" not in eval_cols:
                conn.execute("ALTER TABLE post_evaluations ADD COLUMN failure_code TEXT NOT NULL DEFAULT 'none'")
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

    def record_post(self, post: PostEntry) -> None:
        """Inserts a tweet-style post into the ledger."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO posts (
                    post_id, season_id, actor_id, day, slot,
                    roll_value, content, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    post.post_id,
                    post.season_id,
                    post.actor_id,
                    post.day,
                    post.slot,
                    post.roll_value,
                    post.content,
                    post.created_at.isoformat(),
                ),
            )

    def record_citation(self, citation: PostCitation) -> None:
        """Links a post claim directly to an event ID for explicit provenance."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO post_citations (
                    citation_id, post_id, referenced_event_id
                ) VALUES (?, ?, ?)
                """,
                (
                    citation.citation_id,
                    citation.post_id,
                    citation.referenced_event_id,
                ),
            )

    def record_evaluation(self, eval_record: PostEvalRecord) -> None:
        """Inserts an auto-judge evaluation into the ledger."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO post_evaluations (
                    eval_id, post_id, claim_text, referenced_event_id,
                    evaluation, failure_code, judge_model_name, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    eval_record.eval_id,
                    eval_record.post_id,
                    eval_record.claim_text,
                    eval_record.referenced_event_id,
                    eval_record.evaluation.value,
                    eval_record.failure_code.value if hasattr(eval_record.failure_code, "value") else eval_record.failure_code,
                    eval_record.judge_model_name,
                    eval_record.created_at.isoformat(),
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

    def get_visible_events_for_actor(
        self, season_id: str, actor_id: str, up_to_day: int, up_to_slot: int
    ) -> List[Event]:
        """Queries all events witnessed by an actor up to a specific day and slot."""
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

    def get_last_completed_turn(self, season_id: str) -> Optional[Tuple[int, int, int]]:
        """Returns (day, slot, turn_number) of the latest turn for crash recovery."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT day, slot, turn_number FROM events
                WHERE season_id = ?
                ORDER BY day DESC, slot DESC, turn_number DESC
                LIMIT 1
                """,
                (season_id,),
            ).fetchone()
            return (row["day"], row["slot"], row["turn_number"]) if row else None

    def export_season_telemetry(self, season_id: str, output_dir: Union[str, Path] = "data/exports") -> Path:
        out_path = Path(output_dir) / season_id
        out_path.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            self._write_csv(
                out_path / "events.csv",
                conn.execute("SELECT * FROM events WHERE season_id = ? ORDER BY day, slot, turn_number", (season_id,)).fetchall(),
            )
            self._write_csv(
                out_path / "posts.csv",
                conn.execute("SELECT * FROM posts WHERE season_id = ? ORDER BY day, slot, actor_id", (season_id,)).fetchall(),
            )
            self._write_csv(
                out_path / "model_calls.csv",
                conn.execute("SELECT * FROM model_calls WHERE season_id = ? ORDER BY created_at", (season_id,)).fetchall(),
            )
        return out_path

    @staticmethod
    def _write_csv(file_path: Path, rows: list) -> None:
        if not rows:
            return
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(rows[0].keys())
            for row in rows:
                writer.writerow(list(row))
