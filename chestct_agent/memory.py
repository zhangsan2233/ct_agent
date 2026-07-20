from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid

from chestct_agent.config import Settings
from chestct_agent.feedback import FeedbackSubmission
from chestct_agent.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    CorrectionEvent,
    ConversationMessage,
    ToolPlan,
)


class AgentMemory:
    """Local audit, case-context, and conversation memory backed by SQLite."""

    def __init__(self, settings: Settings):
        self.path = Path(settings.memory_db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    session_id TEXT,
                    case_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    planned_tools_json TEXT NOT NULL,
                    labels_json TEXT NOT NULL,
                    warning_count INTEGER NOT NULL,
                    approval_status TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    reviewer_role TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    label TEXT NOT NULL,
                    before_status TEXT NOT NULL,
                    corrected_status TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reviewed_by TEXT,
                    reviewed_at TEXT,
                    review_note TEXT NOT NULL DEFAULT '',
                    response_snapshot_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feedback_status_created
                ON feedback_events(status, created_at)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS approvals (
                    case_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    reviewer TEXT,
                    note TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS case_contexts (
                    session_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, case_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_session_case
                ON conversation_messages(session_id, case_id, id)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS correction_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    before_response_json TEXT NOT NULL,
                    after_response_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_correction_session_case
                ON correction_events(session_id, case_id, id)
                """
            )

    def record(self, request: AnalyzeRequest, response: AnalyzeResponse, plan: ToolPlan | None) -> None:
        labels = [
            {"name": item.name, "status": item.status, "confidence": item.confidence}
            for item in response.labels
            if item.status != "negative"
        ]
        planned_tools = [step.tool for step in plan.steps] if plan else []
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO run_summaries (
                    created_at, session_id, case_id, question, planned_tools_json,
                    labels_json, warning_count, approval_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    request.session_id,
                    request.case_id,
                    request.question,
                    json.dumps(planned_tools),
                    json.dumps(labels),
                    len(response.warnings),
                    response.approval.status,
                ),
            )
            if request.session_id:
                connection.execute(
                    """
                    INSERT INTO case_contexts(
                        session_id, case_id, request_json, response_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, case_id) DO UPDATE SET
                        request_json=excluded.request_json,
                        response_json=excluded.response_json,
                        updated_at=excluded.updated_at
                    """,
                    (
                        request.session_id,
                        request.case_id,
                        request.model_dump_json(),
                        response.model_dump_json(),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )

    def get_case_context(
        self, session_id: str, case_id: str
    ) -> tuple[AnalyzeRequest, AnalyzeResponse] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT request_json, response_json
                FROM case_contexts
                WHERE session_id=? AND case_id=?
                """,
                (session_id, case_id),
            ).fetchone()
        if row is None:
            return None
        return (
            AnalyzeRequest.model_validate_json(row[0]),
            AnalyzeResponse.model_validate_json(row[1]),
        )

    def record_correction(
        self,
        session_id: str,
        case_id: str,
        event: CorrectionEvent,
        before: AnalyzeResponse,
        after: AnalyzeResponse,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO correction_events(
                    created_at, session_id, case_id, source, reviewer, event_json,
                    before_response_json, after_response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.created_at,
                    session_id,
                    case_id,
                    event.source,
                    event.reviewer,
                    event.model_dump_json(),
                    before.model_dump_json(),
                    after.model_dump_json(),
                ),
            )
            connection.execute(
                """
                UPDATE case_contexts
                SET response_json=?, updated_at=?
                WHERE session_id=? AND case_id=?
                """,
                (
                    after.model_dump_json(),
                    datetime.now(timezone.utc).isoformat(),
                    session_id,
                    case_id,
                ),
            )

    def get_corrections(self, session_id: str, case_id: str) -> list[CorrectionEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_json
                FROM correction_events
                WHERE session_id=? AND case_id=?
                ORDER BY id
                """,
                (session_id, case_id),
            ).fetchall()
        return [CorrectionEvent.model_validate_json(row[0]) for row in rows]

    def submit_feedback(self, case_id: str, submission: FeedbackSubmission) -> list[dict[str, str]]:
        context = self.get_case_context(submission.session_id, case_id)
        if context is None:
            raise LookupError(f"No stored case context for {case_id} in session {submission.session_id}.")
        _, response = context
        labels = {item.name: item.status for item in response.labels}
        unknown = sorted({item.label for item in submission.items} - set(labels))
        if unknown:
            raise ValueError("Unknown feedback labels: " + ", ".join(unknown))
        if len({item.label for item in submission.items}) != len(submission.items):
            raise ValueError("Feedback labels must be unique.")
        created_at = datetime.now(timezone.utc).isoformat()
        records: list[dict[str, str]] = []
        with self._connect() as connection:
            for item in submission.items:
                event_id = str(uuid.uuid4())
                connection.execute(
                    """
                    INSERT INTO feedback_events(
                        id, created_at, session_id, case_id, reviewer, reviewer_role,
                        model_version, label, before_status, corrected_status, reason,
                        status, response_snapshot_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    (event_id, created_at, submission.session_id, case_id, submission.reviewer,
                     submission.reviewer_role, submission.model_version, item.label,
                     labels[item.label], item.corrected_status, item.reason,
                     response.model_dump_json()),
                )
                records.append({"id": event_id, "status": "pending", "label": item.label})
        return records

    def list_feedback(self, status: str | None = None, limit: int = 100) -> list[dict[str, str]]:
        query = "SELECT id, created_at, session_id, case_id, reviewer, reviewer_role, model_version, label, before_status, corrected_status, reason, status, reviewed_by, reviewed_at, review_note FROM feedback_events"
        params: tuple[object, ...] = ()
        if status:
            query += " WHERE status=?"
            params = (status,)
        query += " ORDER BY created_at DESC LIMIT ?"
        with self._connect() as connection:
            rows = connection.execute(query, (*params, max(1, min(limit, 500)))).fetchall()
        keys = ["id", "created_at", "session_id", "case_id", "reviewer", "reviewer_role", "model_version", "label", "before_status", "corrected_status", "reason", "status", "reviewed_by", "reviewed_at", "review_note"]
        return [dict(zip(keys, row, strict=True)) for row in rows]

    def review_feedback(self, event_id: str, status: str, reviewer: str, note: str = "") -> dict[str, str]:
        reviewed_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            updated = connection.execute(
                """UPDATE feedback_events SET status=?, reviewed_by=?, reviewed_at=?, review_note=?
                WHERE id=? AND status='pending'""",
                (status, reviewer, reviewed_at, note, event_id),
            ).rowcount
        if not updated:
            raise LookupError(f"Pending feedback event not found: {event_id}")
        return {"id": event_id, "status": status, "reviewed_by": reviewer, "reviewed_at": reviewed_at}

    def append_message(
        self, session_id: str, case_id: str, role: str, content: str
    ) -> ConversationMessage:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversation_messages(
                    session_id, case_id, role, content, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, case_id, role, content, created_at),
            )
        return ConversationMessage(role=role, content=content, created_at=created_at)

    def get_messages(
        self, session_id: str, case_id: str, limit: int = 20
    ) -> list[ConversationMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content, created_at
                FROM conversation_messages
                WHERE session_id=? AND case_id=?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, case_id, max(1, limit)),
            ).fetchall()
        return [
            ConversationMessage(role=row[0], content=row[1], created_at=row[2])
            for row in reversed(rows)
        ]

    def clear_conversation(self, session_id: str, case_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM conversation_messages
                WHERE session_id=? AND case_id=?
                """,
                (session_id, case_id),
            )

    def set_approval(self, case_id: str, status: str, reviewer: str, note: str = "") -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO approvals(case_id, status, reviewer, note, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    status=excluded.status,
                    reviewer=excluded.reviewer,
                    note=excluded.note,
                    updated_at=excluded.updated_at
                """,
                (
                    case_id,
                    status,
                    reviewer,
                    note,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_approval(self, case_id: str) -> dict[str, str] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status, reviewer, note, updated_at FROM approvals WHERE case_id=?",
                (case_id,),
            ).fetchone()
        if row is None:
            return None
        return {"status": row[0], "reviewer": row[1] or "", "note": row[2] or "", "updated_at": row[3]}
