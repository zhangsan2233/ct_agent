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
        self.settings = settings
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
            feedback_columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(feedback_events)").fetchall()
            }
            if "annotations_json" not in feedback_columns:
                connection.execute(
                    "ALTER TABLE feedback_events ADD COLUMN annotations_json TEXT NOT NULL DEFAULT '[]'"
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
                CREATE TABLE IF NOT EXISTS experience_memories (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    fold INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    source_case_ids_json TEXT NOT NULL,
                    memory_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_experience_memory_experiment_label
                ON experience_memories(experiment_id, label, fold)
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
                        status, response_snapshot_json, annotations_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                    """,
                    (event_id, created_at, submission.session_id, case_id, submission.reviewer,
                     submission.reviewer_role, submission.model_version, item.label,
                     labels[item.label], item.corrected_status, item.reason,
                     response.model_dump_json(),
                     json.dumps(
                         [annotation.model_dump() for annotation in item.annotations],
                         ensure_ascii=False,
                     )),
                )
                records.append({"id": event_id, "status": "pending", "label": item.label})
        return records

    def list_feedback(self, status: str | None = None, limit: int = 100) -> list[dict[str, str]]:
        query = "SELECT id, created_at, session_id, case_id, reviewer, reviewer_role, model_version, label, before_status, corrected_status, reason, status, reviewed_by, reviewed_at, review_note, annotations_json FROM feedback_events"
        params: tuple[object, ...] = ()
        if status:
            query += " WHERE status=?"
            params = (status,)
        query += " ORDER BY created_at DESC LIMIT ?"
        with self._connect() as connection:
            rows = connection.execute(query, (*params, max(1, min(limit, 500)))).fetchall()
        keys = ["id", "created_at", "session_id", "case_id", "reviewer", "reviewer_role", "model_version", "label", "before_status", "corrected_status", "reason", "status", "reviewed_by", "reviewed_at", "review_note", "annotations_json"]
        records = [dict(zip(keys, row, strict=True)) for row in rows]
        for record in records:
            record["annotations"] = json.loads(record.pop("annotations_json") or "[]")
        return records

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
        result = {"id": event_id, "status": status, "reviewed_by": reviewer, "reviewed_at": reviewed_at}
        if status == "approved":
            with self._connect() as connection:
                row = connection.execute(
                    """SELECT case_id, label, before_status, corrected_status, reason,
                    annotations_json FROM feedback_events WHERE id=?""",
                    (event_id,),
                ).fetchone()
            if row is not None:
                case_id, label, before_status, corrected_status, reason, annotations_json = row
                memory_id = self.record_experience_memory(
                    experiment_id=self.settings.experience_memory_experiment_id,
                    fold=-1,
                    label=label,
                    source_case_ids=[case_id],
                    memory={
                        "feedback_event_id": event_id,
                        "lesson": reason,
                        "visual_lesson": reason,
                        "before_status": before_status,
                        "corrected_status": corrected_status,
                        "annotations": json.loads(annotations_json or "[]"),
                        "scope": "医生审核通过的病例级经验；仅用于复查策略，不替代当前病例证据。",
                        "prohibition": "Memory is review policy context, never patient evidence.",
                    },
                )
                result["memory_id"] = memory_id
        return result

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

    def record_experience_memory(
        self,
        *,
        experiment_id: str,
        fold: int,
        label: str,
        source_case_ids: list[str],
        memory: dict,
    ) -> str:
        memory_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO experience_memories(
                    id, created_at, experiment_id, fold, label,
                    source_case_ids_json, memory_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    datetime.now(timezone.utc).isoformat(),
                    experiment_id,
                    int(fold),
                    label,
                    json.dumps(source_case_ids, ensure_ascii=False),
                    json.dumps(memory, ensure_ascii=False),
                ),
            )
        return memory_id

    def list_experience_memories(
        self,
        *,
        experiment_id: str,
        label: str | None = None,
        fold: int | None = None,
    ) -> list[dict]:
        clauses = ["experiment_id=?"]
        params: list[object] = [experiment_id]
        if label is not None:
            clauses.append("label=?")
            params.append(label)
        if fold is not None:
            clauses.append("fold=?")
            params.append(int(fold))
        query = (
            "SELECT id, created_at, experiment_id, fold, label, "
            "source_case_ids_json, memory_json FROM experience_memories WHERE "
            + " AND ".join(clauses)
            + " ORDER BY fold, label, created_at"
        )
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            {
                "id": row[0],
                "created_at": row[1],
                "experiment_id": row[2],
                "fold": row[3],
                "label": row[4],
                "source_case_ids": json.loads(row[5]),
                "memory": json.loads(row[6]),
            }
            for row in rows
        ]

    def experience_prompt_context(
        self,
        *,
        experiment_id: str,
        labels: list[str],
        fold: int = -1,
        limit: int = 9,
    ) -> tuple[list[dict], list[str]]:
        selected: list[dict] = []
        memory_ids: list[str] = []
        for label in labels:
            records = self.list_experience_memories(
                experiment_id=experiment_id,
                label=label,
                fold=fold,
            )
            if not records:
                continue
            record = records[-1]
            value = record["memory"]
            selected.append(
                {
                    "label": label,
                    "lesson": value.get("visual_lesson") or value.get("lesson", ""),
                    "decision_threshold": value.get("threshold"),
                    "false_positives_seen": value.get("false_positives_at_0_5"),
                    "false_negatives_seen": value.get("false_negatives_at_0_5"),
                    "scope": value.get("scope", ""),
                    "prohibition": value.get(
                        "prohibition",
                        "Memory is policy context, never evidence for this patient.",
                    ),
                }
            )
            memory_ids.append(record["id"])
            if len(selected) >= limit:
                break
        return selected, memory_ids

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
