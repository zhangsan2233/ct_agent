from fastapi.testclient import TestClient

from chestct_agent.api.main import agent, app
from chestct_agent.llm import QwenClient
from chestct_agent.memory import AgentMemory
from chestct_agent.schemas import AnalyzeRequest, AnalyzeResponse, ExecutionMetadata, LabelOutput


def test_approval_round_trip(tmp_path, monkeypatch):
    settings = agent.settings.model_copy(update={"memory_db_path": tmp_path / "memory.sqlite3"})
    monkeypatch.setattr(agent, "memory", AgentMemory(settings))
    client = TestClient(app)

    written = client.post(
        "/api/cases/approval-test/approval",
        json={"status": "approved", "reviewer": "test-reviewer", "note": "reviewed"},
    )
    read = client.get("/api/cases/approval-test/approval")

    assert written.status_code == 200
    assert read.status_code == 200
    assert read.json()["approval"]["status"] == "approved"
    assert read.json()["approval"]["reviewer"] == "test-reviewer"


def test_upload_endpoint_requires_ct_or_report():
    response = TestClient(app).post(
        "/api/analyze/upload",
        data={"case_id": "empty-upload", "question": "有哪些异常？"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "请至少上传 CT 或提供影像报告。"

    stream_response = TestClient(app).post(
        "/api/analyze/upload/stream",
        data={"case_id": "empty-stream", "question": "有哪些异常？"},
    )
    assert stream_response.status_code == 400


def test_human_correction_endpoint_updates_case_and_exposes_history(tmp_path, monkeypatch):
    settings = agent.settings.model_copy(
        update={
            "memory_db_path": tmp_path / "memory.sqlite3",
            "openai_compatible_api_key": "replace-me",
        }
    )
    memory = AgentMemory(settings)
    monkeypatch.setattr(agent, "memory", memory)
    monkeypatch.setattr(agent, "qwen", QwenClient(settings))
    request = AnalyzeRequest(
        case_id="correction-api",
        session_id="correction-session",
        report_text="Findings: No pleural effusion.",
    )
    response = AnalyzeResponse(
        case_id="correction-api",
        labels=[
            LabelOutput(name="pleural_effusion", status="positive", confidence=0.8),
            LabelOutput(name="pulmonary_nodule", status="negative", confidence=0.1),
        ],
        disclaimer="research only",
        execution=ExecutionMetadata(input_mode="report_only"),
    )
    memory.record(request, response, None)
    client = TestClient(app)

    written = client.post(
        "/api/cases/correction-api/corrections",
        json={
            "session_id": "correction-session",
            "reviewer": "doctor-api",
            "source": "human",
            "corrections": [
                {
                    "label": "pleural_effusion",
                    "corrected_status": "negative",
                    "reason": "reviewed CT",
                },
                {
                    "label": "pulmonary_nodule",
                    "corrected_status": "negative",
                    "reason": "confirmed",
                },
            ],
        },
    )
    history = client.get(
        "/api/sessions/correction-session/cases/correction-api/corrections"
    )

    assert written.status_code == 200
    assert written.json()["labels"][0]["status"] == "negative"
    assert written.json()["labels"][0]["decision_source"] == "human_correction"
    assert history.status_code == 200
    assert history.json()["events"][0]["reviewer"] == "doctor-api"


def test_case_conversation_keeps_history_and_streams_steps(tmp_path, monkeypatch):
    settings = agent.settings.model_copy(
        update={
            "memory_db_path": tmp_path / "memory.sqlite3",
            "openai_compatible_api_key": "replace-me",
        }
    )
    memory = AgentMemory(settings)
    monkeypatch.setattr(agent, "memory", memory)
    monkeypatch.setattr(agent.conversation, "qwen", QwenClient(settings))
    request = AnalyzeRequest(
        case_id="chat-case",
        session_id="chat-session",
        report_text="Findings: A small right pleural effusion is present.",
    )
    response = AnalyzeResponse(
        case_id="chat-case",
        labels=[
            LabelOutput(
                name="pleural_effusion",
                status="positive",
                confidence=0.82,
            )
        ],
        explanation_zh="主要发现为胸腔积液。",
        disclaimer="research only",
        execution=ExecutionMetadata(input_mode="report_only"),
    )
    memory.record(request, response, None)
    client = TestClient(app)

    first = client.post(
        "/api/chat",
        json={
            "session_id": "chat-session",
            "case_id": "chat-case",
            "message": "主要发现是什么？",
        },
    )
    second = client.post(
        "/api/chat/stream",
        json={
            "session_id": "chat-session",
            "case_id": "chat-case",
            "message": "刚才说的发现有什么报告证据？",
        },
    )
    history = client.get("/api/sessions/chat-session/cases/chat-case/messages")

    assert first.status_code == 200
    assert "胸腔积液" in first.json()["answer_zh"]
    assert second.status_code == 200
    assert '"status": "running"' in second.text
    assert '"type": "result"' in second.text
    assert history.status_code == 200
    assert [item["role"] for item in history.json()["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
