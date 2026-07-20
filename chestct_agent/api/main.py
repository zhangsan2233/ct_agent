import asyncio
from contextlib import asynccontextmanager
import json
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from chestct_agent.agent.graph import ChestCtAgent
from chestct_agent.config import get_settings
from chestct_agent.feedback import FeedbackReview, FeedbackSubmission
from chestct_agent.input_ingestion import (
    InputIngestionError,
    decode_report_bytes,
    ingest_ct_upload,
)
from chestct_agent.schemas import (
    AgentState,
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
    CorrectionRequest,
    ExecutionEvent,
    SandboxCorrectionRequest,
)

@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    agent.report_graph.close()


app = FastAPI(
    title="ChestCT-Agent",
    version="0.1.0",
    description="Qwen3.6 + LangGraph Agentic RAG prototype for chest CT evidence integration.",
    lifespan=lifespan,
)

settings = get_settings()
agent = ChestCtAgent(settings)


class ApprovalRequest(BaseModel):
    status: str
    reviewer: str
    note: str = ""


@app.post("/api/cases/{case_id}/feedback")
def submit_feedback(case_id: str, submission: FeedbackSubmission) -> dict[str, object]:
    try:
        return {"case_id": case_id, "events": agent.memory.submit_feedback(case_id, submission)}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/feedback")
def list_feedback(status: str | None = None, limit: int = 100) -> dict[str, object]:
    return {"events": agent.memory.list_feedback(status=status, limit=limit)}


@app.post("/api/feedback/{event_id}/review")
def review_feedback(event_id: str, review: FeedbackReview) -> dict[str, str]:
    try:
        return agent.memory.review_feedback(event_id, review.status, review.reviewer, review.note)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, object]:
    ct_error = agent.ct_classifier.readiness_error()
    return {
        "status": "ok",
        "agent_model": settings.agent_model,
        "model_backend": settings.model_backend,
        "ct_model_backend": settings.ct_model_backend,
        "ct_model_variant": settings.ctclip_variant,
        "ct_model_checkpoint": settings.ctclip_checkpoint.name,
        "ct_model_ready": ct_error is None,
        "ct_model_error": ct_error,
        "rag_backend": settings.embedding_backend,
        "knowledge_documents": len(agent.medical_rag.documents),
        "similar_case_documents": len(agent.similar_cases.index),
        "similar_case_index_ready": agent.similar_cases.report_matrix is not None,
        "dynamic_planning": settings.agent_dynamic_planning,
        "report_graph_model": settings.radgraph_model_type,
        "report_graph_ready": agent.report_graph.readiness_error() is None,
        "report_graph_error": agent.report_graph.readiness_error(),
        "radgenome_index_ready": settings.radgenome_index_path.exists(),
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    state = AgentState(request=request)
    return await agent.run(state)


@app.post("/api/analyze/stream")
async def analyze_stream(request: AnalyzeRequest) -> StreamingResponse:
    async def stream():
        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()

        async def publish(event: ExecutionEvent) -> None:
            await queue.put({"type": "node", "event": event.model_dump(mode="json")})

        async def run_agent() -> None:
            try:
                response = await agent.run(
                    AgentState(request=request), event_callback=publish
                )
                await queue.put(
                    {"type": "result", "response": response.model_dump(mode="json")}
                )
            except Exception as exc:
                await queue.put(
                    {
                        "type": "error",
                        "message": f"Agent execution failed: {type(exc).__name__}",
                    }
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(run_agent())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield json.dumps(item, ensure_ascii=False) + "\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/analyze/upload", response_model=AnalyzeResponse)
async def analyze_upload(
    case_id: Annotated[str, Form()] = "uploaded_case",
    session_id: Annotated[str, Form()] = "",
    question: Annotated[str, Form()] = "请分析该检查中的胸部异常和证据。",
    report_text: Annotated[str, Form()] = "",
    require_human_approval: Annotated[bool, Form()] = False,
    ct_file: Annotated[UploadFile | None, File()] = None,
    report_file: Annotated[UploadFile | None, File()] = None,
) -> AnalyzeResponse:
    request = await _prepare_upload_request(
        case_id,
        session_id,
        question,
        report_text,
        require_human_approval,
        ct_file,
        report_file,
    )
    return await agent.run(AgentState(request=request))


async def _prepare_upload_request(
    case_id: str,
    session_id: str,
    question: str,
    report_text: str,
    require_human_approval: bool,
    ct_file: UploadFile | None,
    report_file: UploadFile | None,
) -> AnalyzeRequest:
    try:
        report_parts = [report_text.strip()] if report_text.strip() else []
        if report_file is not None:
            report_parts.append(decode_report_bytes(await report_file.read()))
        merged_report = "\n".join(part for part in report_parts if part).strip()

        ct_volume_path = None
        if ct_file is not None:
            ct_volume_path = str(
                ingest_ct_upload(
                    ct_file.filename or "uploaded_ct",
                    await ct_file.read(),
                    case_id,
                    settings.upload_dir,
                )
            )
        if not merged_report and not ct_volume_path:
            raise InputIngestionError("请至少上传 CT 或提供影像报告。")
    except InputIngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AnalyzeRequest(
        case_id=case_id,
        session_id=session_id.strip() or None,
        report_text=merged_report,
        question=question,
        ct_volume_path=ct_volume_path,
        ct_source_name=ct_file.filename if ct_file is not None else None,
        require_human_approval=require_human_approval,
    )


@app.post("/api/analyze/upload/stream")
async def analyze_upload_stream(
    case_id: Annotated[str, Form()] = "uploaded_case",
    session_id: Annotated[str, Form()] = "",
    question: Annotated[str, Form()] = "请分析该检查中的胸部异常和证据。",
    report_text: Annotated[str, Form()] = "",
    require_human_approval: Annotated[bool, Form()] = False,
    ct_file: Annotated[UploadFile | None, File()] = None,
    report_file: Annotated[UploadFile | None, File()] = None,
) -> StreamingResponse:
    request = await _prepare_upload_request(
        case_id,
        session_id,
        question,
        report_text,
        require_human_approval,
        ct_file,
        report_file,
    )

    async def stream():
        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()

        async def publish(event: ExecutionEvent) -> None:
            await queue.put({"type": "node", "event": event.model_dump(mode="json")})

        async def run_agent() -> None:
            try:
                response = await agent.run(AgentState(request=request), event_callback=publish)
                await queue.put(
                    {"type": "result", "response": response.model_dump(mode="json")}
                )
            except Exception as exc:
                await queue.put(
                    {
                        "type": "error",
                        "message": f"Agent execution failed: {type(exc).__name__}",
                    }
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(run_agent())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield json.dumps(item, ensure_ascii=False) + "\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        return await agent.chat(request)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    async def stream():
        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()

        async def publish(event: ExecutionEvent) -> None:
            await queue.put({"type": "node", "event": event.model_dump(mode="json")})

        async def run_agent() -> None:
            try:
                response = await agent.chat(request, event_callback=publish)
                await queue.put(
                    {"type": "result", "response": response.model_dump(mode="json")}
                )
            except LookupError as exc:
                await queue.put({"type": "error", "message": str(exc)})
            except Exception as exc:
                await queue.put(
                    {
                        "type": "error",
                        "message": f"Follow-up agent failed: {type(exc).__name__}",
                    }
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(run_agent())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield json.dumps(item, ensure_ascii=False) + "\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/api/sessions/{session_id}/cases/{case_id}/messages")
def conversation_messages(session_id: str, case_id: str) -> dict[str, object]:
    return {
        "session_id": session_id,
        "case_id": case_id,
        "messages": [
            item.model_dump(mode="json")
            for item in agent.memory.get_messages(session_id, case_id, limit=50)
        ],
    }


@app.delete("/api/sessions/{session_id}/cases/{case_id}/messages")
def clear_conversation(session_id: str, case_id: str) -> dict[str, object]:
    agent.memory.clear_conversation(session_id, case_id)
    return {"session_id": session_id, "case_id": case_id, "cleared": True}


@app.post("/api/cases/{case_id}/approval")
def set_approval(case_id: str, request: ApprovalRequest) -> dict[str, str]:
    if request.status not in {"approved", "rejected"}:
        raise HTTPException(status_code=400, detail="Approval status must be approved or rejected.")
    agent.memory.set_approval(case_id, request.status, request.reviewer, request.note)
    return {"case_id": case_id, "status": request.status}


@app.get("/api/cases/{case_id}/approval")
def get_approval(case_id: str) -> dict[str, object]:
    return {"case_id": case_id, "approval": agent.memory.get_approval(case_id)}


@app.post("/api/cases/{case_id}/corrections", response_model=AnalyzeResponse)
async def apply_case_corrections(
    case_id: str, request: CorrectionRequest
) -> AnalyzeResponse:
    if request.source != "human":
        raise HTTPException(
            status_code=400,
            detail="Use the sandbox endpoint for dataset weak-label feedback.",
        )
    try:
        return await agent.correct_case(case_id, request)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/cases/{case_id}/sandbox-correct", response_model=AnalyzeResponse)
async def apply_dataset_sandbox_correction(
    case_id: str, request: SandboxCorrectionRequest
) -> AnalyzeResponse:
    try:
        return await agent.correct_case_with_dataset(case_id, request.session_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}/cases/{case_id}/corrections")
def get_case_corrections(session_id: str, case_id: str) -> dict[str, object]:
    return {
        "session_id": session_id,
        "case_id": case_id,
        "events": [
            item.model_dump(mode="json")
            for item in agent.memory.get_corrections(session_id, case_id)
        ],
    }
