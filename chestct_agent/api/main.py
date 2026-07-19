from fastapi import FastAPI

from chestct_agent.agent.graph import ChestCtAgent
from chestct_agent.config import get_settings
from chestct_agent.schemas import AgentState, AnalyzeRequest, AnalyzeResponse

app = FastAPI(
    title="ChestCT-Agent",
    version="0.1.0",
    description="Qwen3.6 + LangGraph Agentic RAG prototype for chest CT evidence integration.",
)

settings = get_settings()
agent = ChestCtAgent(settings)


@app.get("/health")
def health() -> dict[str, object]:
    ct_error = agent.ct_classifier.readiness_error()
    return {
        "status": "ok",
        "agent_model": settings.agent_model,
        "model_backend": settings.model_backend,
        "ct_model_backend": settings.ct_model_backend,
        "ct_model_ready": ct_error is None,
        "ct_model_error": ct_error,
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    state = AgentState(request=request)
    return await agent.run(state)
