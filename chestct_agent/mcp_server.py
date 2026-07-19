from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP

from chestct_agent.agent.graph import ChestCtAgent
from chestct_agent.config import get_settings
from chestct_agent.schemas import AgentState, AnalyzeRequest
from chestct_agent.tools.report_parser import parse_report


mcp = FastMCP("ChestCT-Agent")
settings = get_settings()
agent = ChestCtAgent(settings)


def _local_case(case_id: str) -> dict[str, str]:
    manifest_path = Path(settings.artifact_dir) / "evaluation" / "multimodal_manifest.csv"
    if not manifest_path.exists():
        raise ValueError("Local multimodal manifest is missing.")
    frame = pd.read_csv(manifest_path).fillna("")
    rows = frame[frame["case_id"].astype(str).eq(case_id)]
    if rows.empty:
        raise ValueError(f"Unknown local case: {case_id}")
    row = rows.iloc[0]
    ct_path = Path(str(row["ct_volume_path"])).resolve()
    data_root = Path(settings.data_dir).resolve()
    if data_root not in ct_path.parents:
        raise ValueError("Case CT path is outside the configured data directory.")
    return {"report_text": str(row["report_text"]), "ct_volume_path": str(ct_path)}


@mcp.tool()
def parse_radiology_report(report_text: str) -> dict:
    """Parse a report into findings, impression, and full text."""
    return parse_report(report_text).model_dump()


@mcp.tool()
async def search_medical_knowledge(query: str, top_k: int = 5) -> dict:
    """Run the BM25+dense+reranker medical retriever."""
    documents, backend = await agent.medical_rag.retrieve([query], top_k=min(max(top_k, 1), 10))
    return {"backend": backend, "documents": [item.model_dump() for item in documents]}


@mcp.tool()
async def analyze_report(report_text: str, question: str = "报告中有哪些胸部异常？") -> dict:
    """Analyze report text without accepting an arbitrary local file path."""
    request = AnalyzeRequest(report_text=report_text, question=question)
    response = await agent.run(AgentState(request=request))
    return response.model_dump(mode="json")


@mcp.tool()
async def analyze_local_case(case_id: str, question: str = "该病例有哪些异常及证据？") -> dict:
    """Analyze a case already present in the allow-listed local manifest."""
    case = _local_case(case_id)
    request = AnalyzeRequest(
        case_id=case_id,
        report_text=case["report_text"],
        ct_volume_path=case["ct_volume_path"],
        question=question,
    )
    response = await agent.run(AgentState(request=request))
    return response.model_dump(mode="json")


@mcp.tool()
def review_case(case_id: str, approved: bool, reviewer: str, note: str = "") -> dict:
    """Record an explicit human approval or rejection."""
    status = "approved" if approved else "rejected"
    agent.memory.set_approval(case_id, status, reviewer, note)
    return {"case_id": case_id, "status": status}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
