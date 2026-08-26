"""Deterministic Chinese narrative for Stage-2 JSON outputs (CT and CXR)."""
from __future__ import annotations

from typing import Any

from chestct_agent.knowledge import STATUS_ZH
from chestct_agent.labels import LABEL_ZH
from chestct_agent.stage2_contract import CXR_APPLICABILITY, DISCLAIMER, LABELS


def build_report_zh(
    stage2_json: dict[str, Any] | None,
    *,
    modality: str,
    report_text: str,
    scores: dict[str, float],
) -> str:
    """Build a multi-paragraph Chinese report aligned with the full Agent explanation style."""
    if not stage2_json or not isinstance(stage2_json.get("labels"), list):
        return (
            "模型未能生成可解析的结构化 JSON，请结合原始影像与报告人工复核。"
            f"{DISCLAIMER}"
        )
    by_name = {
        item.get("name"): item for item in stage2_json["labels"] if isinstance(item, dict)
    }
    positive = [name for name in LABELS if by_name.get(name, {}).get("status") == "positive"]
    uncertain = [
        name
        for name in LABELS
        if by_name.get(name, {}).get("status") == "uncertain"
    ]
    parts: list[str] = []
    modality_zh = "胸部 CT" if modality == "ct_chest" else "胸部 X 光"
    if positive:
        parts.append(
            "主要结论："
            + "；".join(
                f"{LABEL_ZH.get(name, name)}为阳性"
                f"（融合分数 {float(by_name[name].get('confidence', 0)):.2f}）"
                for name in positive
            )
            + "。"
        )
    else:
        parts.append("当前没有达到阳性阈值的主要结论。")
    if uncertain:
        parts.append(
            "建议重点复核："
            + "；".join(
                f"{LABEL_ZH.get(name, name)}为{STATUS_ZH['uncertain']}"
                f"（模型分数 {float(by_name[name].get('confidence', 0)):.2f}）"
                for name in uncertain[:5]
            )
            + "。"
        )
    if modality == "cxr_chest":
        limited = [name for name in LABELS if CXR_APPLICABILITY.get(name) == "limited"]
        if limited:
            parts.append(
                "胸片示意后端说明："
                + "、".join(LABEL_ZH.get(name, name) for name in limited)
                + " 在胸片上显示有限，编码器分数为中性占位，需结合报告与其他检查复核。"
            )
    if report_text.strip():
        parts.append(f"输入报告摘要已纳入融合；请对照原始{modality_zh}报告全文。")
    parts.append(f"影像证据分数已写入 JSON 的 ctclip_score 字段（{modality_zh} 编码器映射）。")
    parts.append("以上结果必须结合原始影像和报告进行人工复核。")
    parts.append(DISCLAIMER)
    return "".join(parts)
