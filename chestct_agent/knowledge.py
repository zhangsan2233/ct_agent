from chestct_agent.labels import LABEL_SPECS, LABEL_ZH as LABEL_ZH


LABEL_KNOWLEDGE = {
    spec.id: {
        "title": spec.title,
        "terms": list(spec.terms),
        "zh": spec.definition,
        "imaging": spec.imaging,
        "anatomy_regions": list(spec.anatomy_regions),
    }
    for spec in LABEL_SPECS
}

STATUS_ZH = {
    "positive": "阳性",
    "negative": "阴性",
    "uncertain": "不确定",
}

INPUT_MODE_ZH = {
    "report_only": "仅报告",
    "ct_only": "仅 CT",
    "report_and_ct": "报告 + CT",
}

TOOL_TRACE_ZH = {
    "parse_input": "解析输入",
    "task_planner": "规划工具调用",
    "report_parser_tool": "解析报告",
    "text_classifier_tool": "报告分类",
    "report_graph_tool": "构建RadGraph-XL报告图谱",
    "ct_preprocess_tool": "CT 预处理",
    "ct_classifier_tool": "CT 分类",
    "organ_segmentation_tool": "器官分割",
    "lesion_grounding_tool": "病灶定位",
    "visual_evidence_tool": "生成图像证据",
    "agentic_rag_query_planner": "规划检索问题",
    "medical_rag_tool": "检索医学知识",
    "retrieval_grader": "评估检索结果",
    "query_rewriter": "改写检索问题",
    "similar_case_retriever_tool": "检索相似病例",
    "evidence_extractor_tool": "提取报告证据",
    "consistency_checker_tool": "检查多模态一致性",
    "human_approval_gate": "人工审批",
    "structured_output_generator": "生成结构化结果",
    "json_validator_tool": "校验 JSON",
    "explanation_generator": "生成中文结论",
    "human_correction_tool": "应用医生纠错",
    "dataset_oracle_tool": "应用开发沙箱弱标签反馈",
}

NEGATION_TERMS = (
    "no",
    "without",
    "absent",
    "negative for",
    "free of",
    "not seen",
    "not observed",
    "not detected",
    "no evidence of",
    "neither",
)

UNCERTAIN_TERMS = (
    "possible",
    "possibly",
    "probable",
    "probably",
    "may represent",
    "cannot exclude",
    "could represent",
    "suspicious",
    "questionable",
    "suggestive of",
)

HISTORY_TERMS = (
    "history of",
    "previous",
    "previously",
    "status post",
    "treated",
    "resolved",
)
