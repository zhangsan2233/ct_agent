from chestct_agent.calibration import CalibrationStore
from chestct_agent.knowledge import LABEL_ZH
from chestct_agent.schemas import (
    DiagnosticToolEvidence,
    LabelPrediction,
    QwenVisualLabelReview,
    ReportEvidence,
)


def fuse_predictions(
    report_predictions: list[LabelPrediction],
    ct_predictions: list[LabelPrediction],
    report_weight: float = 0.55,
    ct_weight: float = 0.45,
    positive_threshold: float = 0.5,
    uncertain_threshold: float = 0.35,
    strong_negative_threshold: float = 0.15,
    evidence_by_label: dict[str, list[ReportEvidence]] | None = None,
    calibration: CalibrationStore | None = None,
) -> tuple[list[LabelPrediction], list[str]]:
    report_by_label = {item.name: item for item in report_predictions}
    ct_by_label = {item.name: item for item in ct_predictions}
    warnings: list[str] = []
    fused: list[LabelPrediction] = []

    labels = list(report_by_label)
    labels.extend(sorted(set(ct_by_label) - set(report_by_label)))
    for label in labels:
        report = report_by_label.get(label)
        ct = ct_by_label.get(label)
        calibrated = None
        label_positive_threshold = positive_threshold
        label_uncertain_threshold = uncertain_threshold
        if report is None and ct is not None:
            confidence = ct.confidence
            status = ct.status
        elif ct is None and report is not None:
            confidence = report.confidence
            status = report.status
        else:
            assert report is not None and ct is not None
            total_weight = report_weight + ct_weight
            confidence = (
                report.confidence * report_weight + ct.confidence * ct_weight
            ) / total_weight
            if calibration is not None:
                calibrated = calibration.calibrate("fusion", label, confidence)
                confidence = calibrated.probability
                label_positive_threshold = calibrated.positive_threshold
                label_uncertain_threshold = calibrated.uncertain_threshold
            evidence = (evidence_by_label or {}).get(label, [])
            explicit_negative = any(item.polarity == "negative" for item in evidence)
            explicit_positive = any(item.polarity == "positive" for item in evidence)
            legacy_conflict_rules = evidence_by_label is None
            strong_conflict = (
                report.status == "positive"
                and ct.status == "negative"
                and ct.confidence <= strong_negative_threshold
                and (explicit_positive or legacy_conflict_rules)
            ) or (
                ct.status == "positive"
                and report.status == "negative"
                and report.confidence <= strong_negative_threshold
                and (explicit_negative or legacy_conflict_rules)
            )
            if strong_conflict:
                status = "uncertain"
            elif confidence >= label_positive_threshold:
                status = "positive"
            elif confidence >= label_uncertain_threshold:
                status = "uncertain"
            else:
                status = "negative"

        if report and ct and {report.status, ct.status} == {"positive", "negative"}:
            warnings.append(
                f"报告与 CT 结果不一致，需人工复核：{LABEL_ZH.get(label, label)}（{label}）"
            )

        fused.append(
            LabelPrediction(
                name=label,
                status=status,
                confidence=round(float(confidence), 4),
                source="fusion",
                calibrated=bool(
                    (calibrated and calibrated.calibrated)
                    or (report and report.calibrated)
                    or (ct and ct.calibrated)
                ),
                calibration_version=(
                    calibrated.version
                    if calibrated and calibrated.version
                    else report.calibration_version
                    if report and report.calibration_version
                    else ct.calibration_version if ct else None
                ),
            )
        )

    return fused, warnings


def apply_credibility_gate(
    fused_predictions: list[LabelPrediction],
    ct_predictions: list[LabelPrediction],
    evidence_by_label: dict[str, list[ReportEvidence]],
    ct_quality_degraded: bool,
) -> tuple[list[LabelPrediction], list[str]]:
    """Prevent unsupported classifier scores from becoming positive conclusions."""
    ct_by_label = {item.name: item for item in ct_predictions}
    gated: list[LabelPrediction] = []
    warnings: list[str] = []
    for prediction in fused_predictions:
        evidence = evidence_by_label.get(prediction.name, [])
        report_supported = any(item.polarity == "positive" for item in evidence)
        ct_prediction = ct_by_label.get(prediction.name)
        ct_supported = bool(
            ct_prediction
            and ct_prediction.status == "positive"
            and not ct_quality_degraded
        )
        if prediction.status == "positive" and not (report_supported or ct_supported):
            gated.append(prediction.model_copy(update={"status": "uncertain"}))
            warnings.append(
                "可信度门控：缺少阳性报告证据或可靠 CT 支持，已将"
                f"{LABEL_ZH.get(prediction.name, prediction.name)}降级为待复核候选。"
            )
        else:
            gated.append(prediction)
    return gated, warnings


def apply_qwen_visual_review(
    predictions: list[LabelPrediction],
    reviews: list[QwenVisualLabelReview],
    minimum_confidence: float = 0.85,
) -> tuple[list[LabelPrediction], list[str]]:
    """Use high-confidence slice evidence to adjudicate, never silently overwrite, CT labels."""
    review_by_label = {item.name: item for item in reviews}
    updated: list[LabelPrediction] = []
    warnings: list[str] = []
    for prediction in predictions:
        review = review_by_label.get(prediction.name)
        if review is None or review.confidence < minimum_confidence:
            updated.append(prediction)
            continue

        next_status = prediction.status
        if review.status == "negative" and prediction.status == "positive":
            next_status = "uncertain"
        elif review.status == "positive" and prediction.status == "uncertain":
            next_status = "positive"
        elif review.status == "positive" and prediction.status == "negative":
            next_status = "uncertain"

        if next_status != prediction.status:
            updated.append(prediction.model_copy(update={"status": next_status}))
            warnings.append(
                "Qwen视觉复核调整："
                f"{LABEL_ZH.get(prediction.name, prediction.name)}由{prediction.status}调整为"
                f"{next_status}；视觉判断={review.status}，置信度={review.confidence:.2f}。"
            )
        else:
            updated.append(prediction)
    return updated, warnings


def apply_diagnostic_tool_evidence(
    predictions: list[LabelPrediction],
    evidence: list[DiagnosticToolEvidence],
    visual_reviews: list[QwenVisualLabelReview],
    visual_minimum_confidence: float = 0.85,
) -> tuple[list[LabelPrediction], list[str]]:
    """Adjudicate selected labels using independent segmentation or measurement evidence."""
    by_label: dict[str, list[DiagnosticToolEvidence]] = {}
    for item in evidence:
        if item.verdict != "unavailable" and item.coverage != "unavailable":
            by_label.setdefault(item.label, []).append(item)
    visual_by_label = {item.name: item for item in visual_reviews}
    updated: list[LabelPrediction] = []
    warnings: list[str] = []

    for prediction in predictions:
        tools = by_label.get(prediction.name, [])
        if not tools:
            updated.append(prediction)
            continue
        strong_positive = [
            item for item in tools if item.verdict == "positive" and item.confidence >= 0.78
        ]
        strong_negative = [
            item
            for item in tools
            if item.verdict == "negative"
            and item.confidence >= 0.85
            and item.coverage == "complete"
        ]
        uncertain = any(item.verdict == "uncertain" for item in tools)
        visual = visual_by_label.get(prediction.name)
        visual_positive = bool(
            visual
            and visual.status == "positive"
            and visual.confidence >= visual_minimum_confidence
        )
        visual_negative = bool(
            visual
            and visual.status == "negative"
            and visual.confidence >= visual_minimum_confidence
        )

        next_status = prediction.status
        next_confidence = prediction.confidence
        if prediction.name == "pulmonary_nodule":
            # The nodule segmenter is a candidate detector. A positive diagnosis requires an
            # independent visual confirmation; a complete double-negative can rule a candidate out.
            if strong_positive and visual_positive:
                next_status = "positive"
                next_confidence = max(next_confidence, min(strong_positive[0].confidence, visual.confidence))
            elif prediction.status == "positive" and visual_positive:
                next_status = "positive"
                next_confidence = max(next_confidence, visual.confidence)
            elif strong_negative and visual_negative:
                next_status = "negative"
                next_confidence = min(next_confidence, 1.0 - max(strong_negative[0].confidence, visual.confidence))
            elif strong_positive and prediction.status == "negative":
                next_status = "uncertain"
                next_confidence = max(next_confidence, 0.5)
            elif strong_negative and prediction.status == "positive":
                next_status = "uncertain"
                next_confidence = min(next_confidence, 0.5)
            elif uncertain and prediction.status == "positive":
                next_status = "uncertain"
                next_confidence = 0.5
        elif prediction.name == "pericardial_effusion":
            # Small pericardial masks are sensitive to protocol and segmentation error. They
            # may trigger review, but cannot independently confirm or exclude an effusion.
            if strong_positive and visual_positive:
                next_status = "positive"
                next_confidence = max(
                    next_confidence,
                    min(strong_positive[0].confidence, visual.confidence),
                )
            elif strong_positive or uncertain:
                next_status = "uncertain"
                next_confidence = 0.5
        else:
            # Volume and HU measurements are independent quantitative evidence. They can promote
            # an uncertain finding, but a conflict with the screening model remains auditable.
            if strong_positive and visual_positive:
                next_status = "positive"
                next_confidence = max(
                    next_confidence,
                    min(strong_positive[0].confidence, visual.confidence),
                )
            elif strong_negative and visual_negative:
                next_status = "negative"
                next_confidence = min(
                    next_confidence,
                    1.0 - max(strong_negative[0].confidence, visual.confidence),
                )
            elif strong_positive:
                next_status = "positive" if prediction.status != "negative" else "uncertain"
                next_confidence = max(next_confidence, strong_positive[0].confidence)
            elif strong_negative:
                next_status = "negative" if prediction.status != "positive" else "uncertain"
                next_confidence = min(next_confidence, 1.0 - strong_negative[0].confidence)
            elif uncertain and prediction.status in {"positive", "negative"}:
                next_status = "uncertain"
                next_confidence = 0.5

        if next_status != prediction.status:
            tools_used = ", ".join(sorted({item.tool for item in tools}))
            warnings.append(
                f"独立影像工具改判：{LABEL_ZH.get(prediction.name, prediction.name)}由"
                f"{prediction.status}调整为{next_status}；依据={tools_used}。"
            )
        updated.append(
            prediction.model_copy(
                update={"status": next_status, "confidence": round(float(next_confidence), 4)}
            )
        )
    return updated, warnings
