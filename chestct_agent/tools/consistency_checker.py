from chestct_agent.calibration import CalibrationStore
from chestct_agent.knowledge import LABEL_ZH
from chestct_agent.schemas import LabelPrediction, ReportEvidence


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
