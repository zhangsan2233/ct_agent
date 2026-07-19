from chestct_agent.schemas import LabelPrediction


def fuse_predictions(
    report_predictions: list[LabelPrediction],
    ct_predictions: list[LabelPrediction],
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
        report_conf = report.confidence if report else 0.0
        ct_conf = ct.confidence if ct else 0.0
        confidence = max(report_conf, ct_conf)
        if (report and report.status == "positive") or (ct and ct.status == "positive"):
            status = "positive"
        elif (report and report.status == "uncertain") or (ct and ct.status == "uncertain"):
            status = "uncertain"
        else:
            status = "negative"

        if report and report.status == "positive" and ct is None:
            warnings.append(f"No CT model result available for report-positive label: {label}")
        if report and ct and {report.status, ct.status} == {"positive", "negative"}:
            warnings.append(f"Report/CT disagreement requires review: {label}")

        fused.append(
            LabelPrediction(
                name=label,
                status=status,
                confidence=round(float(confidence), 4),
                source="fusion",
            )
        )

    return fused, warnings
