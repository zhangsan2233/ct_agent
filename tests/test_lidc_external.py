from pathlib import Path

from chestct_agent.external_validation.lidc import (
    load_annotation_index,
    parse_annotation,
    select_balanced_cohort,
)


def write_xml(path: Path, uid: str, positive_readers: int) -> None:
    sessions = []
    for index in range(4):
        nodule = ""
        if index < positive_readers:
            nodule = """
            <unblindedReadNodule><noduleID>1</noduleID>
              <characteristics><subtlety>3</subtlety></characteristics>
              <roi><imageSOP_UID>sop</imageSOP_UID><inclusion>TRUE</inclusion>
                <edgeMap><xCoord>1</xCoord><yCoord>2</yCoord></edgeMap>
              </roi>
            </unblindedReadNodule>"""
        sessions.append(f"<readingSession>{nodule}</readingSession>")
    path.write_text(
        "<LidcReadMessage><ResponseHeader>"
        f"<SeriesInstanceUid>{uid}</SeriesInstanceUid><StudyInstanceUID>study</StudyInstanceUID>"
        "</ResponseHeader>"
        + "".join(sessions)
        + "</LidcReadMessage>",
        encoding="utf-8",
    )


def test_parse_annotation_uses_reader_consensus_policy(tmp_path: Path) -> None:
    positive = tmp_path / "positive.xml"
    ambiguous = tmp_path / "ambiguous.xml"
    negative = tmp_path / "negative.xml"
    write_xml(positive, "positive", 3)
    write_xml(ambiguous, "ambiguous", 2)
    write_xml(negative, "negative", 0)

    assert parse_annotation(positive).ground_truth == "positive"
    assert parse_annotation(ambiguous).ground_truth == "ambiguous"
    assert parse_annotation(negative).ground_truth == "negative"


def test_duplicate_xml_is_excluded_and_selection_is_balanced(tmp_path: Path) -> None:
    write_xml(tmp_path / "positive.xml", "positive", 4)
    write_xml(tmp_path / "negative.xml", "negative", 0)
    write_xml(tmp_path / "duplicate-a.xml", "duplicate", 4)
    write_xml(tmp_path / "duplicate-b.xml", "duplicate", 4)
    annotations, duplicates = load_annotation_index(tmp_path)
    assert "duplicate" in duplicates
    assert "duplicate" not in annotations

    metadata = [
        {"SeriesInstanceUID": "positive", "PatientID": "p1", "FileSize": 10},
        {"SeriesInstanceUID": "negative", "PatientID": "p2", "FileSize": 20},
        {"SeriesInstanceUID": "duplicate", "PatientID": "p3", "FileSize": 10},
    ]
    cohort = select_balanced_cohort(annotations, metadata, 1, 1, max_bytes=30)
    assert {item["ground_truth"] for item in cohort} == {"positive", "negative"}
