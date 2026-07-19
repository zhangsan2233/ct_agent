import re

from chestct_agent.schemas import ParsedReport


SECTION_PATTERNS = {
    "findings": re.compile(r"(?:findings?|description)\s*:?\s*(.*?)(?=(?:impression|conclusion)\s*:|$)", re.I | re.S),
    "impression": re.compile(r"(?:impression|conclusion)\s*:?\s*(.*)$", re.I | re.S),
}


def parse_report(report_text: str) -> ParsedReport:
    text = re.sub(r"\s+", " ", report_text or "").strip()
    findings = ""
    impression = ""

    findings_match = SECTION_PATTERNS["findings"].search(text)
    impression_match = SECTION_PATTERNS["impression"].search(text)
    if findings_match:
        findings = findings_match.group(1).strip()
    if impression_match:
        impression = impression_match.group(1).strip()

    if not findings and not impression:
        findings = text

    return ParsedReport(findings=findings, impression=impression, full_report=text)

