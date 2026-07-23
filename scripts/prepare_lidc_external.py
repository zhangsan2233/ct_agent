from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from urllib.parse import quote

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chestct_agent.external_validation.lidc import (
    archive_is_valid,
    convert_series_archive,
    load_annotation_index,
    select_balanced_cohort,
)


SERIES_API = (
    "https://services.cancerimagingarchive.net/nbia-api/services/v1/"
    "getSeries?Collection=LIDC-IDRI&Modality=CT&format=json"
)
IMAGE_API = (
    "https://services.cancerimagingarchive.net/nbia-api/services/v1/"
    "getImage?SeriesInstanceUID="
)
XML_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/LIDC-XML-only.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan, download, and convert a balanced LIDC-IDRI external test cohort."
    )
    parser.add_argument("--root", default="data/external/lidc_idri")
    parser.add_argument("--manifest", default="artifacts/evaluation/lidc_external_manifest.csv")
    parser.add_argument("--positive-count", type=int, default=50)
    parser.add_argument("--negative-count", type=int, default=50)
    parser.add_argument("--max-download-gb", type=float, default=25.0)
    parser.add_argument("--seed", default="chestct-agent-lidc-v1")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def download_file(url: str, destination: Path, expected_bytes: int = 0) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (
        expected_bytes <= 0 or destination.stat().st_size > expected_bytes * 0.45
    ):
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    for attempt in range(1, 4):
        try:
            with httpx.stream("GET", url, timeout=600.0, follow_redirects=True) as response:
                response.raise_for_status()
                with partial.open("wb") as output:
                    for chunk in response.iter_bytes(1024 * 1024):
                        output.write(chunk)
            partial.replace(destination)
            return destination
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(attempt * 2)
    raise RuntimeError("unreachable")


def ensure_annotations(root: Path, refresh: bool) -> Path:
    archive = root / "LIDC-XML-only.zip"
    annotations = root / "annotations_xml"
    if refresh:
        archive.unlink(missing_ok=True)
    download_file(XML_URL, archive)
    if not any(annotations.rglob("*.xml")):
        from zipfile import ZipFile

        annotations.mkdir(parents=True, exist_ok=True)
        with ZipFile(archive) as payload:
            payload.extractall(annotations)
    return annotations


def load_series_metadata(root: Path, refresh: bool) -> list[dict[str, object]]:
    path = root / "series_metadata.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    response = httpx.get(SERIES_API, timeout=90.0, follow_redirects=True)
    response.raise_for_status()
    payload = response.json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def build_rows(root: Path, cohort: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for item in cohort:
        uid = str(item["SeriesInstanceUID"])
        patient_id = str(item["PatientID"])
        case_id = patient_id.lower().replace("-", "_")
        target = root / "prepared" / case_id
        rows.append(
            {
                "case_id": case_id,
                "patient_id": patient_id,
                "series_uid": uid,
                "study_uid": str(item.get("StudyInstanceUID", "")),
                "ground_truth": str(item["ground_truth"]),
                "labels": "pulmonary_nodule" if item["ground_truth"] == "positive" else "",
                "reader_count": int(item["reader_count"]),
                "large_nodule_count": int(item["large_nodule_count"]),
                "source_bytes": int(item.get("FileSize", 0)),
                "ct_volume_path": relative(target / "ct.nii.gz"),
                "nodule_union_mask_path": relative(target / "nodule_union.nii.gz"),
                "nodule_consensus2_mask_path": relative(target / "nodule_consensus2.nii.gz"),
                "xml_path": relative(Path(str(item["xml_path"]))),
                "archive_path": relative(root / "raw" / f"{case_id}.zip"),
                "split": "external_test",
                "dataset": "LIDC-IDRI",
                "dataset_url": "https://www.cancerimagingarchive.net/collection/lidc-idri/",
                "license": "CC BY 3.0",
                "label_definition": "positive: >=3 readers marked >=3mm nodule; negative: 0 readers",
            }
        )
    return rows


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def download_row(row: dict[str, object]) -> str:
    archive = PROJECT_ROOT / str(row["archive_path"])
    if archive_is_valid(archive):
        return f"cached {row['case_id']}"
    archive.unlink(missing_ok=True)
    download_file(
        IMAGE_API + quote(str(row["series_uid"]), safe="."),
        archive,
        int(row["source_bytes"]),
    )
    if not archive_is_valid(archive):
        archive.unlink(missing_ok=True)
        raise ValueError(f"Downloaded archive failed validation: {row['case_id']}")
    return f"downloaded {row['case_id']}"


def convert_row(row: dict[str, object], keep_archive: bool) -> str:
    archive = PROJECT_ROOT / str(row["archive_path"])
    volume = PROJECT_ROOT / str(row["ct_volume_path"])
    union_mask = PROJECT_ROOT / str(row["nodule_union_mask_path"])
    consensus_mask = PROJECT_ROOT / str(row["nodule_consensus2_mask_path"])
    if volume.exists() and union_mask.exists() and consensus_mask.exists():
        return f"prepared {row['case_id']} (cached)"
    result = convert_series_archive(
        archive,
        PROJECT_ROOT / str(row["xml_path"]),
        volume,
        union_mask,
        consensus_mask,
    )
    if not keep_archive:
        archive.unlink(missing_ok=True)
    return f"prepared {row['case_id']} {result}"


def main() -> None:
    args = parse_args()
    root = (PROJECT_ROOT / args.root).resolve()
    annotations_root = ensure_annotations(root, args.refresh)
    metadata = load_series_metadata(root, args.refresh)
    annotations, duplicates = load_annotation_index(annotations_root)
    cohort = select_balanced_cohort(
        annotations,
        metadata,
        args.positive_count,
        args.negative_count,
        int(args.max_download_gb * 2**30),
        args.seed,
    )
    rows = build_rows(root, cohort)
    manifest = (PROJECT_ROOT / args.manifest).resolve()
    write_manifest(manifest, rows)
    total_gib = sum(int(row["source_bytes"]) for row in rows) / 2**30
    print(
        f"Planned {len(rows)} cases ({args.positive_count} positive, "
        f"{args.negative_count} negative), source archives={total_gib:.2f} GiB, "
        f"excluded_duplicate_xml_series={len(duplicates)}"
    )
    print(f"Manifest: {manifest}")
    if not (args.download or args.convert):
        return

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(download_row, row): row for row in rows}
        for completed, future in enumerate(as_completed(futures), start=1):
            print(f"[{completed}/{len(rows)}] {future.result()}", flush=True)
    if args.convert:
        for index, row in enumerate(rows, start=1):
            print(f"[{index}/{len(rows)}] {convert_row(row, args.keep_archives)}", flush=True)


if __name__ == "__main__":
    main()
