from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET
from zipfile import BadZipFile, ZipFile

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class AnnotationRecord:
    series_uid: str
    study_uid: str
    xml_path: Path
    reader_count: int
    large_nodule_count: int

    @property
    def ground_truth(self) -> str:
        if self.reader_count >= 3:
            return "positive"
        if self.reader_count == 0:
            return "negative"
        return "ambiguous"


def _local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _first_text(root: ET.Element, name: str) -> str:
    return next(
        (
            (element.text or "").strip()
            for element in root.iter()
            if _local_name(element) == name
        ),
        "",
    )


def _large_nodules(session: ET.Element) -> list[ET.Element]:
    return [
        nodule
        for nodule in session
        if _local_name(nodule) == "unblindedReadNodule"
        and any(_local_name(child) == "characteristics" for child in nodule)
    ]


def parse_annotation(path: Path) -> AnnotationRecord:
    root = ET.parse(path).getroot()
    sessions = [element for element in root if _local_name(element) == "readingSession"]
    large_by_reader = [_large_nodules(session) for session in sessions]
    return AnnotationRecord(
        series_uid=_first_text(root, "SeriesInstanceUid"),
        study_uid=_first_text(root, "StudyInstanceUID"),
        xml_path=path,
        reader_count=sum(bool(nodules) for nodules in large_by_reader),
        large_nodule_count=sum(len(nodules) for nodules in large_by_reader),
    )


def load_annotation_index(root: Path) -> tuple[dict[str, AnnotationRecord], set[str]]:
    """Load unique series and explicitly exclude corrected/duplicate XML records."""
    grouped: dict[str, list[AnnotationRecord]] = {}
    for path in sorted(root.rglob("*.xml")):
        try:
            record = parse_annotation(path)
        except ET.ParseError:
            continue
        if record.series_uid:
            grouped.setdefault(record.series_uid, []).append(record)
    duplicates = {uid for uid, records in grouped.items() if len(records) > 1}
    unique = {uid: records[0] for uid, records in grouped.items() if len(records) == 1}
    return unique, duplicates


def _stable_key(series_uid: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{series_uid}".encode("utf-8")).hexdigest()


def select_balanced_cohort(
    annotations: dict[str, AnnotationRecord],
    series_metadata: list[dict[str, object]],
    positive_count: int,
    negative_count: int,
    max_bytes: int,
    seed: str = "chestct-agent-lidc-v1",
) -> list[dict[str, object]]:
    metadata_by_uid = {
        str(item["SeriesInstanceUID"]): item
        for item in series_metadata
        if item.get("SeriesInstanceUID")
    }
    candidates: list[dict[str, object]] = []
    for uid, annotation in annotations.items():
        metadata = metadata_by_uid.get(uid)
        if metadata is None or annotation.ground_truth == "ambiguous":
            continue
        candidates.append(
            {
                **metadata,
                "reader_count": annotation.reader_count,
                "large_nodule_count": annotation.large_nodule_count,
                "ground_truth": annotation.ground_truth,
                "xml_path": str(annotation.xml_path.resolve()),
            }
        )

    selected: list[dict[str, object]] = []
    used_patients: set[str] = set()
    for truth, count in (("positive", positive_count), ("negative", negative_count)):
        pool = sorted(
            (item for item in candidates if item["ground_truth"] == truth),
            key=lambda item: _stable_key(str(item["SeriesInstanceUID"]), seed),
        )
        chosen = []
        for item in pool:
            patient_id = str(item.get("PatientID", ""))
            if patient_id in used_patients:
                continue
            chosen.append(item)
            used_patients.add(patient_id)
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError(f"Only {len(chosen)} unique-patient {truth} cases are available.")
        selected.extend(chosen)

    total_bytes = sum(int(item.get("FileSize", 0)) for item in selected)
    if total_bytes > max_bytes:
        raise ValueError(
            f"Selected cohort is {total_bytes / 2**30:.2f} GiB, above the "
            f"{max_bytes / 2**30:.2f} GiB limit."
        )
    return selected


def archive_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with ZipFile(path) as archive:
            return archive.testzip() is None and any(
                item.filename.lower().endswith(".dcm") for item in archive.infolist()
            )
    except BadZipFile:
        return False


def _annotation_reader_polygons(
    xml_path: Path,
) -> list[dict[str, list[list[tuple[int, int]]]]]:
    root = ET.parse(xml_path).getroot()
    readers: list[dict[str, list[list[tuple[int, int]]]]] = []
    for session in [element for element in root if _local_name(element) == "readingSession"]:
        polygons: dict[str, list[list[tuple[int, int]]]] = {}
        for nodule in _large_nodules(session):
            for roi in [child for child in nodule if _local_name(child) == "roi"]:
                inclusion = _first_text(roi, "inclusion").upper()
                if inclusion == "FALSE":
                    continue
                sop_uid = _first_text(roi, "imageSOP_UID")
                points: list[tuple[int, int]] = []
                for edge in [child for child in roi if _local_name(child) == "edgeMap"]:
                    x = _first_text(edge, "xCoord")
                    y = _first_text(edge, "yCoord")
                    if x and y:
                        points.append((int(float(x)), int(float(y))))
                if sop_uid and points:
                    polygons.setdefault(sop_uid, []).append(points)
        readers.append(polygons)
    return readers


def _sop_uid(path: str) -> str:
    import SimpleITK as sitk

    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    return reader.GetMetaData("0008|0018").strip() if reader.HasMetaDataKey("0008|0018") else ""


def convert_series_archive(
    archive_path: Path,
    xml_path: Path,
    volume_path: Path,
    union_mask_path: Path,
    consensus_mask_path: Path,
) -> dict[str, object]:
    """Convert one TCIA DICOM archive and rasterize expert nodule contours."""
    import SimpleITK as sitk

    if not archive_is_valid(archive_path):
        raise ValueError(f"Invalid TCIA series archive: {archive_path}")
    volume_path.parent.mkdir(parents=True, exist_ok=True)
    union_mask_path.parent.mkdir(parents=True, exist_ok=True)
    consensus_mask_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="lidc_dicom_") as temporary:
        temporary_path = Path(temporary)
        with ZipFile(archive_path) as archive:
            for item in archive.infolist():
                if item.filename.lower().endswith(".dcm"):
                    target = temporary_path / Path(item.filename).name
                    with archive.open(item) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination)

        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(temporary_path)) or []
        if len(series_ids) != 1:
            raise ValueError(f"Expected one DICOM series, found {len(series_ids)}")
        files = list(
            sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
                str(temporary_path), series_ids[0]
            )
        )
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        image = reader.Execute()
        sitk.WriteImage(image, str(volume_path), True)

        size_x, size_y, size_z = image.GetSize()
        sop_to_z = {_sop_uid(path): index for index, path in enumerate(files)}
        reader_masks = []
        for polygons_by_sop in _annotation_reader_polygons(xml_path):
            mask = np.zeros((size_z, size_y, size_x), dtype=np.uint8)
            for sop_uid, polygons in polygons_by_sop.items():
                z_index = sop_to_z.get(sop_uid)
                if z_index is None:
                    continue
                canvas = Image.new("L", (size_x, size_y), 0)
                draw = ImageDraw.Draw(canvas)
                for polygon in polygons:
                    if len(polygon) >= 3:
                        draw.polygon(polygon, fill=1)
                    else:
                        x, y = polygon[0]
                        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=1)
                mask[z_index] = np.maximum(mask[z_index], np.asarray(canvas, dtype=np.uint8))
            reader_masks.append(mask)

        agreement = (
            np.stack(reader_masks, axis=0).sum(axis=0)
            if reader_masks
            else np.zeros((size_z, size_y, size_x), dtype=np.uint8)
        )
        union = (agreement >= 1).astype(np.uint8)
        consensus = (agreement >= 2).astype(np.uint8)
        for values, path in ((union, union_mask_path), (consensus, consensus_mask_path)):
            mask_image = sitk.GetImageFromArray(values)
            mask_image.CopyInformation(image)
            sitk.WriteImage(mask_image, str(path), True)

    return {
        "shape_zyx": [int(value) for value in union.shape],
        "annotated_slices": int(np.any(union, axis=(1, 2)).sum()),
        "union_voxels": int(union.sum()),
        "consensus2_voxels": int(consensus.sum()),
        "dicom_slices": len(files),
    }
