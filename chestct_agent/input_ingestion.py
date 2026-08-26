from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
import shutil
import zipfile


MAX_CT_UPLOAD_BYTES = 4 * 1024**3
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 12 * 1024**3
MAX_ARCHIVE_FILES = 20_000
MAX_REPORT_BYTES = 2 * 1024**2


class InputIngestionError(ValueError):
    pass


def safe_case_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())[:80].strip("_")
    return cleaned or "uploaded_case"


def decode_report_bytes(data: bytes) -> str:
    if len(data) > MAX_REPORT_BYTES:
        raise InputIngestionError("报告文件超过 2 MB。")
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    raise InputIngestionError("报告文件编码无法识别，请上传 UTF-8 或 GB18030 文本。")


def ingest_ct_upload(filename: str, data: bytes, case_id: str, upload_root: Path) -> Path:
    if not data:
        raise InputIngestionError("上传的 CT 文件为空。")
    if len(data) > MAX_CT_UPLOAD_BYTES:
        raise InputIngestionError("CT 上传文件超过 4 GB。")

    lower_name = filename.lower()
    upload_id = sha256(data).hexdigest()[:16]
    case_dir = Path(upload_root) / safe_case_id(case_id) / upload_id
    case_dir.mkdir(parents=True, exist_ok=True)

    if lower_name.endswith((".nii", ".nii.gz")):
        import nibabel as nib

        suffix = ".nii.gz" if lower_name.endswith(".nii.gz") else ".nii"
        output = case_dir / f"volume{suffix}"
        if not output.exists():
            output.write_bytes(data)
        try:
            image = nib.load(str(output))
            if len(image.shape) != 3 or min(image.shape) < 2:
                raise InputIngestionError("CT NIfTI 必须是有效的三维体数据。")
        except InputIngestionError:
            raise
        except Exception as exc:
            output.unlink(missing_ok=True)
            raise InputIngestionError("无法读取上传的 NIfTI 文件。") from exc
        return output.resolve()

    if lower_name.endswith(".zip"):
        return _convert_dicom_zip(data, case_dir)

    raise InputIngestionError("CT 仅支持 .nii、.nii.gz 或包含 DICOM 序列的 .zip。")


MAX_CXR_UPLOAD_BYTES = 40 * 1024**2


def ingest_cxr_upload(filename: str, data: bytes, case_id: str, upload_root: Path) -> Path:
    """Store a 2D chest radiograph (PNG/JPEG) under a deidentified case directory."""
    if not data:
        raise InputIngestionError("上传的 X 光文件为空。")
    if len(data) > MAX_CXR_UPLOAD_BYTES:
        raise InputIngestionError("X 光上传文件超过 40 MB。")
    lower_name = filename.lower()
    if not lower_name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
        raise InputIngestionError("胸部 X 光示意接口仅支持 .png、.jpg、.jpeg、.webp 或 .bmp。")
    upload_id = sha256(data).hexdigest()[:16]
    case_dir = Path(upload_root) / safe_case_id(case_id) / upload_id
    case_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(lower_name).suffix
    output = case_dir / f"cxr{suffix}"
    output.write_bytes(data)
    try:
        from PIL import Image

        image = Image.open(output)
        image.load()
        if min(image.size) < 32:
            raise InputIngestionError("X 光图像尺寸过小。")
    except InputIngestionError:
        output.unlink(missing_ok=True)
        raise
    except Exception as exc:
        output.unlink(missing_ok=True)
        raise InputIngestionError("无法读取上传的 X 光图像。") from exc
    return output.resolve()


def _convert_dicom_zip(data: bytes, case_dir: Path) -> Path:
    dicom_dir = (case_dir / "dicom_extract").resolve()
    output = (case_dir / "volume.nii.gz").resolve()
    if output.exists():
        return output
    dicom_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if not members or len(members) > MAX_ARCHIVE_FILES:
                raise InputIngestionError("DICOM ZIP 文件数量为空或超过 20000。")
            if sum(item.file_size for item in members) > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                raise InputIngestionError("DICOM ZIP 解压后超过 12 GB。")
            for item in members:
                relative = Path(item.filename.replace("\\", "/"))
                if relative.is_absolute() or ".." in relative.parts:
                    raise InputIngestionError("DICOM ZIP 包含不安全路径。")
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise InputIngestionError("DICOM ZIP 不允许包含符号链接。")
                target = (dicom_dir / relative).resolve()
                if dicom_dir not in target.parents:
                    raise InputIngestionError("DICOM ZIP 包含越界路径。")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)

        try:
            import SimpleITK as sitk
        except ImportError as exc:
            raise InputIngestionError("服务端未安装 DICOM 转换组件 SimpleITK。") from exc

        candidates: list[tuple[int, list[str]]] = []
        directories = {dicom_dir, *(path.parent for path in dicom_dir.rglob("*") if path.is_file())}
        for directory in directories:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory)) or []
            for series_id in series_ids:
                filenames = list(
                    sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), series_id)
                )
                if filenames:
                    candidates.append((len(filenames), filenames))
        if not candidates:
            raise InputIngestionError("ZIP 中没有识别到可读取的 DICOM 序列。")

        _, filenames = max(candidates, key=lambda item: item[0])
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(filenames)
        image = reader.Execute()
        if image.GetDimension() != 3 or min(image.GetSize()) < 2:
            raise InputIngestionError("DICOM 主序列不是有效的三维 CT。")
        sitk.WriteImage(image, str(output), True)
        return output
    except zipfile.BadZipFile as exc:
        raise InputIngestionError("上传文件不是有效的 ZIP。") from exc
    finally:
        if dicom_dir.exists() and case_dir.resolve() in dicom_dir.parents:
            shutil.rmtree(dicom_dir, ignore_errors=True)
