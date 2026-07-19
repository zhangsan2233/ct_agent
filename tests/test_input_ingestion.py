from io import BytesIO
import zipfile

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

from chestct_agent.input_ingestion import InputIngestionError, ingest_ct_upload


def test_ingest_nifti_upload_uses_deidentified_case_directory(tmp_path):
    source = tmp_path / "source.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.int16), np.eye(4)), source)

    output = ingest_ct_upload(
        "patient-name.nii.gz",
        source.read_bytes(),
        "case/without name",
        tmp_path / "uploads",
    )

    assert output.exists()
    assert output.name == "volume.nii.gz"
    assert "case_without_name" in output.parts
    assert nib.load(output).shape == (4, 5, 6)


def test_dicom_zip_rejects_parent_directory_escape(tmp_path):
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("../escape.dcm", b"not-dicom")

    with pytest.raises(InputIngestionError, match="不安全路径"):
        ingest_ct_upload(
            "study.zip",
            payload.getvalue(),
            "safe-case",
            tmp_path / "uploads",
        )


def test_dicom_zip_converts_largest_series_to_nifti(tmp_path):
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    study_uid = "1.2.826.0.1.3680043.2.1125.20260717"
    series_uid = study_uid + ".1"
    for index in range(3):
        image = sitk.Image(8, 7, sitk.sitkInt16)
        image.SetSpacing((0.7, 0.7))
        image.SetMetaData("0008|0060", "CT")
        image.SetMetaData("0020|000d", study_uid)
        image.SetMetaData("0020|000e", series_uid)
        image.SetMetaData("0020|0013", str(index + 1))
        image.SetMetaData("0020|0032", f"0\\0\\{index}")
        image.SetMetaData("0020|0037", "1\\0\\0\\0\\1\\0")
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        writer.SetFileName(str(dicom_dir / f"slice_{index}.dcm"))
        writer.Execute(image)

    payload = BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for path in dicom_dir.glob("*.dcm"):
            archive.write(path, f"nested/{path.name}")

    output = ingest_ct_upload(
        "dicom-study.zip",
        payload.getvalue(),
        "dicom-case",
        tmp_path / "uploads",
    )

    assert output.name == "volume.nii.gz"
    assert nib.load(output).shape == (8, 7, 3)
    assert not (output.parent / "dicom_extract").exists()
