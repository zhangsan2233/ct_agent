import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index and verify RadGenome mask alignment.")
    parser.add_argument("--mask-root", default="data/radgenome")
    parser.add_argument("--volumes-root", default="data/dataset/valid_fixed")
    parser.add_argument("--out", default="artifacts/radgenome/mask_index.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_root = Path(args.mask_root)
    volume_by_case = {
        path.name.removesuffix(".nii.gz"): path
        for path in Path(args.volumes_root).rglob("*.nii.gz")
    }
    rows: list[dict[str, object]] = []
    invalid_masks: list[str] = []
    for path in sorted(mask_root.rglob("*.nii.gz")):
        if "seg_" not in path.parent.name:
            continue
        case_id = path.parent.name.removeprefix("seg_")
        ct_path = volume_by_case.get(case_id)
        try:
            if path.stat().st_size == 0:
                raise ValueError("zero-byte file")
            mask_image = nib.load(str(path))
        except Exception as exc:
            invalid_masks.append(f"{path.as_posix()}: {type(exc).__name__}")
            continue
        aligned = False
        ct_shape = ""
        if ct_path:
            ct_image = nib.load(str(ct_path))
            ct_shape = "x".join(str(value) for value in ct_image.shape)
            aligned = mask_image.shape == ct_image.shape and bool(
                np.allclose(mask_image.affine, ct_image.affine, atol=1e-3)
            )
        mask_type = "anatomy" if "anatomy" in str(path).lower() else "region"
        rows.append(
            {
                "case_id": case_id,
                "mask_type": mask_type,
                "anatomy_name": path.name.removesuffix(".nii.gz"),
                "mask_path": str(path.as_posix()),
                "mask_shape": "x".join(str(value) for value in mask_image.shape),
                "ct_path": str(ct_path.as_posix()) if ct_path else "",
                "ct_shape": ct_shape,
                "aligned": aligned,
            }
        )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    matched_cases = int(frame.loc[frame["ct_path"].ne(""), "case_id"].nunique()) if not frame.empty else 0
    aligned_rate = float(frame["aligned"].mean()) if not frame.empty else 0.0
    print(f"Wrote {len(frame)} masks for {matched_cases} cases; aligned_rate={aligned_rate:.3f}")
    if invalid_masks:
        print(f"Skipped {len(invalid_masks)} invalid masks:")
        for item in invalid_masks[:20]:
            print(f"  {item}")


if __name__ == "__main__":
    main()
