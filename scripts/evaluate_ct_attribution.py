from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chestct_agent.ctclip import CtClipRuntime
from chestct_agent.tools.ct_attribution import CtAttributionTool


def load_attribution(path: Path, label: str) -> tuple[np.ndarray, list[str], dict]:
    with np.load(path, allow_pickle=False) as payload:
        values = np.asarray(payload["attributions"], dtype=np.float32)
        labels = [str(value) for value in payload["labels"].tolist()]
        preprocess = json.loads(str(payload["preprocess_json"].item()))
    if label not in labels:
        raise ValueError(f"Label is absent from attribution artifact: {label}")
    attribution = values[labels.index(label)]
    valid_weights = CtAttributionTool._valid_grid_weights(
        preprocess, attribution.shape
    )
    return attribution * valid_weights, labels, preprocess


def choose_deletion_patches(
    attribution: np.ndarray, valid_mask: np.ndarray, count: int, seed: int
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    flat = attribution.reshape(-1)
    valid_flat = np.flatnonzero(valid_mask.reshape(-1))
    count = min(max(1, count), len(valid_flat) // 2)
    top_flat = valid_flat[np.argsort(flat[valid_flat])[-count:]]
    rng = np.random.default_rng(seed)
    excluded = set(int(index) for index in top_flat)
    random_pool = np.asarray(
        [index for index in valid_flat if int(index) not in excluded], dtype=np.int64
    )
    random_flat = rng.choice(random_pool, size=count, replace=False)

    def unravel(values) -> list[tuple[int, int, int]]:
        return [
            tuple(int(item) for item in np.unravel_index(int(index), attribution.shape))
            for index in values
        ]

    return unravel(top_flat), unravel(random_flat)


def mask_metrics(
    attribution: np.ndarray, preprocess: dict, mask_path: Path
) -> dict[str, float | bool | str]:
    import torch
    import torch.nn.functional as functional

    mask_image = nib.load(str(mask_path))
    mask = np.asarray(mask_image.dataobj, dtype=np.float32)
    original_shape = [int(value) for value in preprocess["original_shape"]]
    shape_aligned = list(mask.shape) == original_shape
    expected_affine = preprocess.get("original_affine")
    affine_aligned = bool(
        expected_affine is not None
        and np.allclose(mask_image.affine, np.asarray(expected_affine), atol=1e-4)
    )
    if expected_affine is not None and not (shape_aligned and affine_aligned):
        from nibabel.processing import resample_from_to

        mask_image = resample_from_to(
            mask_image,
            (tuple(original_shape), np.asarray(expected_affine)),
            order=0,
        )
        mask = np.asarray(mask_image.dataobj, dtype=np.float32)
    mask_tensor = torch.from_numpy(np.transpose(mask > 0, (2, 0, 1))).float()[
        None, None
    ]
    mask_tensor = functional.interpolate(
        mask_tensor,
        size=tuple(int(value) for value in preprocess["resampled_shape"]),
        mode="nearest",
    )
    crop_start = [int(value) for value in preprocess["crop_start"]]
    crop_shape = [int(value) for value in preprocess["crop_shape"]]
    mask_tensor = mask_tensor[
        :,
        :,
        crop_start[0] : crop_start[0] + crop_shape[0],
        crop_start[1] : crop_start[1] + crop_shape[1],
        crop_start[2] : crop_start[2] + crop_shape[2],
    ]
    pad_before = [int(value) for value in preprocess["pad_before"]]
    pad_after = [int(value) for value in preprocess["pad_after"]]
    mask_tensor = functional.pad(
        mask_tensor,
        (
            pad_before[2],
            pad_after[2],
            pad_before[1],
            pad_after[1],
            pad_before[0],
            pad_after[0],
        ),
    )
    mask_grid = functional.adaptive_avg_pool3d(
        mask_tensor, output_size=attribution.shape
    )[0, 0].numpy()
    maximum = np.unravel_index(int(np.argmax(attribution)), attribution.shape)
    total_energy = float(attribution.sum())
    inside_energy = float((attribution * mask_grid).sum())
    return {
        "pointing_game_hit": bool(mask_grid[maximum] > 0),
        "mask_energy_ratio": inside_energy / max(total_energy, 1e-8),
        "alignment_method": (
            "native_shape_and_affine"
            if shape_aligned and affine_aligned
            else "affine_resample"
            if expected_affine is not None
            else "normalized_index_resample"
        ),
        "alignment_verified": bool(shape_aligned and affine_aligned),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CT-CLIP attribution with deletion and optional masks."
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--volume")
    parser.add_argument("--mask")
    parser.add_argument("--checkpoint")
    parser.add_argument("--source-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--variant", choices=("lipro", "zeroshot"), default="zeroshot")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--patch-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output")
    args = parser.parse_args()

    attribution, _, preprocess = load_attribution(Path(args.artifact), args.label)
    result: dict[str, object] = {
        "label": args.label,
        "artifact": str(Path(args.artifact).resolve()),
        "grid_shape": list(attribution.shape),
    }
    if args.mask:
        result["mask_metrics"] = mask_metrics(
            attribution, preprocess, Path(args.mask)
        )
    if args.volume:
        if not args.checkpoint or not args.source_dir:
            parser.error("--checkpoint and --source-dir are required with --volume")
        top_indices, random_indices = choose_deletion_patches(
            attribution,
            CtAttributionTool._valid_grid_weights(
                preprocess, attribution.shape
            )
            > 0,
            args.patch_count,
            args.seed,
        )
        runtime = CtClipRuntime(
            checkpoint=Path(args.checkpoint),
            source_dir=Path(args.source_dir),
            device=args.device,
            use_fp16=args.fp16,
            variant=args.variant,
        )
        deletion = runtime.deletion_scores(
            args.volume,
            args.label,
            top_indices,
            random_indices,
            tuple(int(value) for value in attribution.shape),
        )
        deletion["passes_expected_order"] = (
            deletion["top_patch_drop"] > deletion["random_patch_drop"]
        )
        result["deletion"] = deletion
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
