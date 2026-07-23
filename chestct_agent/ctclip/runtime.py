from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time
import types
from typing import Any

import nibabel as nib
import numpy as np


PATHOLOGIES = {
    "Medical material": "medical_material",
    "Arterial wall calcification": "arterial_wall_calcification",
    "Cardiomegaly": "cardiomegaly",
    "Pericardial effusion": "pericardial_effusion",
    "Coronary artery wall calcification": "coronary_artery_wall_calcification",
    "Hiatal hernia": "hiatal_hernia",
    "Lymphadenopathy": "lymphadenopathy",
    "Emphysema": "emphysema",
    "Atelectasis": "atelectasis",
    "Lung nodule": "pulmonary_nodule",
    "Lung opacity": "lung_opacity",
    "Pulmonary fibrotic sequela": "pulmonary_fibrotic_sequela",
    "Pleural effusion": "pleural_effusion",
    "Mosaic attenuation pattern": "mosaic_attenuation_pattern",
    "Peribronchial thickening": "peribronchial_thickening",
    "Consolidation": "consolidation",
    "Bronchiectasis": "bronchiectasis",
    "Interlobular septal thickening": "interlobular_septal_thickening",
}

ATTRIBUTION_METHOD = "gradient_x_token"
ATTRIBUTION_VERSION = 1


class CtClipUnavailable(RuntimeError):
    pass


class CtClipRuntime:
    """Lazy, reusable wrapper around the official 3D CT-CLIP implementation."""

    def __init__(
        self,
        checkpoint: Path,
        source_dir: Path,
        device: str = "auto",
        use_fp16: bool = True,
        variant: str = "lipro",
    ):
        self.checkpoint = Path(checkpoint)
        self.source_dir = Path(source_dir)
        self.requested_device = device
        self.use_fp16 = use_fp16
        self.variant = variant.lower()
        self.model = None
        self.tokenizer = None
        self.device = None

    def readiness_error(self) -> str | None:
        asset_error = self.asset_error()
        if asset_error:
            return asset_error
        if importlib.util.find_spec("torch") is None:
            return "CT-CLIP dependency missing: torch is not installed."

        return None

    def asset_error(self) -> str | None:
        if self.variant not in {"lipro", "zeroshot"}:
            return f"Unsupported CT-CLIP variant: {self.variant}"
        if not self.checkpoint.exists():
            return f"CT-CLIP checkpoint not found: {self.checkpoint}"
        required = [
            self.source_dir / "CT_CLIP" / "ct_clip" / "ct_clip.py",
            self.source_dir / "transformer_maskgit" / "transformer_maskgit" / "ctvit.py",
        ]
        if not all(path.exists() for path in required):
            return f"Official CT-CLIP source not found or incomplete: {self.source_dir}"
        return None

    def _load(self) -> None:
        if self.model is not None:
            return
        error = self.readiness_error()
        if error:
            raise CtClipUnavailable(error)

        import torch
        from transformers import BertModel, BertTokenizer

        for package_root in (
            self.source_dir / "CT_CLIP",
            self.source_dir / "transformer_maskgit",
        ):
            package_root_str = str(package_root.resolve())
            if package_root_str not in sys.path:
                sys.path.insert(0, package_root_str)

        from ct_clip import CTCLIP

        # The official package __init__ imports training-only modules. Register a
        # lightweight package namespace so inference only loads CTViT and attention.
        transformer_root = self.source_dir / "transformer_maskgit" / "transformer_maskgit"
        transformer_package = types.ModuleType("transformer_maskgit")
        transformer_package.__path__ = [str(transformer_root.resolve())]
        sys.modules["transformer_maskgit"] = transformer_package
        from transformer_maskgit.ctvit import CTViT

        if self.requested_device == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_name = self.requested_device
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise CtClipUnavailable("CT-CLIP requested CUDA, but PyTorch cannot access the GPU.")
        self.device = torch.device(device_name)

        self.tokenizer = BertTokenizer.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True
        )
        text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
        text_encoder.resize_token_embeddings(len(self.tokenizer))
        image_encoder = CTViT(
            dim=512,
            codebook_size=8192,
            image_size=480,
            patch_size=20,
            temporal_patch_size=10,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8,
        )
        clip_model = CTCLIP(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            dim_image=294912,
            dim_text=768,
            dim_latent=512,
            extra_latent_projection=False,
            use_mlm=False,
            downsample_image_embeds=False,
            use_all_token_embeds=False,
        )
        try:
            state_dict = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(self.checkpoint, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        if self.variant == "lipro":
            class ImageLatentsClassifier(torch.nn.Module):
                def __init__(self, trained_model):
                    super().__init__()
                    self.trained_model = trained_model
                    self.dropout = torch.nn.Dropout(0.3)
                    self.relu = torch.nn.ReLU()
                    self.classifier = torch.nn.Linear(512, len(PATHOLOGIES))

                def forward(self, text, image, device):
                    _, image_latents, _ = self.trained_model(
                        text,
                        image,
                        device=device,
                        return_latents=True,
                    )
                    return self.classifier(self.dropout(self.relu(image_latents)))

            model = ImageLatentsClassifier(clip_model)
            state_dict.pop(
                "trained_model.text_transformer.embeddings.position_ids", None
            )
        else:
            model = clip_model
            state_dict.pop("text_transformer.embeddings.position_ids", None)
        model.load_state_dict(state_dict)
        model.eval().to(self.device)
        self.model = model

    @staticmethod
    def _center_crop_pad(tensor, target_shape: tuple[int, int, int]):
        tensor, _ = CtClipRuntime._center_crop_pad_with_metadata(tensor, target_shape)
        return tensor

    @staticmethod
    def _center_crop_pad_with_metadata(tensor, target_shape: tuple[int, int, int]):
        import torch.nn.functional as functional

        slices = []
        crop_start: list[int] = []
        crop_shape: list[int] = []
        for current, target in zip(tensor.shape[-3:], target_shape, strict=True):
            start = max((current - target) // 2, 0)
            end = min(start + target, current)
            slices.append(slice(start, end))
            crop_start.append(start)
            crop_shape.append(end - start)
        tensor = tensor[(..., *slices)]

        pads: list[int] = []
        pad_before: list[int] = []
        pad_after: list[int] = []
        current_shape = tensor.shape[-3:]
        for current, target in reversed(list(zip(current_shape, target_shape, strict=True))):
            total = max(target - current, 0)
            before = total // 2
            after = total - before
            pads.extend([before, after])
            pad_before.insert(0, before)
            pad_after.insert(0, after)
        metadata = {
            "target_shape": list(target_shape),
            "crop_start": crop_start,
            "crop_shape": crop_shape,
            "pad_before": pad_before,
            "pad_after": pad_after,
        }
        return functional.pad(tensor, pads, value=-1.0), metadata

    def _preprocess(self, volume_path: str):
        tensor, _ = self._preprocess_with_metadata(volume_path)
        return tensor

    def _preprocess_with_metadata(self, volume_path: str):
        import torch
        import torch.nn.functional as functional

        image = nib.load(volume_path)
        volume = image.get_fdata(dtype=np.float32)
        original_shape = list(volume.shape)
        original_spacing = [float(value) for value in image.header.get_zooms()[:3]]
        volume = np.clip(volume, -1000.0, 1000.0)
        volume = np.transpose(volume, (2, 0, 1))
        spacing_x, spacing_y, spacing_z = image.header.get_zooms()[:3]
        new_shape = (
            max(1, round(volume.shape[0] * spacing_z / 1.5)),
            max(1, round(volume.shape[1] * spacing_x / 0.75)),
            max(1, round(volume.shape[2] * spacing_y / 0.75)),
        )
        tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
        tensor = functional.interpolate(
            tensor, size=new_shape, mode="trilinear", align_corners=False
        )
        tensor = tensor / 1000.0
        tensor, crop_metadata = self._center_crop_pad_with_metadata(
            tensor, (240, 480, 480)
        )
        metadata = {
            "axis_order": "zxy",
            "original_shape": original_shape,
            "original_spacing": original_spacing,
            "original_affine": np.asarray(image.affine, dtype=float).tolist(),
            "transposed_shape": list(volume.shape),
            "resampled_shape": list(new_shape),
            **crop_metadata,
        }
        return tensor, metadata

    def _text_tokens(self):
        if self.variant == "lipro":
            return self.tokenizer(
                "",
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=200,
            ).to(self.device)

        prompts: list[str] = []
        for pathology in PATHOLOGIES:
            prompts.extend([f"{pathology} is present.", f"{pathology} is not present."])
        return self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        ).to(self.device)

    @staticmethod
    def _gradient_x_token(
        image_tokens,
        unnormalized_image_latent,
        normalized_image_latent,
        visual_projection_weight,
        target_gradients,
    ):
        """Analytically computes positive Gradient x Token for every target."""
        import torch

        tokens = image_tokens[0].float()
        z = unnormalized_image_latent[0].float()
        u = normalized_image_latent[0].float()
        target_gradients = target_gradients.float()
        projection_weight = visual_projection_weight.float()

        norm = z.norm().clamp_min(1e-8)
        gradient_z = (
            target_gradients
            - u.unsqueeze(0)
            * (target_gradients * u.unsqueeze(0)).sum(dim=-1, keepdim=True)
        ) / norm
        gradient_flat = gradient_z @ projection_weight
        temporal_tokens, height_tokens, width_tokens, embedding_dim = tokens.shape
        gradient_grid = gradient_flat.reshape(
            target_gradients.shape[0], height_tokens, width_tokens, embedding_dim
        ) / float(temporal_tokens)
        attribution = torch.einsum("thwd,nhwd->nthw", tokens, gradient_grid)
        attribution = attribution.float().clamp_min(0.0)

        flattened = attribution.flatten(start_dim=1)
        percentiles = torch.quantile(flattened, 0.99, dim=1).clamp_min(1e-8)
        attribution = (attribution / percentiles[:, None, None, None]).clamp(0.0, 1.0)
        return attribution

    def _infer(self, volume_path: str, include_attribution: bool) -> dict[str, Any]:
        self._load()
        import torch

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        text_tokens = self._text_tokens()
        volume, preprocess_metadata = self._preprocess_with_metadata(volume_path)
        volume = volume.to(self.device)
        clip_model = self.model.trained_model if self.variant == "lipro" else self.model

        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_fp16 and self.device.type == "cuda",
        ):
            text_latents, image_latents, image_tokens = clip_model(
                text_tokens,
                volume,
                device=self.device,
                return_latents=True,
            )
            if self.variant == "lipro":
                logits = self.model.classifier(self.model.relu(image_latents)).reshape(-1)
                probabilities_tensor = logits.sigmoid()
            else:
                raw_logits = (
                    torch.einsum("pd,bd->p", text_latents, image_latents)
                    * clip_model.temperature.exp()
                )
                paired_logits = raw_logits.reshape(len(PATHOLOGIES), 2)
                probabilities_tensor = paired_logits.softmax(dim=1)[:, 0]

        probabilities = {
            internal_name: float(probability)
            for internal_name, probability in zip(
                PATHOLOGIES.values(),
                probabilities_tensor.detach().float().cpu(),
                strict=True,
            )
        }
        result: dict[str, Any] = {
            "probabilities": probabilities,
            "preprocess": preprocess_metadata,
        }
        if not include_attribution:
            return result

        attribution_started = time.perf_counter()
        try:
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_fp16 and self.device.type == "cuda",
            ):
                pooled_tokens = image_tokens.mean(dim=1).reshape(image_tokens.shape[0], -1)
                visual_projection = clip_model.to_visual_latent
                if not isinstance(visual_projection, torch.nn.Linear):
                    raise CtClipUnavailable(
                        "CT-CLIP visual projection is not a linear layer; attribution is unavailable."
                    )
                unnormalized_image_latent = visual_projection(pooled_tokens)
                if self.variant == "lipro":
                    relu_derivative = (image_latents > 0).to(image_latents.dtype)
                    target_gradients = self.model.classifier.weight * relu_derivative[0]
                else:
                    prompt_gradients = text_latents.reshape(len(PATHOLOGIES), 2, -1)
                    target_gradients = (
                        prompt_gradients[:, 0] - prompt_gradients[:, 1]
                    ) * clip_model.temperature.exp()
                attribution = self._gradient_x_token(
                    image_tokens,
                    unnormalized_image_latent,
                    image_latents,
                    visual_projection.weight,
                    target_gradients,
                )
            attribution_array = attribution.detach().float().cpu().numpy().astype(np.float16)
            if not np.isfinite(attribution_array).all():
                raise ValueError("CT-CLIP attribution contains non-finite values.")
            result["attributions"] = attribution_array
            result["method"] = ATTRIBUTION_METHOD
            result["grid_shape"] = list(attribution_array.shape[1:])
        except (RuntimeError, ValueError, CtClipUnavailable) as exc:
            result["attribution_error"] = f"{type(exc).__name__}: {exc}"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        result["attribution_latency_ms"] = round(
            (time.perf_counter() - attribution_started) * 1000, 2
        )
        if self.device.type == "cuda":
            result["peak_gpu_memory_mb"] = round(
                torch.cuda.max_memory_allocated(self.device) / (1024**2), 2
            )
        return result

    def predict(self, volume_path: str) -> dict[str, float]:
        return self._infer(volume_path, include_attribution=False)["probabilities"]

    def predict_with_attribution(self, volume_path: str) -> dict[str, Any]:
        return self._infer(volume_path, include_attribution=True)

    def deletion_scores(
        self,
        volume_path: str,
        target_label: str,
        top_token_indices: list[tuple[int, int, int]],
        random_token_indices: list[tuple[int, int, int]],
        grid_shape: tuple[int, int, int],
    ) -> dict[str, float]:
        """Compares target-score drops after top and random token-patch deletion."""
        self._load()
        import torch

        if target_label not in PATHOLOGIES.values():
            raise ValueError(f"Unknown CT-CLIP target label: {target_label}")
        text_tokens = self._text_tokens()
        volume = self._preprocess(volume_path).to(self.device)

        def score(input_volume) -> float:
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_fp16 and self.device.type == "cuda",
            ):
                logits = self.model(text_tokens, input_volume, device=self.device)
                if self.variant == "lipro":
                    probabilities = logits.sigmoid().reshape(-1)
                else:
                    probabilities = logits.reshape(len(PATHOLOGIES), 2).softmax(dim=1)[
                        :, 0
                    ]
            label_index = list(PATHOLOGIES.values()).index(target_label)
            return float(probabilities[label_index].detach().float().cpu())

        def occlude(indices: list[tuple[int, int, int]]):
            output = volume.clone()
            target_shape = output.shape[-3:]
            for token_index in indices:
                bounds: list[tuple[int, int]] = []
                for index, token_count, target_size in zip(
                    token_index, grid_shape, target_shape, strict=True
                ):
                    start = round(index * target_size / token_count)
                    end = round((index + 1) * target_size / token_count)
                    bounds.append((start, max(start + 1, end)))
                output[
                    :,
                    :,
                    bounds[0][0] : bounds[0][1],
                    bounds[1][0] : bounds[1][1],
                    bounds[2][0] : bounds[2][1],
                ] = -1.0
            return output

        baseline = score(volume)
        top_score = score(occlude(top_token_indices))
        random_score = score(occlude(random_token_indices))
        return {
            "baseline_score": baseline,
            "top_patch_score": top_score,
            "random_patch_score": random_score,
            "top_patch_drop": baseline - top_score,
            "random_patch_drop": baseline - random_score,
        }


def save_attribution_artifact(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            attributions=np.asarray(result["attributions"], dtype=np.float16),
            labels=np.asarray(list(PATHOLOGIES.values())),
            method=np.asarray(str(result.get("method", ATTRIBUTION_METHOD))),
            grid_shape=np.asarray(result["grid_shape"], dtype=np.int16),
            preprocess_json=np.asarray(
                json.dumps(result["preprocess"], sort_keys=True, ensure_ascii=True)
            ),
        )
    temporary.replace(path)
