from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types

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

        # Prefer an explicit local path, then the standard sibling models/cxrbert
        # directory.  This keeps the protected CT-CLIP workflow offline after assets
        # are provisioned and only falls back to the public Hub name for fresh setups.
        configured_text_model = os.environ.get("CTCLIP_TEXT_MODEL_DIR", "").strip()
        sibling_text_model = self.checkpoint.parent.parent / "cxrbert"
        if configured_text_model and Path(configured_text_model).is_dir():
            text_model_source = configured_text_model
            text_model_kwargs = {"local_files_only": True}
        elif sibling_text_model.is_dir():
            text_model_source = str(sibling_text_model)
            text_model_kwargs = {"local_files_only": True}
        else:
            text_model_source = "microsoft/BiomedVLP-CXR-BERT-specialized"
            text_model_kwargs = {}
        self.tokenizer = BertTokenizer.from_pretrained(
            text_model_source, do_lower_case=True, **text_model_kwargs
        )
        text_encoder = BertModel.from_pretrained(text_model_source, **text_model_kwargs)
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
        import torch.nn.functional as functional

        slices = []
        for current, target in zip(tensor.shape[-3:], target_shape, strict=True):
            start = max((current - target) // 2, 0)
            slices.append(slice(start, min(start + target, current)))
        tensor = tensor[(..., *slices)]

        pads: list[int] = []
        current_shape = tensor.shape[-3:]
        for current, target in reversed(list(zip(current_shape, target_shape, strict=True))):
            total = max(target - current, 0)
            pads.extend([total // 2, total - total // 2])
        return functional.pad(tensor, pads, value=-1.0)

    def _preprocess(self, volume_path: str):
        import torch
        import torch.nn.functional as functional

        image = nib.load(volume_path)
        volume = image.get_fdata(dtype=np.float32)
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
        return self._center_crop_pad(tensor, (240, 480, 480))

    def predict(self, volume_path: str) -> dict[str, float]:
        self._load()
        import torch

        if self.variant == "lipro":
            text_tokens = self.tokenizer(
                "",
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=200,
            ).to(self.device)
        else:
            prompts: list[str] = []
            for pathology in PATHOLOGIES:
                prompts.extend(
                    [f"{pathology} is present.", f"{pathology} is not present."]
                )
            text_tokens = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64,
            ).to(self.device)
        volume = self._preprocess(volume_path).to(self.device)

        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_fp16 and self.device.type == "cuda",
        ):
            logits = self.model(text_tokens, volume, device=self.device)
            if self.variant == "lipro":
                probabilities = logits.sigmoid().reshape(-1)
            else:
                probabilities = logits.reshape(len(PATHOLOGIES), 2).softmax(dim=1)[:, 0]
        return {
            internal_name: float(probability)
            for internal_name, probability in zip(
                PATHOLOGIES.values(), probabilities.detach().float().cpu(), strict=True
            )
        }
