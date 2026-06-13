from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from .lfp_encoder import LFPEncoder


class CLIPLFPEncoder(nn.Module):
    """Locked inference path for the CLIP-aligned LFP representation."""

    def __init__(
        self,
        encoder_kwargs: dict[str, Any],
        pool: Literal["cls", "mean"],
        projector_input_dim: int,
        projector_hidden_dim: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.backbone = LFPEncoder(**encoder_kwargs)
        self.pool = pool
        self.projector = nn.Sequential(
            nn.Linear(projector_input_dim, projector_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_hidden_dim, embedding_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(values, pool=self.pool)
        return F.normalize(self.projector(features), dim=-1)


def load_clip_lfp_encoder(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    device: torch.device,
) -> CLIPLFPEncoder:
    model = CLIPLFPEncoder(
        encoder_kwargs=config["encoder"],
        pool=config["pool"],
        projector_input_dim=config["projector"]["input_dim"],
        projector_hidden_dim=config["projector"]["hidden_dim"],
        embedding_dim=config["projector"]["output_dim"],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("CLIP checkpoint must contain an inference package dictionary.")
    if checkpoint.get("format_version") != 1 or checkpoint.get("model_type") != "clip_lfp_inference":
        raise ValueError("Unsupported or non-inference CLIP checkpoint format.")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("CLIP inference checkpoint is missing state_dict.")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"CLIP inference checkpoint architecture mismatch: {error}") from error
    model.requires_grad_(False).eval()
    return model.to(device)
