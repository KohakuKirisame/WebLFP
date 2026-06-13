from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


class LFPEncoder(nn.Module):
    """Inference-only LFP Transformer encoder."""

    def __init__(
        self,
        time_patch_size: int,
        channel_pad_multiple: int,
        max_channels: int,
        max_time_patches: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        pad_value: float,
    ) -> None:
        super().__init__()
        self.time_patch_size = time_patch_size
        self.channel_pad_multiple = channel_pad_multiple
        self.max_channels = max_channels
        self.max_time_patches = max_time_patches
        self.pad_value = pad_value

        self.patch_embed = nn.Linear(time_patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.channel_embed = nn.Embedding(max_channels, embed_dim)
        self.time_embed = nn.Embedding(max_time_patches, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(embed_dim),
        )

    def _patchify(self, values: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if values.ndim != 3:
            raise ValueError(f"Expected [B, C, T] input, got shape={tuple(values.shape)}.")
        batch_size, channels, time_samples = values.shape
        pad_time = (-time_samples) % self.time_patch_size
        pad_channels = 0
        if self.channel_pad_multiple > 1:
            pad_channels = (-channels) % self.channel_pad_multiple
        padded = F.pad(values, (0, pad_time, 0, pad_channels), value=self.pad_value)
        padded_channels = channels + pad_channels
        time_patches = (time_samples + pad_time) // self.time_patch_size
        if padded_channels > self.max_channels:
            raise ValueError(
                f"Padded channel count {padded_channels} exceeds limit {self.max_channels}."
            )
        if time_patches > self.max_time_patches:
            raise ValueError(
                f"Time patch count {time_patches} exceeds limit {self.max_time_patches}."
            )
        patches = padded.view(
            batch_size,
            padded_channels,
            time_patches,
            self.time_patch_size,
        ).reshape(batch_size, padded_channels * time_patches, self.time_patch_size)
        return patches, padded_channels, time_patches

    def forward_features(
        self,
        values: torch.Tensor,
        pool: Literal["cls", "mean"] = "cls",
    ) -> torch.Tensor:
        patches, channels, time_patches = self._patchify(values)
        tokens = self.patch_embed(patches)
        channel_ids = torch.arange(channels, device=values.device)
        time_ids = torch.arange(time_patches, device=values.device)
        positions = (
            self.channel_embed(channel_ids)[:, None, :]
            + self.time_embed(time_ids)[None, :, :]
        ).reshape(1, channels * time_patches, -1)
        tokens = tokens + positions
        cls_token = (self.cls_token + self.cls_pos_embed).expand(tokens.shape[0], -1, -1)
        latent = self.blocks(torch.cat((cls_token, tokens), dim=1))
        if pool == "cls":
            return latent[:, 0]
        if pool == "mean":
            return latent[:, 1:].mean(dim=1)
        raise ValueError(f"Unsupported pool mode: {pool}")

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.forward_features(values)
