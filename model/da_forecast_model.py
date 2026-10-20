"""Regional data-assimilation forecast model.

The model encodes ERA5 fields and gridded GNSS-ZTD observations separately,
fuses them at two spatial scales, and predicts an increment over the latest
ERA5 analysis.  Unlike the global FuXi model, all convolutions in this module
use ordinary (non-periodic) padding.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fuxi import ResBlock, SwinLayer


def _check_feature_width(embed_dim: int, num_heads: int) -> None:
    if embed_dim % 32 != 0:
        raise ValueError("embed_dim must be divisible by 32 for GroupNorm")
    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")


class ERA5Encoder(nn.Module):
    """Encode two ERA5 analyses and static fields at 1/4 and 1/8 scales."""

    def __init__(
        self,
        era5_channels: int = 70,
        era5_frames: int = 2,
        static_channels: int = 5,
        embed_dim: int = 256,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        self.era5_channels = era5_channels
        self.era5_frames = era5_frames
        self.static_channels = static_channels
        self.patch_size = patch_size

        self.field_patch = nn.Conv2d(
            era5_channels * era5_frames,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.static_patch = nn.Conv2d(
            static_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.high_refine = ResBlock(embed_dim, embed_dim, conv_class=nn.Conv2d)
        self.downsample = nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1)
        self.low_refine = ResBlock(embed_dim, embed_dim, conv_class=nn.Conv2d)

    def forward(
        self, era5: torch.Tensor, era_static: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if era5.ndim != 5:
            raise ValueError("era5 must have shape [B, T, C, H, W]")
        b, t, c, h, w = era5.shape
        if (t, c) != (self.era5_frames, self.era5_channels):
            raise ValueError(
                f"expected ERA5 [B,{self.era5_frames},{self.era5_channels},H,W], "
                f"got {tuple(era5.shape)}"
            )
        if era_static.shape != (b, self.static_channels, h, w):
            raise ValueError(
                f"expected era_static {(b, self.static_channels, h, w)}, "
                f"got {tuple(era_static.shape)}"
            )
        fields = era5.reshape(b, t * c, h, w)
        high = self.field_patch(fields) + self.static_patch(era_static)
        high = self.high_refine(high)
        low = self.low_refine(self.downsample(high))
        return high, low


class ObsEncoder(nn.Module):
    """Mask-aware encoder for a time sequence of gridded GNSS-ZTD values."""

    def __init__(
        self,
        obs_static_channels: int = 3,
        embed_dim: int = 256,
        patch_size: int = 4,
        temporal_channels: Tuple[int, int] = (32, 64),
    ) -> None:
        super().__init__()
        self.obs_static_channels = obs_static_channels
        self.patch_size = patch_size
        mid_channels, out_channels = temporal_channels
        self.temporal = nn.Sequential(
            nn.Conv3d(2, mid_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(mid_channels, out_channels, 3, padding=1),
            nn.SiLU(),
        )
        self.patch = nn.Conv2d(
            out_channels + obs_static_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.high_refine = ResBlock(embed_dim, embed_dim, conv_class=nn.Conv2d)
        self.downsample = nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1)
        self.low_refine = ResBlock(embed_dim, embed_dim, conv_class=nn.Conv2d)

    def forward(
        self,
        obs_ztd: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_static: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs_ztd.ndim != 5 or obs_ztd.shape[2] != 1:
            raise ValueError("obs_ztd must have shape [B, T, 1, H, W]")
        if obs_mask.shape != obs_ztd.shape:
            raise ValueError("obs_mask must have the same shape as obs_ztd")
        b, _, _, h, w = obs_ztd.shape
        if obs_static.shape != (b, self.obs_static_channels, h, w):
            raise ValueError(
                f"expected obs_static {(b, self.obs_static_channels, h, w)}, "
                f"got {tuple(obs_static.shape)}"
            )

        # Treat every non-positive value as missing and prevent NaNs in the
        # supplied ZTD array from entering the convolution.
        mask = ((obs_mask > 0) & torch.isfinite(obs_ztd)).to(dtype=obs_ztd.dtype)
        ztd = torch.where(mask.bool(), obs_ztd, torch.zeros_like(obs_ztd))
        temporal_input = torch.cat((ztd * mask, mask), dim=2).permute(0, 2, 1, 3, 4)
        features = self.temporal(temporal_input)

        time_mask = mask.permute(0, 2, 1, 3, 4)
        count = time_mask.sum(dim=2).clamp_min(1.0)
        features = (features * time_mask).sum(dim=2) / count
        station_mask = time_mask.amax(dim=2)
        features = features * station_mask
        static = obs_static * station_mask

        high_mask = F.max_pool2d(
            station_mask, kernel_size=self.patch_size, stride=self.patch_size
        )
        high = self.patch(torch.cat((features, static), dim=1))
        # Conv bias and refinement must not create a fake observation in an
        # empty patch.  This also makes the all-missing path exactly neutral.
        high = self.high_refine(high) * high_mask
        low_mask = F.max_pool2d(high_mask, kernel_size=2, stride=2)
        low = self.low_refine(self.downsample(high)) * low_mask
        return high, low, high_mask, low_mask


class GatedFusion(nn.Module):
    """Add a gated observation correction to the ERA5 feature map."""

    def __init__(self, embed_dim: int, gate_init: float = -4.0) -> None:
        super().__init__()
        self.correction = nn.Sequential(
            nn.Conv2d(2 * embed_dim + 1, embed_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
        )
        self.gate_logit = nn.Parameter(
            torch.full((1, embed_dim, 1, 1), float(gate_init))
        )

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(
        self, era: torch.Tensor, obs: torch.Tensor, patch_mask: torch.Tensor
    ) -> torch.Tensor:
        if era.shape != obs.shape:
            raise ValueError("era and obs feature maps must have the same shape")
        if patch_mask.shape != (era.shape[0], 1, era.shape[2], era.shape[3]):
            raise ValueError("patch_mask has an incompatible shape")
        delta = self.correction(torch.cat((era, obs, patch_mask), dim=1))
        # Preserve spatial spreading when observations exist, while making an
        # all-missing sample identical to the ERA-only path.
        available = patch_mask.amax(dim=(-2, -1), keepdim=True)
        return era + available * self.gate * delta


class ForecastBackbone(nn.Module):
    """Lightweight regional Swin backbone operating on the 1/8 feature map."""

    def __init__(
        self,
        embed_dim: int,
        input_size: Sequence[int],
        window_size: int = 5,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        time_dim: int = 12,
    ) -> None:
        super().__init__()
        self.input_size = tuple(int(v) for v in input_size)
        self.window_size = int(window_size)
        if any(v % self.window_size != 0 for v in self.input_size):
            raise ValueError(
                f"bottleneck size {self.input_size} must be divisible by "
                f"window_size={self.window_size}"
            )
        if not depths or any(depth <= 0 for depth in depths):
            raise ValueError("depths must contain positive integers")

        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.layers = nn.ModuleList(
            [
                SwinLayer(
                    in_chans=embed_dim,
                    embed_dim=embed_dim,
                    input_size=self.input_size,
                    window_size=self.window_size,
                    depth=depth,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    mask_type="hw",
                )
                for depth in depths
            ]
        )
        self.merge = nn.Linear(embed_dim * len(depths), embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if (h, w) != self.input_size:
            raise ValueError(
                f"expected bottleneck spatial size {self.input_size}, got {(h, w)}"
            )
        tokens = x.flatten(2).transpose(1, 2)
        if time_features.ndim != 2 or time_features.shape[0] != b:
            raise ValueError("time_features must have shape [B, time_dim]")
        tokens = tokens + self.time_embed(time_features).unsqueeze(1)
        outputs = []
        for layer in self.layers:
            tokens = layer(tokens)
            outputs.append(tokens)
        tokens = self.norm(self.merge(torch.cat(outputs, dim=-1)))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class ForecastDecoder(nn.Module):
    """Decode the bottleneck with fused low/high-resolution skip features."""

    def __init__(self, embed_dim: int = 256, out_channels: int = 70) -> None:
        super().__init__()
        self.low_fuse = ResBlock(2 * embed_dim, embed_dim, conv_class=nn.Conv2d)
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1)
        self.high_fuse = ResBlock(2 * embed_dim, embed_dim, conv_class=nn.Conv2d)
        self.head = nn.ConvTranspose2d(embed_dim, out_channels, 4, stride=4)
        # Start close to persistence while retaining a gradient path into the
        # encoders/backbone on the first optimization step.
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(
        self, bottleneck: torch.Tensor, low_skip: torch.Tensor, high_skip: torch.Tensor
    ) -> torch.Tensor:
        x = self.low_fuse(torch.cat((bottleneck, low_skip), dim=1))
        x = self.up(x)
        if x.shape[-2:] != high_skip.shape[-2:]:
            raise ValueError("decoder and high-resolution skip sizes do not match")
        x = self.high_fuse(torch.cat((x, high_skip), dim=1))
        return self.head(x)


class DAForecastNet(nn.Module):
    """ERA5 + GNSS-ZTD regional forecast network.

    Expected production shapes are ERA5 ``[B,2,70,80,120]`` and observations
    ``[B,T,1,80,120]``.  The output is ``[B,70,80,120]`` and is formed as an
    increment over the latest ERA5 analysis.
    """

    def __init__(
        self,
        image_size: Sequence[int] = (80, 120),
        era5_channels: int = 70,
        era5_frames: int = 2,
        era_static_channels: int = 5,
        obs_static_channels: int = 3,
        out_channels: int = 70,
        embed_dim: int = 256,
        patch_size: int = 4,
        window_size: int = 5,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        time_dim: int = 12,
        use_obs: bool = True,
    ) -> None:
        super().__init__()
        _check_feature_width(embed_dim, num_heads)
        if out_channels != era5_channels:
            raise ValueError("residual forecast requires out_channels == era5_channels")
        self.image_size = tuple(int(v) for v in image_size)
        self.era5_channels = era5_channels
        self.era5_frames = era5_frames
        self.era_static_channels = era_static_channels
        self.obs_static_channels = obs_static_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.use_obs = use_obs

        scale = patch_size * 2
        if any(v % scale != 0 for v in self.image_size):
            raise ValueError(
                f"image_size={self.image_size} must be divisible by {scale}"
            )
        bottleneck_size = tuple(v // scale for v in self.image_size)

        self.era_encoder = ERA5Encoder(
            era5_channels, era5_frames, era_static_channels, embed_dim, patch_size
        )
        if use_obs:
            self.obs_encoder: Optional[ObsEncoder] = ObsEncoder(
                obs_static_channels, embed_dim, patch_size
            )
            self.high_fusion: Optional[GatedFusion] = GatedFusion(embed_dim)
            self.low_fusion: Optional[GatedFusion] = GatedFusion(embed_dim)
        else:
            self.obs_encoder = None
            self.high_fusion = None
            self.low_fusion = None
        self.backbone = ForecastBackbone(
            embed_dim,
            bottleneck_size,
            window_size,
            depths,
            num_heads,
            mlp_ratio,
            time_dim,
        )
        self.decoder = ForecastDecoder(embed_dim, out_channels)

    def forward(
        self,
        era5: torch.Tensor,
        era_static: torch.Tensor,
        time_features: torch.Tensor,
        obs_ztd: Optional[torch.Tensor] = None,
        obs_mask: Optional[torch.Tensor] = None,
        obs_static: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tuple(era5.shape[-2:]) != self.image_size:
            raise ValueError(
                f"expected image size {self.image_size}, got {tuple(era5.shape[-2:])}"
            )
        high, low = self.era_encoder(era5, era_static)

        if self.use_obs:
            if obs_ztd is None or obs_mask is None or obs_static is None:
                raise ValueError(
                    "obs_ztd, obs_mask and obs_static are required when use_obs=True"
                )
            assert self.obs_encoder is not None
            assert self.high_fusion is not None and self.low_fusion is not None
            obs_high, obs_low, high_mask, low_mask = self.obs_encoder(
                obs_ztd, obs_mask, obs_static
            )
            high = self.high_fusion(high, obs_high, high_mask)
            low = self.low_fusion(low, obs_low, low_mask)

        bottleneck = self.backbone(low, time_features)
        delta = self.decoder(bottleneck, low, high)
        return era5[:, -1] + delta


__all__ = [
    "ERA5Encoder",
    "ObsEncoder",
    "GatedFusion",
    "ForecastBackbone",
    "ForecastDecoder",
    "DAForecastNet",
]
