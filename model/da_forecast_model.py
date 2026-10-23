"""Global data-assimilation forecast model.

The ERA5-only path is layer-for-layer compatible with the original global
FuXi short-range model. GNSS-ZTD is the only added branch and enters through
gated feature corrections at the two FuXi encoder scales.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange

from .fuxi import CubeEmbed, DownBlock2D, PeriodicConv2d, ResBlock, SwinLayer, UpBlock2D


def _check_feature_width(embed_dim: int, num_heads: int) -> None:
    if embed_dim % 32 != 0:
        raise ValueError("embed_dim must be divisible by 32 for GroupNorm")
    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")


class ERA5Encoder(nn.Module):
    """Original FuXi patch embedding and first down block."""

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
        self.patch_size = int(patch_size)
        self.patch_embed = CubeEmbed(
            in_chans=era5_channels + static_channels,
            in_frames=era5_frames,
            patch_size=patch_size,
            embed_dim=embed_dim,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            flatten=True,
        )
        self.down_block = DownBlock2D(
            embed_dim,
            embed_dim,
            temb_dim=embed_dim,
            num_layers=1,
            scale_shift=True,
            conv_class=PeriodicConv2d,
        )

    def encode_high(self, era5: torch.Tensor, era_static: torch.Tensor) -> torch.Tensor:
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
        static = era_static[:, None].expand(-1, t, -1, -1, -1)
        fields = torch.cat((era5, static), dim=2).reshape(
            b, t * (c + self.static_channels), h, w
        )
        tokens = self.patch_embed(fields)
        return rearrange(
            tokens, "b (h w) c -> b c h w", h=h // self.patch_size
        )

    def encode_low(self, high: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        return self.down_block(high, temb)

    def forward(
        self, era5: torch.Tensor, era_static: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        high = self.encode_high(era5, era_static)
        return high, self.encode_low(high, temb)


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
    """Original FuXi Swin stages and multi-stage FPN at 1/8 resolution."""

    def __init__(
        self,
        embed_dim: int,
        input_size: Sequence[int],
        window_size: int = 9,
        depths: Sequence[int] = (12, 12, 12, 12),
        num_heads: int = 24,
        mlp_ratio: float = 4.0,
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
                    # FuXi masks shifted windows only north/south. Longitude
                    # remains periodic across the 0/360-degree seam.
                    mask_type="h",
                )
                for depth in depths
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(embed_dim, eps=1e-6) for _ in depths]
        )
        self.fpn = nn.Sequential(
            nn.Linear(embed_dim * len(depths), embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if (h, w) != self.input_size:
            raise ValueError(
                f"expected bottleneck spatial size {self.input_size}, got {(h, w)}"
            )
        tokens = x.flatten(2).transpose(1, 2)
        outputs = []
        for layer, norm in zip(self.layers, self.norms):
            tokens = checkpoint.checkpoint(layer, tokens, use_reentrant=False)
            outputs.append(norm(tokens))
        tokens = self.fpn(torch.cat(outputs, dim=-1))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class ForecastDecoder(nn.Module):
    """FuXi time-conditioned up-block and linear patch reconstruction head."""

    def __init__(
        self,
        embed_dim: int = 1536,
        out_channels: int = 70,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.up_block = UpBlock2D(
            embed_dim,
            embed_dim,
            embed_dim,
            temb_dim=embed_dim,
            num_layers=1,
            scale_shift=True,
            conv_class=PeriodicConv2d,
        )
        self.head = nn.Linear(
            embed_dim, out_channels * self.patch_size * self.patch_size
        )

    def forward(
        self,
        bottleneck: torch.Tensor,
        low_skip: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        if bottleneck.shape != low_skip.shape:
            raise ValueError("bottleneck and low-resolution skip sizes do not match")
        x = self.up_block(torch.cat((bottleneck, low_skip), dim=1), temb)
        tokens = rearrange(x, "b c h w -> b (h w) c")
        tokens = checkpoint.checkpoint(self.head, tokens, use_reentrant=False)
        return rearrange(
            tokens,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=x.shape[-2],
            p1=self.patch_size,
            p2=self.patch_size,
        )


class DAForecastNet(nn.Module):
    """ERA5 + GNSS-ZTD global forecast network.

    Production shapes are ERA5 ``[B,2,70,720,1440]`` and observations
    ``[B,T,1,720,1440]``. The output is the directly predicted next ERA5
    analysis with shape ``[B,70,720,1440]``. FuXi predicts a six-hour
    increment, which is added to the current ERA5 analysis.
    """

    def __init__(
        self,
        image_size: Sequence[int] = (720, 1440),
        era5_channels: int = 70,
        era5_frames: int = 2,
        era_static_channels: int = 5,
        obs_static_channels: int = 3,
        out_channels: int = 70,
        embed_dim: int = 1536,
        patch_size: int = 4,
        window_size: int = 9,
        depths: Sequence[int] = (12, 12, 12, 12),
        num_heads: int = 24,
        mlp_ratio: float = 4.0,
        time_dim: int = 12,
        use_obs: bool = True,
    ) -> None:
        super().__init__()
        _check_feature_width(embed_dim, num_heads)
        if out_channels != era5_channels:
            raise ValueError("forecast output requires out_channels == era5_channels")
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
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
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
        )
        self.decoder = ForecastDecoder(embed_dim, out_channels, patch_size)

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
        if time_features.ndim != 2 or time_features.shape != (era5.shape[0], 12):
            raise ValueError("time_features must have shape [B,12]")
        temb = self.time_embed(time_features.to(era5))
        high = self.era_encoder.encode_high(era5, era_static)

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
        low = self.era_encoder.encode_low(high, temb)
        if self.use_obs:
            assert self.low_fusion is not None
            low = self.low_fusion(low, obs_low, low_mask)

        bottleneck = self.backbone(low)
        residual = self.decoder(bottleneck, low, temb)
        return era5[:, -1] + residual


__all__ = [
    "ERA5Encoder",
    "ObsEncoder",
    "GatedFusion",
    "ForecastBackbone",
    "ForecastDecoder",
    "DAForecastNet",
]
