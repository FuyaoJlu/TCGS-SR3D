from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

def _group_count(channels: int) -> int:
    groups = min(16, channels)
    while channels % groups:
        groups -= 1
    return groups

class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int, residual_scale: float = 0.1):
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.act = nn.GELU()
        self.residual_scale = float(residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(self.act(self.norm1(x)))
        y = self.conv2(self.act(self.norm2(y)))
        return x + self.residual_scale * y

class TCGSSR3D(nn.Module):
    

    def __init__(
        self,
        scale: int = 2,
        hidden_channels: int = 192,
        feature_channels: int = 96,
        num_blocks: int = 24,
        hr_blocks: int = 4,
        levels: str = "tri",
        gradient_checkpointing: bool = True,
        mid_radius: int = 2,
        coarse_radius: int = 2,
    ):
        super().__init__()
        if levels not in {"fine", "fine_mid", "tri"}:
            raise ValueError(f"Unsupported levels: {levels}")
        if scale < 1:
            raise ValueError("scale must be positive")
        self.scale = int(scale)
        self.levels = levels
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.mid_radius = int(mid_radius)
        self.coarse_radius = int(coarse_radius)

        self.stem = nn.Conv3d(1, hidden_channels, 3, padding=1)
        self.blocks = nn.ModuleList([ResidualBlock3D(hidden_channels) for _ in range(num_blocks)])
        self.trunk_end = nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1)
        self.hr_projection = nn.Conv3d(hidden_channels, feature_channels, 3, padding=1)
        self.hr_blocks = nn.ModuleList([ResidualBlock3D(feature_channels) for _ in range(hr_blocks)])

        self.fine_value = nn.Conv3d(feature_channels, 1, 3, padding=1)
        self.fine_params = nn.Conv3d(feature_channels, 7, 3, padding=1)
        self.mid_value = nn.Conv3d(feature_channels, 1, 3, padding=1)
        self.mid_params = nn.Conv3d(feature_channels, 9, 3, padding=1)
        self.coarse_value = nn.Conv3d(feature_channels, 1, 3, padding=1)
        self.coarse_params = nn.Conv3d(feature_channels, 14, 3, padding=1)
        self.gate_head = nn.Conv3d(feature_channels, 3, 3, padding=1)

        self.residual_gain = nn.Parameter(torch.tensor(0.1))
        self._initialize_heads()

    def _initialize_heads(self) -> None:
        for layer in (self.fine_value, self.mid_value, self.coarse_value):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.gate_head.weight)
        with torch.no_grad():
            self.gate_head.bias.copy_(torch.tensor([2.0, 0.0, -1.0]))

    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and x.requires_grad:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.stem(x)
        feat = stem
        for block in self.blocks:
            feat = self._run_block(block, feat)
        feat = stem + self.trunk_end(feat)
        feat = F.interpolate(feat, scale_factor=self.scale, mode="trilinear", align_corners=False)
        feat = self.hr_projection(feat)
        for block in self.hr_blocks:
            feat = self._run_block(block, feat)
        return feat

    @staticmethod
    def _bounded(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.sigmoid(raw)

    @staticmethod
    def _neighbor_stack(
        value: torch.Tensor,
        inline_radius: int,
        time_radius: int,
        xline_radius: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if value.shape[1] != 1:
            raise ValueError("Gaussian rendering heads must be scalar")
        ri, rt, rx = inline_radius, time_radius, xline_radius
        padded = F.pad(value, (rx, rx, rt, rt, ri, ri), mode="replicate")
        d, h, w = value.shape[-3:]
        neighbours = []
        coordinates = []
        for di in range(-ri, ri + 1):
            for dt in range(-rt, rt + 1):
                for dx in range(-rx, rx + 1):
                    neighbours.append(
                        padded[:, 0, di + ri : di + ri + d, dt + rt : dt + rt + h, dx + rx : dx + rx + w]
                    )
                    coordinates.append((di, dt, dx))
        stack = torch.stack(neighbours, dim=1)
        coords = value.new_tensor(coordinates).view(1, -1, 1, 1, 1, 3)
        return stack, coords

    @classmethod
    def _point_gaussian(
        cls,
        value: torch.Tensor,
        offset: torch.Tensor,
        sigma: torch.Tensor,
        opacity: torch.Tensor,
    ) -> torch.Tensor:
        neighbours, coords = cls._neighbor_stack(value, 1, 1, 1)
        source_opacity, _ = cls._neighbor_stack(opacity, 1, 1, 1)
        delta = coords - offset.movedim(1, -1).unsqueeze(1)
        scale = sigma.movedim(1, -1).unsqueeze(1).clamp_min(0.05)
        distance2 = (delta / scale).square().sum(dim=-1)
        weight = torch.exp(-0.5 * distance2) * source_opacity
        return (neighbours * weight).sum(dim=1, keepdim=True) / weight.sum(dim=1, keepdim=True).clamp_min(1e-6)

    @classmethod
    def _event_sheet(
        cls,
        value: torch.Tensor,
        offset: torch.Tensor,
        dip_inline: torch.Tensor,
        dip_xline: torch.Tensor,
        sigma_inline: torch.Tensor,
        sigma_xline: torch.Tensor,
        sigma_normal: torch.Tensor,
        opacity: torch.Tensor,
        radius: int,
        curvature: torch.Tensor | None = None,
        visibility: torch.Tensor | None = None,
        termination: torch.Tensor | None = None,
    ) -> torch.Tensor:
        neighbours, coords = cls._neighbor_stack(value, radius, 1, radius)
        source_opacity, _ = cls._neighbor_stack(opacity, radius, 1, radius)
        offset_last = offset.movedim(1, -1).unsqueeze(1)
        delta = coords - offset_last
        di, dt, dx = delta.unbind(dim=-1)
        surface_time = dip_inline * di + dip_xline * dx
        if curvature is not None:
            c_inline, c_cross, c_xline = curvature.split(1, dim=1)
            surface_time = surface_time + c_inline * di.square() + c_cross * di * dx + c_xline * dx.square()
        normal = dt - surface_time
        distance2 = (di / sigma_inline.clamp_min(0.05)).square()
        distance2 = distance2 + (dx / sigma_xline.clamp_min(0.05)).square()
        distance2 = distance2 + (normal / sigma_normal.clamp_min(0.05)).square()
        weight = torch.exp(-0.5 * distance2) * source_opacity
        if visibility is not None:
            source_visibility, _ = cls._neighbor_stack(visibility, radius, 1, radius)
            weight = weight * source_visibility
        if termination is not None:
            source_termination, _ = cls._neighbor_stack(termination, radius, 1, radius)
            radial = torch.sqrt(di.square() + dx.square() + 1e-6)
            weight = weight * torch.exp(-0.5 * radial * (termination + source_termination))
        return (neighbours * weight).sum(dim=1, keepdim=True) / weight.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> Any:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,Inline,Time,Xline], got {tuple(x.shape)}")
        baseline = F.interpolate(x, scale_factor=self.scale, mode="trilinear", align_corners=False)
        feat = self.encode(x)

        fine_raw = self.fine_params(feat)
        fine_offset = 0.45 * torch.tanh(fine_raw[:, 0:3])
        fine_sigma = self._bounded(fine_raw[:, 3:6], 0.18, 0.80)
        fine_opacity = torch.sigmoid(fine_raw[:, 6:7]) + 1e-4
        fine = self._point_gaussian(self.fine_value(feat), fine_offset, fine_sigma, fine_opacity)

        zeros = torch.zeros_like(fine)
        ones = torch.ones_like(fine)
        if self.levels in {"fine_mid", "tri"}:
            mid_raw = self.mid_params(feat)
            mid_offset = 0.75 * torch.tanh(mid_raw[:, 0:3])
            mid_dip_inline = 1.5 * torch.tanh(mid_raw[:, 3:4])
            mid_dip_xline = 1.5 * torch.tanh(mid_raw[:, 4:5])
            mid_sigma_inline = self._bounded(mid_raw[:, 5:6], 0.8, 2.8)
            mid_sigma_xline = self._bounded(mid_raw[:, 6:7], 0.8, 2.8)
            mid_sigma_normal = self._bounded(mid_raw[:, 7:8], 0.22, 0.95)
            mid_opacity = torch.sigmoid(mid_raw[:, 8:9]) + 1e-4
            mid = self._event_sheet(
                self.mid_value(feat), mid_offset, mid_dip_inline, mid_dip_xline,
                mid_sigma_inline, mid_sigma_xline, mid_sigma_normal, mid_opacity,
                radius=self.mid_radius,
            )
        else:
            mid = zeros
            mid_dip_inline = mid_dip_xline = zeros
            mid_sigma_inline = mid_sigma_xline = mid_sigma_normal = ones

        if self.levels == "tri":
            coarse_raw = self.coarse_params(feat)
            coarse_offset = 0.9 * torch.tanh(coarse_raw[:, 0:3])
            coarse_dip_inline = 2.0 * torch.tanh(coarse_raw[:, 3:4])
            coarse_dip_xline = 2.0 * torch.tanh(coarse_raw[:, 4:5])
            coarse_sigma_inline = self._bounded(coarse_raw[:, 5:6], 1.2, 4.5)
            coarse_sigma_xline = self._bounded(coarse_raw[:, 6:7], 1.2, 4.5)
            coarse_sigma_normal = self._bounded(coarse_raw[:, 7:8], 0.28, 1.20)
            coarse_curvature = 0.16 * torch.tanh(coarse_raw[:, 8:11])
            coarse_opacity = torch.sigmoid(coarse_raw[:, 11:12]) + 1e-4
            coarse_visibility = torch.sigmoid(coarse_raw[:, 12:13])
            coarse_termination = torch.sigmoid(coarse_raw[:, 13:14])
            coarse = self._event_sheet(
                self.coarse_value(feat), coarse_offset, coarse_dip_inline, coarse_dip_xline,
                coarse_sigma_inline, coarse_sigma_xline, coarse_sigma_normal, coarse_opacity,
                radius=self.coarse_radius, curvature=coarse_curvature,
                visibility=coarse_visibility, termination=coarse_termination,
            )
        else:
            coarse = zeros
            coarse_dip_inline = coarse_dip_xline = zeros
            coarse_sigma_inline = coarse_sigma_xline = coarse_sigma_normal = ones
            coarse_curvature = torch.zeros_like(fine).expand(-1, 3, -1, -1, -1)
            coarse_visibility = ones
            coarse_termination = zeros

        gate_logits = self.gate_head(feat)
        if self.levels == "fine":
            gates = torch.zeros_like(gate_logits)
            gates[:, 0:1] = 1.0
        elif self.levels == "fine_mid":
            two = torch.softmax(gate_logits[:, :2], dim=1)
            gates = torch.cat((two, torch.zeros_like(two[:, :1])), dim=1)
        else:
            gated_coarse = gate_logits[:, 2:3] + torch.log(coarse_visibility.clamp_min(1e-4))
            gates = torch.softmax(torch.cat((gate_logits[:, :2], gated_coarse), dim=1), dim=1)

        residual = (gates * torch.cat((fine, mid, coarse), dim=1)).sum(dim=1, keepdim=True)
        pred = baseline + self.residual_gain.clamp(0.01, 2.0) * residual
        if not return_aux:
            return pred
        aux = {
            "baseline": baseline,
            "feature_norm": torch.linalg.vector_norm(feat.float(), dim=1, keepdim=True),
            "feature_mean": feat.float().mean(dim=1, keepdim=True),
            "fine": fine,
            "mid": mid,
            "coarse": coarse,
            "gates": gates,
            "fine_sigma": fine_sigma,
            "mid_dip_inline": mid_dip_inline,
            "mid_dip_xline": mid_dip_xline,
            "mid_sigma_inline": mid_sigma_inline,
            "mid_sigma_xline": mid_sigma_xline,
            "mid_sigma_normal": mid_sigma_normal,
            "coarse_dip_inline": coarse_dip_inline,
            "coarse_dip_xline": coarse_dip_xline,
            "coarse_sigma_inline": coarse_sigma_inline,
            "coarse_sigma_xline": coarse_sigma_xline,
            "coarse_sigma_normal": coarse_sigma_normal,
            "coarse_curvature": coarse_curvature,
            "coarse_visibility": coarse_visibility,
            "coarse_termination": coarse_termination,
        }
        return pred, aux

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
