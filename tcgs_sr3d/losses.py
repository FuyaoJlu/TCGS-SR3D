from __future__ import annotations

import torch
import torch.nn.functional as F

def charbonnier(x: torch.Tensor, y: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((x - y).square() + epsilon**2).mean()

def gradients3d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inline = x[..., 1:, :, :] - x[..., :-1, :, :]
    time = x[..., :, 1:, :] - x[..., :, :-1, :]
    xline = x[..., :, :, 1:] - x[..., :, :, :-1]
    return inline, time, xline

def laplacian3d(x: torch.Tensor) -> torch.Tensor:
    kernel = x.new_zeros((3, 3, 3))
    kernel[1, 1, 1] = -6.0
    kernel[0, 1, 1] = kernel[2, 1, 1] = 1.0
    kernel[1, 0, 1] = kernel[1, 2, 1] = 1.0
    kernel[1, 1, 0] = kernel[1, 1, 2] = 1.0
    return F.conv3d(x, kernel.view(1, 1, 3, 3, 3), padding=1)

def high_frequency3d(x: torch.Tensor) -> torch.Tensor:
    return x - F.avg_pool3d(x, 5, stride=1, padding=2)

def probability_bce(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = probability.float().clamp(1e-5, 1.0 - 1e-5)
    target = target.float()
    return -(target * probability.log() + (1.0 - target) * (1.0 - probability).log()).mean()

def multiscale_gate_target3d(target: torch.Tensor, temperature: float = 0.35) -> torch.Tensor:
    smooth3 = F.avg_pool3d(target, 3, stride=1, padding=1)
    smooth7 = F.avg_pool3d(target, 7, stride=1, padding=3)
    fine = (target - smooth3).abs()
    middle = (smooth3 - smooth7).abs()
    coarse = smooth7.abs()
    energies = torch.cat((fine, middle, coarse), dim=1)
    scale = energies.flatten(2).mean(dim=2).view(target.shape[0], 3, 1, 1, 1).clamp_min(1e-5)
    return torch.softmax(energies / scale / temperature, dim=1)

def tri_level_loss3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    aux: dict[str, torch.Tensor],
    structure_weight: float = 1.0,
    levels: str = "tri",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    voxel = charbonnier(pred, target)
    pred_grad = gradients3d(pred)
    target_grad = gradients3d(target)
    grad = sum(charbonnier(a, b) for a, b in zip(pred_grad, target_grad))
    lap = charbonnier(laplacian3d(pred), laplacian3d(target))
    hf = charbonnier(high_frequency3d(pred), high_frequency3d(target))

    gate_target = multiscale_gate_target3d(target)
    gates = aux["gates"].clamp_min(1e-6)
    if levels == "fine":
        gate_class = gates.new_zeros(())
    elif levels == "fine_mid":
        target_two = gate_target[:, :2]
        target_two = target_two / target_two.sum(dim=1, keepdim=True).clamp_min(1e-6)
        gate_class = -(target_two * gates[:, :2].log()).sum(dim=1).mean()
    else:
        gate_class = -(gate_target * gates.log()).sum(dim=1).mean()
    gate_entropy = -(gates * gates.log()).sum(dim=1).mean()

    if levels == "tri":
        visibility_target = (1.0 - gate_target[:, 0:1]).detach()
        visibility = probability_bce(aux["coarse_visibility"], visibility_target)
        termination_target = F.max_pool3d(gate_target[:, 0:1], 5, stride=1, padding=2).detach()
        termination = probability_bce(aux["coarse_termination"], termination_target)
        sparse_coarse = (gates[:, 2:3] * (1.0 - aux["coarse_visibility"])).mean()
        curvature = aux["coarse_curvature"].abs().mean()
    else:
        visibility = termination = sparse_coarse = curvature = gates.new_zeros(())

    total = voxel + 0.55 * grad + 0.25 * lap + 0.20 * hf
    total = total + structure_weight * (
        0.06 * gate_class - 0.002 * gate_entropy + 0.02 * visibility
        + 0.01 * termination + 0.02 * sparse_coarse + 0.002 * curvature
    )
    return total, {
        "voxel": voxel,
        "grad": grad,
        "lap": lap,
        "hf": hf,
        "gate_class": gate_class,
        "gate_entropy": gate_entropy,
        "visibility": visibility,
        "termination": termination,
        "sparse_coarse": sparse_coarse,
        "curvature": curvature,
    }

@torch.no_grad()
def metrics3d(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.float()
    target = target.float()
    error = pred - target
    mse = error.square().mean().clamp_min(1e-12)
    data_range = (target.max() - target.min()).clamp_min(1.0)
    psnr = 20.0 * torch.log10(data_range / torch.sqrt(mse))
    snr = 10.0 * torch.log10(target.square().sum().clamp_min(1e-12) / error.square().sum().clamp_min(1e-12))
    pred_grad = gradients3d(pred)
    target_grad = gradients3d(target)
    pred_energy = sum(item.abs().mean() for item in pred_grad)
    target_energy = sum(item.abs().mean() for item in target_grad)
    return {
        "psnr": float(psnr),
        "snr": float(snr),
        "grad_ratio": float(pred_energy / target_energy.clamp_min(1e-12)),
        "lap_l1": float((laplacian3d(pred) - laplacian3d(target)).abs().mean()),
        "hf_l1": float((high_frequency3d(pred) - high_frequency3d(target)).abs().mean()),
    }
