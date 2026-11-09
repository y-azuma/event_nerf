from __future__ import annotations

import torch

from .model import RadianceField


def sphere_bounds(
    origins: torch.Tensor, directions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    projection = (origins * directions).sum(dim=-1)
    discriminant = projection.square() - (origins.square().sum(dim=-1) - 1.0)
    valid = discriminant > 0
    root = torch.sqrt(torch.clamp_min(discriminant, 0.0))
    near = torch.clamp_min(-projection - root, 0.0)
    far = torch.clamp_min(-projection + root, near + 1e-4)
    return near, far, valid


def render_rays(
    model: RadianceField,
    origins: torch.Tensor,
    directions: torch.Tensor,
    samples_per_ray: int,
    background_intensity: float,
    perturb: bool,
) -> dict[str, torch.Tensor]:
    near, far, valid = sphere_bounds(origins, directions)
    base = torch.linspace(0.0, 1.0, samples_per_ray, device=origins.device)
    depths = near[:, None] * (1.0 - base) + far[:, None] * base
    if perturb and samples_per_ray > 1:
        mids = 0.5 * (depths[:, 1:] + depths[:, :-1])
        upper = torch.cat([mids, depths[:, -1:]], dim=-1)
        lower = torch.cat([depths[:, :1], mids], dim=-1)
        depths = lower + (upper - lower) * torch.rand_like(depths)
    points = origins[:, None, :] + directions[:, None, :] * depths[..., None]
    viewdirs = directions[:, None, :].expand_as(points)
    sigma, gray = model(points.reshape(-1, 3), viewdirs.reshape(-1, 3))
    sigma = sigma.reshape(points.shape[:-1]) * valid[:, None]
    gray = gray.reshape(points.shape[:-1])
    delta = depths[:, 1:] - depths[:, :-1]
    delta = torch.cat([delta, torch.full_like(delta[:, :1], 1e3)], dim=-1)
    alpha = 1.0 - torch.exp(-sigma * delta)
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-8], dim=-1), dim=-1
    )[:, :-1]
    weights = alpha * transmittance
    opacity = weights.sum(dim=-1)
    intensity = (weights * gray).sum(dim=-1) + (1.0 - opacity) * background_intensity
    expected_depth = (weights * depths).sum(dim=-1)
    distortion = (weights.square() * delta.clamp_max(2.0)).sum(dim=-1)
    return {
        "intensity": intensity.clamp_min(1e-6),
        "opacity": opacity,
        "depth": expected_depth,
        "distortion": distortion,
    }
