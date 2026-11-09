from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, levels: int, include_input: bool = True):
        super().__init__()
        self.levels = int(levels)
        self.include_input = bool(include_input)
        self.register_buffer(
            "frequencies", 2.0 ** torch.arange(self.levels, dtype=torch.float32), persistent=False
        )

    @property
    def output_dim(self) -> int:
        return 3 * (int(self.include_input) + 2 * self.levels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        encoded = [value] if self.include_input else []
        phase = value[..., None, :] * self.frequencies[:, None] * math.pi
        encoded.extend([torch.sin(phase).flatten(-2), torch.cos(phase).flatten(-2)])
        return torch.cat(encoded, dim=-1)


class RadianceField(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 96,
        position_levels: int = 6,
        direction_levels: int = 3,
    ):
        super().__init__()
        self.position_encoding = PositionalEncoding(position_levels)
        self.direction_encoding = PositionalEncoding(direction_levels)
        p_dim = self.position_encoding.output_dim
        d_dim = self.direction_encoding.output_dim
        self.trunk = nn.Sequential(
            nn.Linear(p_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.density = nn.Linear(hidden_dim, 1)
        self.feature = nn.Linear(hidden_dim, hidden_dim)
        self.color = nn.Sequential(
            nn.Linear(hidden_dim + d_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.constant_(self.density.bias, -2.0)

    def forward(
        self, points: torch.Tensor, directions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(self.position_encoding(points))
        sigma = F.softplus(self.density(features) - 1.0).squeeze(-1)
        gray = torch.sigmoid(
            self.color(
                torch.cat(
                    [self.feature(features), self.direction_encoding(directions)], dim=-1
                )
            )
        ).squeeze(-1)
        return sigma, gray

    def query_density(self, points: torch.Tensor) -> torch.Tensor:
        features = self.trunk(self.position_encoding(points))
        return F.softplus(self.density(features) - 1.0).squeeze(-1)
