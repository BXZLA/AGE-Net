"""Fourier and scalar bond-angle encoders."""

import math

import torch
import torch.nn as nn


class FourierFeatureMapping(nn.Module):
    """Encode radian angles as sin(2*pi*n*theta) and cos(2*pi*n*theta)."""

    def __init__(self, num_frequencies=5):
        super(FourierFeatureMapping, self).__init__()
        self.num_frequencies = int(num_frequencies)
        self.register_buffer(
            "freqs",
            2
            * math.pi
            * torch.arange(1, self.num_frequencies + 1, dtype=torch.float32),
        )

    @property
    def out_dim(self):
        return 2 * self.num_frequencies

    def forward(self, angles):
        angles = angles.unsqueeze(-1)
        features = torch.cat(
            [torch.sin(self.freqs * angles), torch.cos(self.freqs * angles)],
            dim=-1,
        )
        return features


class FourierAngleEncoder(nn.Module):
    """Project Fourier angle features into the edge hidden dimension."""

    def __init__(self, fc_features, num_frequencies=5):
        super(FourierAngleEncoder, self).__init__()
        self.mapping = FourierFeatureMapping(num_frequencies=num_frequencies)
        self.linear = nn.Linear(self.mapping.out_dim, fc_features)
        self.act = nn.SiLU()

    def forward(self, angles):
        return self.act(self.linear(self.mapping(angles)))

    def frequency_weight_norms(self):
        weight = self.linear.weight.detach()
        rows = []
        for idx in range(self.mapping.num_frequencies):
            rows.append(
                {
                    "frequency_idx": idx + 1,
                    "component": "sin",
                    "weight_norm": float(torch.norm(weight[:, idx]).cpu()),
                }
            )
            rows.append(
                {
                    "frequency_idx": idx + 1,
                    "component": "cos",
                    "weight_norm": float(
                        torch.norm(weight[:, idx + self.mapping.num_frequencies]).cpu()
                    ),
                }
            )
        return rows


def angle_embedding(fc_features, num_frequencies=5):
    """Build the Fourier angle encoder used by AGE-Net."""
    return FourierAngleEncoder(fc_features=fc_features, num_frequencies=num_frequencies)


class PlainAngleEncoder(nn.Module):
    """Encode theta/pi for the scalar-angle architecture variant."""

    def __init__(self, fc_features, num_frequencies=5):
        super(PlainAngleEncoder, self).__init__()
        self.num_frequencies = int(num_frequencies)
        self.linear = nn.Linear(1, fc_features)
        self.act = nn.SiLU()

    def forward(self, angles):
        return self.act(self.linear((angles / math.pi).unsqueeze(-1)))

    def frequency_weight_norms(self):
        weight = self.linear.weight.detach()
        return [
            {
                "frequency_idx": 0,
                "component": "plain_theta_over_pi",
                "weight_norm": float(torch.norm(weight[:, 0]).cpu()),
            }
        ]
