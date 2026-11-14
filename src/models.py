"""Model definitions for the TeamAstro comparison."""
from __future__ import annotations

import torch
from torch import nn


class MLPRegressor(nn.Module):
    def __init__(self, hidden_sizes=(128, 128), dropout=0.2, n_inputs=108, n_outputs=5):
        super().__init__()
        layers = []
        previous = n_inputs
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(dropout)])
            previous = hidden
        layers.append(nn.Linear(previous, n_outputs))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def make_model(name: str):
    if name == "baseline_mlp":
        return MLPRegressor((128, 128), dropout=0.2)
    if name == "deeper_mlp":
        return MLPRegressor((256, 128, 64), dropout=0.2)
    raise ValueError(f"Unknown PyTorch model: {name}")
