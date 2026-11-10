"""The single network architecture shared by all three approaches."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from fedheart.utils import derive_seed


class HeartMLP(nn.Module):
    """Input layer sized to the transformed features -> 16 ReLU units -> one logit."""

    def __init__(self, in_features: int, hidden: int = 16) -> None:
        super().__init__()
        self.hidden = nn.Linear(in_features, hidden)
        self.act = nn.ReLU()
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.act(self.hidden(x))).squeeze(-1)


def initial_state(in_features: int, hidden: int, seed: int) -> OrderedDict[str, torch.Tensor]:
    """PyTorch-default initialization drawn from a seed-specific stream.

    Saved once per seed and loaded by the centralized and FedAvg runs (and, through the
    FedAvg checkpoint, by fine-tuning), so initial weights are matched within a seed.
    """
    gen_state = torch.random.get_rng_state()
    torch.manual_seed(derive_seed(seed, "init"))
    state = HeartMLP(in_features, hidden).state_dict()
    torch.random.set_rng_state(gen_state)
    return OrderedDict((k, v.detach().clone()) for k, v in state.items())


def build_model(state: dict[str, torch.Tensor], in_features: int, hidden: int) -> HeartMLP:
    model = HeartMLP(in_features, hidden)
    model.load_state_dict(state)
    return model


def clone_state(state: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((k, v.detach().clone()) for k, v in state.items())


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
