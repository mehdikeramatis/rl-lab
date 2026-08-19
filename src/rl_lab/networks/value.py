import torch
from torch import nn


class ValueNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(observation).squeeze(-1)