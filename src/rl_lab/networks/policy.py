import torch
from torch import nn
from torch.distributions import Normal


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.mean_head = nn.Linear(hidden_dim, action_dim)

        self.log_std = nn.Parameter(
            torch.zeros(action_dim)
        )

    def forward(
        self,
        observation: torch.Tensor,
    ) -> Normal:
        features = self.network(observation)

        mean = self.mean_head(features)

        std = self.log_std.exp()

        return Normal(mean, std)