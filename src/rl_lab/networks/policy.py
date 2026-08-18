import torch
from torch import nn
from torch.distributions import Normal

from dataclasses import dataclass


@dataclass
class PolicyOutput:
    action: torch.Tensor
    log_prob: torch.Tensor
    

class GaussianPolicy(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        action_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.action_scale = action_scale

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
    ) -> PolicyOutput:
        features = self.network(observation)

        mean = self.mean_head(features)

        std = self.log_std.exp()

        distribution = Normal(mean, std)

        raw_action = distribution.rsample()

        action = squash_action(
            raw_action,
            self.action_scale,
        )

        log_prob = squashed_log_prob(
            distribution,
            raw_action,
            self.action_scale,
        )

        return PolicyOutput(
            action=action,
            log_prob=log_prob,
        )

    def deterministic_action(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        features = self.network(observation)

        mean = self.mean_head(features)

        return squash_action(
            mean,
            self.action_scale,
        )

def squash_action(
    action: torch.Tensor,
    action_scale: float,
) -> torch.Tensor:
    return torch.tanh(action) * action_scale


def squashed_log_prob(
    distribution: torch.distributions.Normal,
    raw_action: torch.Tensor,
    action_scale: float,
) -> torch.Tensor:
    log_prob = distribution.log_prob(raw_action)

    correction = (
        2.0 * (
            torch.log(torch.tensor(2.0))
            - raw_action
            - torch.nn.functional.softplus(-2.0 * raw_action)
        )
    )

    scale_correction = torch.log(
        torch.tensor(action_scale)
    )

    return (
        log_prob
        - correction
        - scale_correction
    ).sum()

