"""REINFORCE update rule with a per-episode return baseline."""

import torch
from torch.optim import Optimizer

from rl_lab.losses.policy_gradient import policy_gradient_loss
from rl_lab.returns.discounted import discounted_returns
from rl_lab.rollouts.episode import Episode


def reinforce_baseline_update(
    policy: torch.nn.Module,
    optimizer: Optimizer,
    episode: Episode,
    gamma: float,
) -> float:
    """Update the policy using returns centred by their episode mean."""
    returns = discounted_returns(episode.rewards, gamma)
    advantages = returns - returns.mean()
    advantages = advantages / (advantages.std() + 1e-8)
    log_probs = torch.stack(episode.log_probs)
    loss = policy_gradient_loss(log_probs, advantages)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.detach())
