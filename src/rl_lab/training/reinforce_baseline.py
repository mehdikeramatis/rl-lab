"""REINFORCE with a per-episode return baseline."""

import torch
from torch.optim import Optimizer

from rl_lab.losses.policy_gradient import policy_gradient_loss
from rl_lab.returns.discounted import discounted_returns
from rl_lab.rollouts.episode import Episode, rollout


def reinforce_baseline_update(
    policy: torch.nn.Module,
    optimizer: Optimizer,
    episode: Episode,
    gamma: float,
) -> float:
    """Update the policy using returns centred by their episode mean.

    The mean return is a state-independent baseline.  It reduces gradient
    variance without changing the expected policy-gradient direction.
    """
    log_probs = torch.stack(episode.log_probs)
    returns = discounted_returns(episode.rewards, gamma)
    advantages = returns - returns.mean()
    loss = policy_gradient_loss(log_probs, advantages)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.detach())


def train(
    env,
    policy: torch.nn.Module,
    optimizer: Optimizer,
    episodes: int,
    gamma: float,
) -> tuple[list[float], list[float]]:
    losses, returns = [], []

    for _ in range(episodes):
        episode = rollout(env, policy)
        losses.append(reinforce_baseline_update(policy, optimizer, episode, gamma))
        returns.append(sum(episode.rewards))

    return losses, returns
