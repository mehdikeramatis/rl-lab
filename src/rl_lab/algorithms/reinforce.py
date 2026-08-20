"""Update rule for the Monte-Carlo REINFORCE algorithm."""

import torch
from torch.optim import Optimizer

from rl_lab.losses.policy_gradient import policy_gradient_loss
from rl_lab.returns.discounted import discounted_returns
from rl_lab.rollouts.episode import Episode


def reinforce_update(
    policy: torch.nn.Module,
    optimizer: Optimizer,
    episode: Episode,
    gamma: float,
) -> float:

    """Update a policy with discounted returns."""
    returns = discounted_returns(episode.rewards, gamma)
    weights = returns
    log_probs = torch.stack(episode.log_probs)
    loss = policy_gradient_loss(log_probs, weights)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()

    return float(loss.detach())


def reinforce_batch_update(
    policy: torch.nn.Module,
    optimizer: Optimizer,
    episodes: list[Episode],
    gamma: float,
) -> float:
    """Update a policy once using a batch of complete episodes."""
    log_probs = []
    returns = []
    for episode in episodes:
        log_probs.extend(episode.log_probs)
        returns.append(discounted_returns(episode.rewards, gamma))

    loss = policy_gradient_loss(
        torch.stack(log_probs),
        torch.cat(returns),
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()

    return float(loss.detach())
