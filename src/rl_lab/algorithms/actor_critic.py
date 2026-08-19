"""One-step TD actor--critic update rule."""

import torch
from torch.optim import Optimizer

from rl_lab.losses.policy_gradient import policy_gradient_loss
from rl_lab.networks.policy import GaussianPolicy
from rl_lab.rollouts import episode
from rl_lab.rollouts.actor_critic import ActorCriticEpisode


def actor_critic_update(
    policy: GaussianPolicy,
    value_network: torch.nn.Module,
    policy_optimizer: Optimizer,
    value_optimizer: Optimizer,
    episode: ActorCriticEpisode,
    gamma: float,
) -> tuple[float, float]:
    """Update actor and critic from one-step TD targets."""
    observations = torch.stack(episode.observations)
    next_observations = torch.stack(episode.next_observations)
    rewards = torch.tensor(episode.rewards, dtype=torch.float32)
    dones = torch.tensor(episode.dones, dtype=torch.float32)
    values = value_network(observations)

    with torch.no_grad():
        next_values = value_network(next_observations)
        td_targets = rewards + gamma * (1.0 - dones) * next_values

    advantages = (td_targets - values).detach()
    actor_advantages = (advantages - advantages.mean()) / (
        advantages.std() + 1e-8
    )

    log_probs = torch.stack(episode.log_probs)
    actor_loss = policy_gradient_loss(log_probs, actor_advantages)
    critic_loss = 0.5 * (td_targets - values).pow(2).mean()

    policy_optimizer.zero_grad()
    value_optimizer.zero_grad()
    (actor_loss + critic_loss).backward()
    policy_optimizer.step()
    value_optimizer.step()

    return float(actor_loss.detach()), float(critic_loss.detach())
