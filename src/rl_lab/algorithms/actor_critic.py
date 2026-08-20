"""One-step TD actor--critic update rule."""

import torch
from torch.optim import Optimizer

from rl_lab.losses.policy_gradient import policy_gradient_loss
from rl_lab.networks.policy import GaussianPolicy
from rl_lab.rollouts.actor_critic import ActorCriticEpisode


def actor_critic_update(
    policy: GaussianPolicy,
    value_network: torch.nn.Module,
    policy_optimizer: Optimizer,
    value_optimizer: Optimizer,
    episode: ActorCriticEpisode,
    gamma: float,
) -> tuple[float, float]:
    """Update actor and critic from one episode of one-step TD targets."""
    return actor_critic_batch_update(
        policy,
        value_network,
        policy_optimizer,
        value_optimizer,
        [episode],
        gamma,
    )


def actor_critic_batch_update(
    policy: GaussianPolicy,
    value_network: torch.nn.Module,
    policy_optimizer: Optimizer,
    value_optimizer: Optimizer,
    episodes: list[ActorCriticEpisode],
    gamma: float,
) -> tuple[float, float]:
    """Update actor and critic once from a batch of complete episodes."""
    observations = torch.cat([torch.stack(item.observations) for item in episodes])
    next_observations = torch.cat(
        [torch.stack(item.next_observations) for item in episodes]
    )
    rewards = torch.cat(
        [torch.tensor(item.rewards, dtype=torch.float32) for item in episodes]
    )
    terminateds = torch.cat(
        [torch.tensor(item.terminateds, dtype=torch.float32) for item in episodes]
    )
    values = value_network(observations)

    with torch.no_grad():
        next_values = value_network(next_observations)
        td_targets = rewards + gamma * (1.0 - terminateds) * next_values

    advantages = (td_targets - values).detach()
    actor_advantages = advantages

    log_probs = torch.cat([torch.stack(item.log_probs) for item in episodes])
    actor_loss = policy_gradient_loss(log_probs, actor_advantages)
    critic_loss = 0.5 * (td_targets - values).pow(2).mean()

    policy_optimizer.zero_grad()
    value_optimizer.zero_grad()
    (actor_loss + critic_loss).backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(value_network.parameters(), max_norm=1.0)
    policy_optimizer.step()
    value_optimizer.step()

    return float(actor_loss.detach()), float(critic_loss.detach())
