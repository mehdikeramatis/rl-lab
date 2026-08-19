from collections.abc import Callable

import torch
from gymnasium import Env
from torch.optim import Optimizer

from rl_lab.algorithms.actor_critic import actor_critic_update
from rl_lab.networks.policy import GaussianPolicy
from rl_lab.rollouts.actor_critic import rollout


def train(
    env: Env,
    policy: GaussianPolicy,
    value_network: torch.nn.Module,
    policy_optimizer: Optimizer,
    value_optimizer: Optimizer,
    episodes: int,
    gamma: float,
    on_episode_end: Callable[[int], None] | None = None,
) -> tuple[list[float], list[float], list[float]]:
    actor_losses, critic_losses, returns = [], [], []

    for episode_index in range(episodes):
        episode = rollout(env, policy)
        actor_loss, critic_loss = actor_critic_update(
            policy,
            value_network,
            policy_optimizer,
            value_optimizer,
            episode,
            gamma,
        )
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        returns.append(sum(episode.rewards))
        if on_episode_end is not None:
            on_episode_end(episode_index + 1)

    return actor_losses, critic_losses, returns
