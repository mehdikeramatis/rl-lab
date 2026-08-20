from collections.abc import Callable

import torch
from gymnasium import Env
from torch.optim import Optimizer

from rl_lab.algorithms.actor_critic import actor_critic_batch_update
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
    *,
    batch_size: int = 10,
    on_episode_end: Callable[[int], None] | None = None,
) -> tuple[list[float], list[float], list[float]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    actor_losses, critic_losses, returns = [], [], []
    batch = []

    for episode_index in range(episodes):

        episode = rollout(env, policy)
        batch.append(episode)
        returns.append(sum(episode.rewards))

        if len(batch) == batch_size or episode_index == episodes - 1:
            actor_loss, critic_loss = actor_critic_batch_update(
                policy,
                value_network,
                policy_optimizer,
                value_optimizer,
                batch,
                gamma,
            )
            actor_losses.extend([actor_loss] * len(batch))
            critic_losses.extend([critic_loss] * len(batch))
            batch.clear()

        if on_episode_end is not None:
            on_episode_end(episode_index + 1)

    return actor_losses, critic_losses, returns
