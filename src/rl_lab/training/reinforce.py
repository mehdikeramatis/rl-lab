from collections.abc import Callable

import torch
from torch.optim import Optimizer

from rl_lab.algorithms.reinforce import reinforce_batch_update
from rl_lab.rollouts.episode import rollout


def train(
    env,
    policy,
    optimizer,
    episodes: int,
    gamma: float,
    batch_size: int = 10,
    on_episode_end: Callable[[int], None] | None = None,
) -> tuple[list[float], list[float]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    losses = []
    returns = []
    batch = []

    for episode_index in range(episodes):
        episode = rollout(env, policy)
        batch.append(episode)
        returns.append(sum(episode.rewards))

        if len(batch) == batch_size or episode_index == episodes - 1:
            loss = reinforce_batch_update(policy, optimizer, batch, gamma)
            losses.extend([loss] * len(batch))
            batch.clear()

        if on_episode_end is not None:
            on_episode_end(episode_index + 1)

    return losses, returns
