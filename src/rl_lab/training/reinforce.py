from collections.abc import Callable

import torch
from torch.optim import Optimizer

from rl_lab.algorithms.reinforce import reinforce_update
from rl_lab.rollouts.episode import rollout


def train(
    env,
    policy,
    optimizer,
    episodes: int,
    gamma: float,
    on_episode_end: Callable[[int], None] | None = None,
) -> tuple[list[float], list[float]]:
    losses = []
    returns = []

    for episode_index in range(episodes):
        episode = rollout(env, policy)

        loss = reinforce_update(
            policy=policy,
            optimizer=optimizer,
            episode=episode,
            gamma=gamma,
        )

        losses.append(loss)
        returns.append(sum(episode.rewards))

        if on_episode_end is not None:
            on_episode_end(episode_index + 1)
        
    return losses, returns
