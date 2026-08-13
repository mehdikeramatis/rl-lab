from dataclasses import dataclass

import numpy as np
import torch
from gymnasium import Env

from rl_lab.networks.policy import GaussianPolicy


@dataclass
class Episode:
    observations: list[torch.Tensor]
    actions: list[torch.Tensor]
    rewards: list[float]
    log_probs: list[torch.Tensor]


def rollout(
    env: Env,
    policy: GaussianPolicy,
) -> Episode:
    observations = []
    actions = []
    rewards = []
    log_probs = []

    observation, _ = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        observation_tensor = torch.tensor(
            observation,
            dtype=torch.float32,
        )

        distribution = policy(observation_tensor)

        action_tensor = distribution.sample()

        log_prob = distribution.log_prob(action_tensor).sum()

        action = action_tensor.detach().numpy()

        action = np.clip(
            action,
            env.action_space.low,
            env.action_space.high,
        )

        next_observation, reward, terminated, truncated, _ = env.step(action)

        observations.append(observation_tensor)
        actions.append(action_tensor)
        rewards.append(float(reward))
        log_probs.append(log_prob)

        observation = next_observation

    return Episode(
        observations=observations,
        actions=actions,
        rewards=rewards,
        log_probs=log_probs,
    )