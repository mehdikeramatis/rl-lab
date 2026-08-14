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

        output = policy(observation_tensor)

        action_tensor = output.action
        log_prob = output.log_prob

        action = action_tensor.detach().numpy()

        next_observation, reward, terminated, truncated, _ = env.step(action)

        observations.append(observation_tensor)
        actions.append(torch.from_numpy(action))
        rewards.append(float(reward))
        log_probs.append(log_prob)

        observation = next_observation

    return Episode(
        observations=observations,
        actions=actions,
        rewards=rewards,
        log_probs=log_probs,
    )
