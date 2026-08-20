"""Episode data and collection for one-step actor--critic."""

from dataclasses import dataclass

import torch
from gymnasium import Env

from rl_lab.networks.policy import GaussianPolicy


@dataclass
class ActorCriticEpisode:
    observations: list[torch.Tensor]
    next_observations: list[torch.Tensor]
    rewards: list[float]
    terminateds: list[float]
    log_probs: list[torch.Tensor]


def rollout(env: Env, policy: GaussianPolicy) -> ActorCriticEpisode:
    """Collect one episode, retaining next states for TD targets."""
    observation, _ = env.reset()
    observations, next_observations, rewards, terminateds, log_probs = [], [], [], [], []
    terminated = truncated = False

    while not (terminated or truncated):
        observation_tensor = torch.tensor(observation, dtype=torch.float32)
        output = policy(observation_tensor)
        next_observation, reward, terminated, truncated, _ = env.step(output.action.detach().numpy())

        observations.append(observation_tensor)
        next_observations.append(torch.tensor(next_observation, dtype=torch.float32))
        rewards.append(float(reward))
        terminateds.append(float(terminated))
        log_probs.append(output.log_prob)
        observation = next_observation

    return ActorCriticEpisode(observations, next_observations, rewards, terminateds, log_probs)
