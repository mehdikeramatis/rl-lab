"""One-step on-policy actor--critic training for continuous-control environments."""

from dataclasses import dataclass

import torch
from gymnasium import Env
from torch.optim import Optimizer

from rl_lab.losses.policy_gradient import policy_gradient_loss
from rl_lab.networks.policy import GaussianPolicy


@dataclass
class ActorCriticEpisode:
    observations: list[torch.Tensor]
    next_observations: list[torch.Tensor]
    rewards: list[float]
    dones: list[float]
    log_probs: list[torch.Tensor]


def rollout(env: Env, policy: GaussianPolicy) -> ActorCriticEpisode:
    """Collect one episode, retaining next states for TD targets."""
    observation, _ = env.reset()
    observations, next_observations, rewards, dones, log_probs = [], [], [], [], []
    terminated = truncated = False

    while not (terminated or truncated):
        observation_tensor = torch.tensor(observation, dtype=torch.float32)
        output = policy(observation_tensor)
        next_observation, reward, terminated, truncated, _ = env.step(output.action.detach().numpy())
        done = terminated or truncated

        observations.append(observation_tensor)
        next_observations.append(torch.tensor(next_observation, dtype=torch.float32))
        rewards.append(float(reward))
        dones.append(float(done))
        log_probs.append(output.log_prob)
        observation = next_observation

    return ActorCriticEpisode(observations, next_observations, rewards, dones, log_probs)


def actor_critic_update(
    policy: GaussianPolicy,
    value_network: torch.nn.Module,
    policy_optimizer: Optimizer,
    value_optimizer: Optimizer,
    episode: ActorCriticEpisode,
    gamma: float,
) -> tuple[float, float]:
    """Apply an actor update and a critic update using one-step TD errors."""
    observations = torch.stack(episode.observations)
    next_observations = torch.stack(episode.next_observations)
    rewards = torch.tensor(episode.rewards, dtype=torch.float32)
    dones = torch.tensor(episode.dones, dtype=torch.float32)
    values = value_network(observations)

    with torch.no_grad():
        next_values = value_network(next_observations)
        td_targets = rewards + gamma * (1.0 - dones) * next_values

    advantages = td_targets - values
    actor_loss = policy_gradient_loss(torch.stack(episode.log_probs), advantages.detach())
    critic_loss = 0.5 * advantages.pow(2).mean()

    policy_optimizer.zero_grad()
    value_optimizer.zero_grad()
    (actor_loss + critic_loss).backward()
    policy_optimizer.step()
    value_optimizer.step()

    return float(actor_loss.detach()), float(critic_loss.detach())


def train(
    env: Env,
    policy: GaussianPolicy,
    value_network: torch.nn.Module,
    policy_optimizer: Optimizer,
    value_optimizer: Optimizer,
    episodes: int,
    gamma: float,
) -> tuple[list[float], list[float], list[float]]:
    actor_losses, critic_losses, returns = [], [], []

    for _ in range(episodes):
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

    return actor_losses, critic_losses, returns
