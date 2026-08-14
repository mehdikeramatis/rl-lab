import gymnasium as gym
import torch
from wandb import env

from rl_lab.networks.policy import GaussianPolicy
from rl_lab.rollouts import episode
from rl_lab.rollouts.episode import Episode, rollout


def test_episode():
    observations = [torch.randn(3)]
    actions = [torch.randn(1)]
    rewards = [1.0]
    log_probs = [torch.randn(1)]

    episode = Episode(
        observations=observations,
        actions=actions,
        rewards=rewards,
        log_probs=log_probs,
    )

    assert episode.observations == observations
    assert episode.actions == actions
    assert episode.rewards == rewards
    assert episode.log_probs == log_probs


def test_rollout():
    env = gym.make("Pendulum-v1")

    policy = GaussianPolicy(
        observation_dim=3,
        action_dim=1,
    )

    episode = rollout(env, policy)

    assert isinstance(episode, Episode)
    assert len(episode.observations) > 0
    assert len(episode.actions) == len(episode.observations)
    assert len(episode.rewards) == len(episode.observations)
    assert len(episode.log_probs) == len(episode.observations)

    for action in episode.actions:
        assert torch.all(action >= -2.0)
        assert torch.all(action <= 2.0)

    env.close()