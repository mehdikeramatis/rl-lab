import torch

from rl_lab.rollouts.episode import Episode


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