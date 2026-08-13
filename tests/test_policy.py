import torch

from rl_lab.networks.policy import GaussianPolicy


def test_gaussian_policy():
    policy = GaussianPolicy(
        observation_dim=3,
        action_dim=1,
    )

    observation = torch.randn(3)

    distribution = policy(observation)

    action = distribution.sample()
    log_probability = distribution.log_prob(action)

    assert action.shape == (1,)
    assert log_probability.shape == (1,)