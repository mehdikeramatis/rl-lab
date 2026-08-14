import torch

from rl_lab.networks.policy import (
    GaussianPolicy,
    squash_action,
    squashed_log_prob,
)

def test_gaussian_policy():
    policy = GaussianPolicy(
        observation_dim=3,
        action_dim=1,
        action_scale=2.0,
    )

    observation = torch.randn(3)

    output = policy(observation)

    assert output.action.shape == (1,)
    assert output.log_prob.ndim == 0
    assert torch.all(output.action >= -2.0)
    assert torch.all(output.action <= 2.0)
    assert torch.isfinite(output.log_prob)


def test_squash_action():
    action = torch.tensor([-10.0, 0.0, 10.0])

    squashed = squash_action(
        action,
        action_scale=2.0,
    )

    assert torch.all(squashed >= -2.0)
    assert torch.all(squashed <= 2.0)


def test_squash_action_is_bounded():
    action = torch.randn(1000)

    squashed = squash_action(
        action,
        action_scale=2.0,
    )

    assert torch.all(squashed > -2.0)
    assert torch.all(squashed < 2.0) 


def test_squashed_log_prob_is_finite():
    distribution = torch.distributions.Normal(
        torch.tensor([0.0]),
        torch.tensor([1.0]),
    )

    raw_action = torch.tensor([0.5])

    log_prob = squashed_log_prob(
        distribution,
        raw_action,
        action_scale=2.0,
    )

    assert torch.isfinite(log_prob)   