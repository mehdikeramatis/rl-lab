import torch

from rl_lab.losses.policy_gradient import (
    policy_gradient_loss,
)


def test_policy_gradient_loss():
    log_probs = torch.tensor(
        [-0.5, -0.7, -0.2]
    )

    returns = torch.tensor(
        [10.0, 5.0, 1.0]
    )

    loss = policy_gradient_loss(
        log_probs,
        returns,
    )

    expected = torch.tensor(8.7)

    assert torch.allclose(
        loss,
        expected,
        atol=1e-5,
    )