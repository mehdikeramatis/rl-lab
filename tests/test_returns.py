import rl_lab.returns.discounted as returns
import torch

def test_returns():
    rewards = [1.0, 2.0, 3.0]
    gamma = 0.9
    expected_returns = torch.tensor([5.23, 4.7, 3.0], dtype=torch.float32)  # Precomputed discounted returns

    computed_returns = returns.discounted_returns(rewards, gamma)

    assert torch.allclose(expected_returns, computed_returns, atol=1e-2)
