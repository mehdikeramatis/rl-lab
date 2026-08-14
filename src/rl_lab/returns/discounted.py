import torch


def discounted_returns(
    rewards: list[float],
    gamma: float,
) -> torch.Tensor:
    returns = [0.0] * len(rewards)

    running_return = 0.0

    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    return torch.tensor(returns, dtype=torch.float32)