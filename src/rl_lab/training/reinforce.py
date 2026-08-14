import torch
from torch.optim import Optimizer

from rl_lab.losses.policy_gradient import policy_gradient_loss
from rl_lab.returns.discounted import discounted_returns
from rl_lab.rollouts.episode import Episode, rollout


def reinforce_update(
    policy: torch.nn.Module,
    optimizer: Optimizer,
    episode: Episode,
    gamma: float,
) -> float:
    log_probs = torch.stack(episode.log_probs)

    returns = discounted_returns(
        episode.rewards,
        gamma,
    )

    loss = policy_gradient_loss(
        log_probs,
        returns,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.detach())


def train(
    env,
    policy,
    optimizer,
    episodes: int,
    gamma: float,
) -> list[float]:
    losses = []

    for _ in range(episodes):
        episode = rollout(env, policy)

        loss = reinforce_update(
            policy=policy,
            optimizer=optimizer,
            episode=episode,
            gamma=gamma,
        )

        losses.append(loss)

    return losses