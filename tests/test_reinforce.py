import gymnasium as gym
import torch

from rl_lab.networks.policy import GaussianPolicy
from rl_lab.rollouts.episode import rollout
from rl_lab.training.reinforce import reinforce_update, train


def test_reinforce_update_changes_policy():
    env = gym.make("Pendulum-v1")

    policy = GaussianPolicy(
        observation_dim=3,
        action_dim=1,
    )

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=1e-3,
    )

    episode = rollout(env, policy)

    before = [
        parameter.detach().clone()
        for parameter in policy.parameters()
    ]

    loss = reinforce_update(
        policy=policy,
        optimizer=optimizer,
        episode=episode,
        gamma=0.99,
    )

    after = list(policy.parameters())

    assert isinstance(loss, float)

    changed = any(
        not torch.equal(before_parameter, after_parameter)
        for before_parameter, after_parameter in zip(
            before,
            after,
        )
    )

    assert changed

    env.close()


def test_train():
    env = gym.make("Pendulum-v1")

    policy = GaussianPolicy(
        observation_dim=3,
        action_dim=1,
    )

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=1e-3,
    )

    losses, returns = train(
        env=env,
        policy=policy,
        optimizer=optimizer,
        episodes=2,
        gamma=0.99,
    )

    assert len(losses) == 2
    assert all(isinstance(loss, float) for loss in losses)
    
    assert len(returns) == 2
    assert all(isinstance(return_val, float) for return_val in returns)

    env.close()