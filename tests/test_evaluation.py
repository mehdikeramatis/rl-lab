import gymnasium as gym

from rl_lab.evaluation.evaluate import evaluate
from rl_lab.networks.policy import GaussianPolicy


def test_evaluate():
    env = gym.make("Pendulum-v1")

    policy = GaussianPolicy(
        observation_dim=3,
        action_dim=1,
        action_scale=2.0,
    )

    returns = evaluate(
        env=env,
        policy=policy,
        episodes=10,
    )

    assert len(returns) == 10
    assert all(isinstance(value, float) for value in returns)
    assert all(value < 0.0 for value in returns)
    env.close()


def test_evaluate_is_reproducible_with_reset_seeds():
    env = gym.make("Pendulum-v1")
    policy = GaussianPolicy(
        observation_dim=3,
        action_dim=1,
        action_scale=2.0,
    )

    first_returns = evaluate(env, policy, episodes=2, seeds=[7, 8])
    second_returns = evaluate(env, policy, episodes=2, seeds=[7, 8])

    assert first_returns == second_returns
    env.close()
