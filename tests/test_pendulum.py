import numpy as np

from rl_lab.environments.pendulum import make_pendulum


def test_pendulum_environment():
    env = make_pendulum()

    observation, info = env.reset()

    # Pendulum-v1 has a 3-dimensional observation.
    assert observation.shape == (3,)
    assert isinstance(info, dict)

    # Sample a valid action from the environment.
    action = env.action_space.sample()

    # Pendulum-v1 has a 1-dimensional continuous action.
    assert action.shape == (1,)

    observation, reward, terminated, truncated, info = env.step(action)

    # Check that the environment returned the expected types/shapes.
    assert observation.shape == (3,)
    assert isinstance(reward, (float, np.floating))
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert isinstance(info, dict)

    env.close()