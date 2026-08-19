from collections.abc import Sequence

import torch


def evaluate(
    env,
    policy,
    episodes: int = 10,
    seeds: Sequence[int] | None = None,
) -> list[float]:
    """Evaluate a policy using its deterministic action for each state."""
    if seeds is not None and len(seeds) != episodes:
        raise ValueError("seeds must contain one seed per evaluation episode")

    returns = []

    for episode in range(episodes):
        seed = None if seeds is None else seeds[episode]
        observation, _ = env.reset(seed=seed)

        terminated = False
        truncated = False

        total_reward = 0.0

        while not (terminated or truncated):
            observation_tensor = torch.tensor(
                observation,
                dtype=torch.float32,
            )

            action_tensor = policy.deterministic_action(
                observation_tensor)

            action = action_tensor.detach().numpy()

            observation, reward, terminated, truncated, _ = env.step(action)

            total_reward += float(reward)

        returns.append(total_reward)

    return returns
