import torch


def evaluate(
    env,
    policy,
) -> float:
    observation, _ = env.reset()

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

        action = action.clip(
            env.action_space.low,
            env.action_space.high,
        )

        observation, reward, terminated, truncated, _ = env.step(action)

        total_reward += float(reward)

    return total_reward