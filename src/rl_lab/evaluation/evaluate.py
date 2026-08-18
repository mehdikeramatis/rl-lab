import torch



def evaluate(
    env,
    policy,
    episodes: int = 10,
) -> list[float]:


    returns = []

    for _ in range(episodes):
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

            observation, reward, terminated, truncated, _ = env.step(action)

            total_reward += float(reward)

        returns.append(total_reward)

    return returns