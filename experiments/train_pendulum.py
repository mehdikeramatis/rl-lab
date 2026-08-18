from rl_lab.networks.policy import GaussianPolicy
from rl_lab.training.reinforce import train
from rl_lab.environments.pendulum import make_pendulum
from rl_lab.evaluation.evaluate import evaluate
import torch
import matplotlib.pyplot as plt


env = make_pendulum()

policy = GaussianPolicy(
    observation_dim=3,
    action_dim=1,
    action_scale=2.0,
)

optimizer = torch.optim.Adam(
    policy.parameters(),
    lr=1e-3,
)

baseline_return = evaluate(
    env=env,
    policy=policy,
)
print("baseline return:", baseline_return)

losses, returns = train(
    env=env,
    policy=policy,
    optimizer=optimizer,
    episodes=100,
    gamma=0.99,
)

print("losses:", losses)
print("returns:", returns)

window = 5

moving_average = [
    sum(returns[i - window + 1 : i + 1]) / window
    for i in range(window - 1, len(returns))
]

plt.plot(returns, label="Return")
plt.plot(
    range(window - 1, len(returns)),
    moving_average,
    label="Moving average",
)

plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("REINFORCE Training Return")
plt.legend()
plt.show()
