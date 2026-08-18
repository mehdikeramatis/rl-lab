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

baseline_returns = evaluate(
    env=env,
    policy=policy,
    episodes=10,
)
baseline_mean = sum(baseline_returns) / len(baseline_returns)

print("baseline returns:", baseline_returns)
print("baseline mean:", baseline_mean)


losses, returns = train(
    env=env,
    policy=policy,
    optimizer=optimizer,
    episodes=100,
    gamma=0.99,
)

trained_returns = evaluate(
    env=env,
    policy=policy,
    episodes=10,
)

trained_mean = sum(trained_returns) / len(trained_returns)

print("trained returns:", trained_returns)
print("trained mean:", trained_mean)

#print("losses:", losses)
#print("returns:", returns)

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
