from rl_lab.networks.policy import GaussianPolicy
from rl_lab.training.reinforce import train
from rl_lab.environments.pendulum import make_pendulum
from rl_lab.evaluation.evaluate import evaluate
import torch
import matplotlib.pyplot as plt



if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)



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
    episodes=20,
)
baseline_mean = sum(baseline_returns) / len(baseline_returns)

print("-" * 100)
print("baseline returns: \n", baseline_returns)
print("\n baseline mean:", baseline_mean)


losses, returns = train(
    env=env,
    policy=policy,
    optimizer=optimizer,
    episodes=200,
    gamma=0.99,
    batch_size=10,
)

trained_returns = evaluate(
    env=env,
    policy=policy,
    episodes=20,
)

trained_mean = sum(trained_returns) / len(trained_returns)

print("-" * 100)
print("trained returns: \n", trained_returns)
print("\n trained mean:", trained_mean)
print("-" * 100)



improvement = trained_mean - baseline_mean
print("improvement:", improvement)
print("-" * 100)

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
