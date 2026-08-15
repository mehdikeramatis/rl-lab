from rl_lab.networks.policy import GaussianPolicy
from rl_lab.training.reinforce import train
from rl_lab.environments.pendulum import make_pendulum
import torch


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

losses = train(
    env=env,
    policy=policy,
    optimizer=optimizer,
    episodes=10,
    gamma=0.99,
)

print(losses)

