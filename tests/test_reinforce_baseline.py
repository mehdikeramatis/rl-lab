import gymnasium as gym
import torch

from rl_lab.networks.policy import GaussianPolicy
from rl_lab.algorithms.reinforce_baseline import reinforce_baseline_update
from rl_lab.rollouts.episode import rollout


def test_reinforce_baseline_update_changes_policy():
    env = gym.make("Pendulum-v1")
    policy = GaussianPolicy(observation_dim=3, action_dim=1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    episode = rollout(env, policy)
    before = [parameter.detach().clone() for parameter in policy.parameters()]

    loss = reinforce_baseline_update(policy, optimizer, episode, gamma=0.99)

    assert isinstance(loss, float)
    assert any(not torch.equal(previous, current) for previous, current in zip(before, policy.parameters()))
    env.close()
