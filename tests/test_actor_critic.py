import gymnasium as gym
import torch

from rl_lab.networks.policy import GaussianPolicy
from rl_lab.networks.value import ValueNetwork
from rl_lab.algorithms.actor_critic import actor_critic_update
from rl_lab.rollouts.actor_critic import rollout


def test_actor_critic_update_changes_actor_and_critic():
    env = gym.make("Pendulum-v1")
    policy = GaussianPolicy(observation_dim=3, action_dim=1)
    value_network = ValueNetwork(observation_dim=3)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    value_optimizer = torch.optim.Adam(value_network.parameters(), lr=1e-3)
    episode = rollout(env, policy)
    actor_before = [parameter.detach().clone() for parameter in policy.parameters()]
    critic_before = [parameter.detach().clone() for parameter in value_network.parameters()]

    actor_loss, critic_loss = actor_critic_update(
        policy,
        value_network,
        policy_optimizer,
        value_optimizer,
        episode,
        gamma=0.99,
    )

    assert isinstance(actor_loss, float)
    assert isinstance(critic_loss, float)
    assert any(not torch.equal(previous, current) for previous, current in zip(actor_before, policy.parameters()))
    assert any(not torch.equal(previous, current) for previous, current in zip(critic_before, value_network.parameters()))
    env.close()
