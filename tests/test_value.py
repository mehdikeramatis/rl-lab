import torch

from rl_lab.networks.value import ValueNetwork


def test_value_network():
    network = ValueNetwork(observation_dim=3)

    observation = torch.zeros(3)

    value = network(observation)

    assert value.shape == torch.Size([])
    assert any(parameter.requires_grad for parameter in network.parameters())
    