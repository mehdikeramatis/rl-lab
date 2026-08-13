from dataclasses import dataclass

import torch


@dataclass
class Episode:
    observations: list[torch.Tensor]
    actions: list[torch.Tensor]
    rewards: list[float]
    log_probs: list[torch.Tensor]