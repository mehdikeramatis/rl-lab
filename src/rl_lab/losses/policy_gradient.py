
import torch

def policy_gradient_loss(
        logprobs: torch.Tensor, 
        discounted_returns: torch.Tensor
        )-> torch.Tensor:

    
    # Multiply by the advantages to get the weighted loss
    weighted_loss = - logprobs * discounted_returns
    
    # Return the mean loss
    return torch.sum(weighted_loss)