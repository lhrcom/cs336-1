import torch
from einops import rearrange
def cross_entropy(inputs, targets):
    inputs = rearrange(inputs,"... v -> (...) v")
    targets = rearrange(targets,"... -> (...)")
    z_correct = inputs[torch.arange(inputs.shape[0]),targets]
    loss = torch.logsumexp(inputs, dim=-1) - z_correct
    return loss.mean()