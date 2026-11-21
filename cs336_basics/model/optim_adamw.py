from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas":betas, "eps":eps, "weight_decay":weight_decay}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            betas_1, betas_2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] 
                t = state.get("t", 1)
                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data)
                if "v" not in state:
                    state["v"] = torch.zeros_like(p.data)
                m = state.get("m")
                v = state.get("v")
                grad = p.grad.data # Get the gradient of loss with respect to p.
                lr_t = lr * (1-betas_2 ** t) ** 0.5 / (1 - betas_1 ** t)
                p.data = p.data - lr * weight_decay * p.data
                m = betas_1 * m + (1 - betas_1) * grad
                v = betas_2 * v + (1 - betas_2) * (grad ** 2)
                p.data = p.data - lr_t * m / (v ** 0.5 + eps)

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        return loss
