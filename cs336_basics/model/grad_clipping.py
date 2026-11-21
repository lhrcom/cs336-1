
def gradient_clipping(parameters, max_l2_norm):
    """
    Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.
        Args:
            parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
            max_l2_norm (float): a positive value containing the maximum l2-norm.

        The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    params = [p for p in parameters if p.grad is not None]
    l2_norm = sum((p.grad ** 2).sum() for p in params) ** 0.5

    if l2_norm > max_l2_norm:
        clip_coef = max_l2_norm / (l2_norm + 10e-6)
        for p in params:
            p.grad.mul_(clip_coef)

       