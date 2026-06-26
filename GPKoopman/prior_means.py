# GPKoopman/means.py

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MeanFunction(nn.Module):
    """
    Base class for GP prior mean functions.

    Input convention:
        X: (d, N)

    Output convention:
        mean(X): (N,)
    """

    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ZeroMean(MeanFunction):
    """
    Zero prior mean:

        m(x) = 0
    """

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 2:
            raise ValueError(
                f"Expected X to have shape (d, N). Got {tuple(X.shape)}."
            )

        return torch.zeros((X.shape[1], 1), device=X.device, dtype=X.dtype)


class MonomialMean(MeanFunction):
    """
    Fixed monomial prior mean:

        m(x) = prod_{i=1}^{d} [x_i^{powers_i}]

    Example
    -------
    For x = [x1, x2]^T:

        powers=[0, 1]  -> m(x) = x2
        powers=[2, 0]  -> m(x) = x1^2
        powers=[2, 1]  -> m(x) = x1^2*x2

    No learnable parameters are used.
    """

    def __init__(self, powers: Sequence[int], device=None):
        super().__init__()

        if len(powers) == 0:
            raise ValueError("powers must contain at least one exponent.")

        if any(p < 0 for p in powers):
            raise ValueError("MonomialMean expects nonnegative integer powers.")

        powers_tensor = torch.tensor(
            list(powers), dtype=torch.int8, device=device,
        )

        self.register_buffer("powers", powers_tensor)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 2:
            raise ValueError(
                f"Expected X to have shape (d, N). Got {tuple(X.shape)}."
            )

        if X.shape[0] != self.powers.numel():
            raise ValueError(
                "Input dimension and monomial powers do not match. "
                f"X has dimension {X.shape[0]}, "
                f"but powers has length {self.powers.numel()}."
            )

        # [X: (d, N), powers.view(-1, 1): (d, 1)] -> (d, N)
        powered = X ** self.powers.view(-1, 1)

        # product over dimensions -> (N,)
        return torch.prod(powered, dim=0, keepdim=True).T