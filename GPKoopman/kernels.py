from __future__ import annotations

from typing import Iterable, Sequence, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utility functions
# ============================================================

def _as_tensor(value, *, device=None, dtype=None) -> torch.Tensor:
    """
    Convert scalar / array / tensor inputs to torch.Tensor while preserving
    device and dtype choices when provided.
    """
    if torch.is_tensor(value):
        return value.to(
            device=device if device is not None else value.device,
            dtype=dtype if dtype is not None else value.dtype,
        )

    return torch.tensor(value, device=device, dtype=dtype)


def _inverse_softplus(
        y: torch.Tensor, eps: float = 1e-12, 
        beta : float = 20.0, thresh: float = 1.0) -> torch.Tensor:
    """
    Numerically stable inverse of softplus for y > 0.

    softplus^{-1}(y) = log(exp(y) - 1)
    """
    y = torch.clamp(y, min=eps)

    return torch.where(
        beta * y > thresh,
        y,
        torch.log(torch.expm1(beta * y)) / beta,
    )


def _positive_raw_parameter(
    value,
    *,
    eps: float, beta : float = 20.0, thresh : float = 1.0,
    device=None,
    dtype=None,
) -> nn.Parameter:
    """
    Create a raw trainable parameter whose softplus transform is approximately
    equal to the provided positive initial value.
    """
    value_t = _as_tensor(value, device=device, dtype=dtype)

    # Since the realized parameter is softplus(raw) + eps,
    # initialize raw from value - eps.
    shifted = torch.clamp(value_t - eps, min=eps)

    return nn.Parameter(_inverse_softplus(shifted, beta=beta, thresh=thresh))


def _positive_from_raw(raw: torch.Tensor, eps: float,
                       beta : float = 20.0,
                       thresh : float = 1.0) -> torch.Tensor:
    """
    Transform an unconstrained raw parameter into a strictly positive value.
    """
    return F.softplus(raw, beta=beta, threshold=thresh) + eps


def _validate_pair(
    X1: torch.Tensor,
    X2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Input convention throughout this module:

        X1: (d, N1)
        X2: (d, N2)

    where d is state/input dimension.
    """
    if X2 is None:
        X2 = X1
    else:
        if X1.shape[0] != X2.shape[0]:
            raise ValueError(
                "Feature dimensions must match. "
                f"Got X1.shape[0]={X1.shape[0]} and X2.shape[0]={X2.shape[0]}."
            )
        
    if X1.ndim != 2 or X2.ndim != 2:
        raise ValueError(
            "Expected X1 and X2 to be 2D tensors shaped (d, N). "
            f"Got X1.shape={tuple(X1.shape)}, X2.shape={tuple(X2.shape)}."
        )

    return X2


def _pairwise_sqdist(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
    """
    Squared pairwise Euclidean distances between column samples.
    """
    return torch.cdist(X1.T, X2.T, p=2) ** 2


def _pairwise_dist(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
    """
    Pairwise Euclidean distances between column samples.
    """
    return torch.cdist(X1.T, X2.T, p=2)


def _unique_parameters(
    parameters: Iterable[nn.Parameter],
) -> List[nn.Parameter]:
    """
    Remove duplicates while preserving order.
    Useful when composite kernels may share sub-kernels.
    """
    unique = []
    seen = set()

    for param in parameters:
        if param is None:
            continue

        key = id(param)

        if key not in seen:
            unique.append(param)
            seen.add(key)

    return unique


# ============================================================
# Base kernel classes
# ============================================================

class Kernel(nn.Module):
    """
    Abstract base class for all covariance kernels.

    All kernels follow the convention:

        X1: (d, N1)
        X2: (d, N2)

    and return:

        K:  (N1, N2)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        X1: torch.Tensor,
        X2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def __add__(self, other: "Kernel") -> "SumKernel":
        if not isinstance(other, Kernel):
            return NotImplemented

        return SumKernel([self, other])

    def __mul__(self, other: "Kernel") -> "ProductKernel":
        if not isinstance(other, Kernel):
            return NotImplemented

        return ProductKernel([self, other])

    def optimization_parameters(
        self,
        opt_mu: bool = False,
    ) -> List[nn.Parameter]:
        """
        Parameters to expose to a hyperparameter optimizer.

        Kernels with optional mu-optimization override this method.
        """
        return _unique_parameters(self.parameters())


class TwoPositiveParameterKernel(Kernel):
    """
    Convenience base class for kernels with two positive hyperparameters:

        hp1 > 0
        hp2 > 0

    Internally stores:
        raw_hp1, raw_hp2

    Externally exposes:
        hp1 = softplus(raw_hp1) + eps
        hp2 = softplus(raw_hp2) + eps
    """

    def __init__(
        self,
        hp1: float | torch.Tensor = 1.0,
        hp2: float | torch.Tensor = 1.0,
        *,
        eps: float = 1e-8,
        beta : float = 20.0, thresh : float = 1.0,
        device=None, dtype=None):
        super().__init__()

        self.eps = eps
        self.beta = beta
        self.thresh = thresh

        self.raw_hp1 = _positive_raw_parameter(
            hp1, eps=eps, beta=self.beta, thresh=self.thresh,
            device=device, dtype=dtype)

        self.raw_hp2 = _positive_raw_parameter(
            hp2, eps=eps, beta=self.beta, thresh=self.thresh,
            device=device, dtype=dtype)

    @property
    def hp1(self) -> torch.Tensor:
        return _positive_from_raw(self.raw_hp1, self.eps,
                                  beta=self.beta, thresh=self.thresh)

    @property
    def hp2(self) -> torch.Tensor:
        return _positive_from_raw(self.raw_hp2, self.eps,
                                  beta=self.beta, thresh=self.thresh)

    def optimization_parameters(
        self,
        opt_mu: bool = False,
    ) -> List[nn.Parameter]:
        return [self.raw_hp1, self.raw_hp2]


class TwoParameterKernel(Kernel):
    
    def __init__(self,
                 hp1 : float | torch.Tensor = 1.0,
                 hp2 : float | torch.Tensor = 1.0,
                 *,
                 eps : float = 1e-8,
                 device = None, dtype = None):
        super().__init__()
        self.eps = eps
        self.raw_hp1 = nn.Parameter(_as_tensor(hp1, 
                        device=device, dtype=dtype).clamp(min=self.eps))
        self.raw_hp2 = nn.Parameter(_as_tensor(hp2, 
                        device=device, dtype=dtype).clamp(min=self.eps))
    
    @property
    def hp1(self) -> torch.Tensor:
        return torch.clamp(self.raw_hp1, min=self.eps)

    @property
    def hp2(self) -> torch.Tensor:
        return torch.clamp(self.raw_hp2, min=self.eps)
    
    def optimization_parameters(
            self, opt_mu : bool = False) -> List[nn.Parameter]:
        return [self.raw_hp1, self.raw_hp2]

# ============================================================
# Stationary kernels
# ============================================================

class GaussianKernel(TwoPositiveParameterKernel):
    """
    Gaussian / squared-exponential kernel.

    This preserves the convention from your current implementation:

        K(x, x')
        = hp1^2 * exp(
            - ||x - x'||^2 / (2 hp2^2)
          )

    Here:
        hp1 -> amplitude-like parameter
        hp2 -> lengthscale
    """

    def forward(
        self,
        X1: torch.Tensor,
        X2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        X2 = _validate_pair(X1, X2)

        dists_sq = _pairwise_sqdist(X1, X2)

        return (self.hp1 ** 2) * torch.exp(
            -dists_sq / (2.0 * self.hp2 ** 2)
        )


class ExpSineSqrKernel(TwoPositiveParameterKernel):
    """
    Exponentiated sine-squared periodic kernel:

        K(x, x')
        = hp1^2 * exp(
            -2 sin^2(pi ||x-x'|| / period) / hp2^2
          )

    Here:
        hp1   -> amplitude-like parameter
        hp2   -> periodic smoothness / lengthscale
        period -> periodicity parameter

    period is fixed by default, but can optionally be learned.
    """

    def __init__(
        self,
        hp1: float | torch.Tensor = 1.0,
        hp2: float | torch.Tensor = 1.0,
        *,
        period: float | torch.Tensor = 1.0,
        learn_period: bool = False,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        super().__init__(
            hp1=hp1,
            hp2=hp2,
            eps=eps,
            device=device,
            dtype=dtype,
        )

        self.learn_period = learn_period

        if learn_period:
            self.raw_period = _positive_raw_parameter(
                period,
                eps=eps,
                device=device,
                dtype=dtype,
            )
        else:
            period_t = _as_tensor(period, device=device, dtype=dtype)
            period_t = torch.clamp(period_t, min=eps)
            self.register_buffer("_period", period_t)

    @property
    def period(self) -> torch.Tensor:
        if self.learn_period:
            return _positive_from_raw(self.raw_period, self.eps)

        return self._period

    def forward(
        self,
        X1: torch.Tensor,
        X2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        X2 = _validate_pair(X1, X2)

        dists = _pairwise_dist(X1, X2)

        sine_term = torch.sin(
            torch.pi * dists / self.period
        ) ** 2

        return (self.hp1 ** 2) * torch.exp(
            -2.0 * sine_term / (self.hp2 ** 2)
        )

    def optimization_parameters(
        self,
        opt_mu: bool = False,
    ) -> List[nn.Parameter]:
        params = [self.raw_hp1, self.raw_hp2]

        if self.learn_period:
            params.append(self.raw_period)

        return params
    

class RationalQuadraticKernel(TwoPositiveParameterKernel):
    """
    Rational quadratic kernel:

        K(x, x')
        = hp1^2 *
          [1 + ||x-x'||^2 / (2 alpha hp2^2)]^{-alpha}

    Here:
        hp1  -> amplitude-like parameter
        hp2  -> lengthscale
        alpha -> shape parameter

    alpha is fixed by default, but can optionally be learned.
    """

    def __init__(
        self,
        hp1: float | torch.Tensor = 1.0,
        hp2: float | torch.Tensor = 1.0,
        *,
        alpha: float | torch.Tensor = 1.0,
        learn_alpha: bool = False,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        super().__init__(
            hp1=hp1,
            hp2=hp2,
            eps=eps,
            device=device,
            dtype=dtype,
        )

        self.learn_alpha = learn_alpha

        if learn_alpha:
            self.raw_alpha = _positive_raw_parameter(
                alpha,
                eps=eps,
                device=device,
                dtype=dtype,
            )
        else:
            alpha_t = _as_tensor(alpha, device=device, dtype=dtype)
            alpha_t = torch.clamp(alpha_t, min=eps)
            self.register_buffer("_alpha", alpha_t)

    @property
    def alpha(self) -> torch.Tensor:
        if self.learn_alpha:
            return _positive_from_raw(self.raw_alpha, self.eps)

        return self._alpha

    def forward(
        self,
        X1: torch.Tensor,
        X2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        X2 = _validate_pair(X1, X2)

        dists_sq = _pairwise_sqdist(X1, X2)

        base = 1.0 + dists_sq / (
            2.0 * self.alpha * self.hp2 ** 2
        )

        return (self.hp1 ** 2) * base.pow(-self.alpha)

    def optimization_parameters(
        self,
        opt_mu: bool = False,
    ) -> List[nn.Parameter]:
        params = [self.raw_hp1, self.raw_hp2]

        if self.learn_alpha:
            params.append(self.raw_alpha)

        return params


# ============================================================
# Composite kernels
# ============================================================

class SumKernel(Kernel):
    """
    Additive composition of kernels:

        K = K1 + K2 + ... + Km
    """

    def __init__(self, kernels: Sequence[Kernel]):
        super().__init__()

        if len(kernels) == 0:
            raise ValueError("SumKernel requires at least one child kernel.")

        self.kernels = nn.ModuleList(kernels)

    def forward(
        self,
        X1: torch.Tensor,
        X2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        X2 = _validate_pair(X1, X2)

        result = None

        for kernel in self.kernels:
            Ki = kernel(X1, X2)
            result = Ki if result is None else result + Ki

        return result

    def optimization_parameters(
        self,
        opt_mu: bool = False,
    ) -> List[nn.Parameter]:
        params = []

        for kernel in self.kernels:
            params.extend(
                kernel.optimization_parameters(opt_mu=opt_mu)
            )

        return _unique_parameters(params)


class ProductKernel(Kernel):
    """
    Multiplicative composition of kernels:

        K = K1 * K2 * ... * Km
    """

    def __init__(self, kernels: Sequence[Kernel]):
        super().__init__()

        if len(kernels) == 0:
            raise ValueError("ProductKernel requires at least one child kernel.")

        self.kernels = nn.ModuleList(kernels)

    def forward(
        self,
        X1: torch.Tensor,
        X2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        X2 = _validate_pair(X1, X2)

        result = None

        for kernel in self.kernels:
            Ki = kernel(X1, X2)
            result = Ki if result is None else result * Ki

        return result

    def optimization_parameters(
        self,
        opt_mu: bool = False,
    ) -> List[nn.Parameter]:
        params = []

        for kernel in self.kernels:
            params.extend(
                kernel.optimization_parameters(opt_mu=opt_mu)
            )

        return _unique_parameters(params)
    

# ============================================================
# Registry + compatibility constructor
# ============================================================

KERNEL_CLASSES = {
    "Gaussian": GaussianKernel,
    "ExpSineSqr": ExpSineSqrKernel,
    "RationalQuadratic": RationalQuadraticKernel,
    # "Cosine": CosineKernel,
    # "ThinSpline": ThinSplineKernel,
    # "GibbsExpAttractor": GibbsExpAttractorKernel,
    # "ExplicitAttractor": ExplicitAttractorKernel,
}


def build_kernel(
    kernel_types: Sequence[str] | str,
    hp1_list: Optional[Sequence[float | torch.Tensor]] = None,
    hp2_list: Optional[Sequence[float | torch.Tensor]] = None,
    mu_list: Optional[Sequence[Optional[torch.Tensor]]] = None,
    *,
    combination: str = "sum",
    eps: float = 1e-6,
    device=None,
    dtype=None,
) -> Kernel:
    """
    Compatibility helper that converts your old list-based specification into
    the new kernel-object structure.

    Example
    -------
    kernel = build_kernel(
        kernel_types=["ExplicitAttractor", "Gaussian"],
        hp1_list=[1.0, 1.0],
        hp2_list=[2.0, 0.5],
        mu_list=[mu_tensor, None],
        combination="sum",
        device=device,
        dtype=torch.float32,
    )
    """

    if isinstance(kernel_types, str):
        kernel_types = [kernel_types]

    n_kernels = len(kernel_types)

    if n_kernels == 0:
        raise ValueError("kernel_types must contain at least one kernel name.")

    if hp1_list is None:
        hp1_list = [1.0] * n_kernels

    if hp2_list is None:
        hp2_list = [1.0] * n_kernels

    if mu_list is None:
        mu_list = [None] * n_kernels

    if not (
        len(hp1_list) == len(hp2_list) == len(mu_list) == n_kernels
    ):
        raise ValueError(
            "kernel_types, hp1_list, hp2_list, and mu_list "
            "must have matching lengths."
        )

    kernels: List[Kernel] = []

    for name, hp1, hp2, mu in zip(
        kernel_types,
        hp1_list,
        hp2_list,
        mu_list,
    ):
        if name not in KERNEL_CLASSES:
            raise ValueError(
                f"Unknown kernel type '{name}'. "
                f"Available kernels: {list(KERNEL_CLASSES.keys())}"
            )

        kernel_cls = KERNEL_CLASSES[name]

        if name in {
            "GibbsExpAttractor",
            "ExplicitAttractor",
        }:
            kernel = kernel_cls(
                hp1=hp1,
                hp2=hp2,
                mu=mu,
                eps=eps,
                device=device,
                dtype=dtype,
            )
        else:
            kernel = kernel_cls(
                hp1=hp1,
                hp2=hp2,
                eps=eps,
                device=device,
                dtype=dtype,
            )

        kernels.append(kernel)

    if len(kernels) == 1:
        return kernels[0]

    if combination == "sum":
        return SumKernel(kernels)

    if combination == "product":
        return ProductKernel(kernels)

    raise ValueError(
        "combination must be either 'sum' or 'product'."
    )


if __name__ == "__main__":

    kernel = GaussianKernel(
        hp1=1.0,
        hp2=2.0,
        device='cpu',
        dtype=torch.float32,
    )
    kernel2 = kernel + ExpSineSqrKernel(dtype=torch.float32)

    torch.manual_seed(100)
    Xtrain = torch.sin(torch.linspace(1., 6., 50)).unsqueeze(0) #torch.randn((1,15))

    Kxx = kernel2(Xtrain, Xtrain)

    import numpy as np
    import matplotlib.pyplot as plt
    def MatViz(matrix: torch.Tensor, plot_type: str = 'surf'):
        """
        Visualize a 2D PyTorch tensor as either a 3D surface or a 2D heatmap.

        Args:
            matrix (torch.Tensor): A 2D tensor of shape (rows, cols).
            plot_type (str): 'surf' for a 3D surface plot, 'heat' for a 2D heatmap.

        Raises:
            ValueError: If the input is not 2D or plot_type is invalid.
        """
        # Validate inputs
        if matrix.dim() != 2:
            raise ValueError("Input must be a 2D tensor")
        if plot_type not in ('surf', 'heat'):
            raise ValueError("plot_type must be either 'surf' or 'heat'")

        # Convert to NumPy
        matrix_np = matrix.cpu().detach().numpy()
        vmin, vmax = matrix_np.min(), matrix_np.max()

        if plot_type == 'heat':
            # 2D heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(matrix_np, cmap='viridis',
                        aspect='auto', vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Element Value')
            ax.set_xlabel('Column Index')
            ax.set_ylabel('Row Index')
            ax.set_title(f'Heatmap | Max={vmax:.3g}, Min={vmin:.3g}')
            # plt.show()

        else:
            # 3D surface plot
            x = np.arange(matrix_np.shape[1])
            y = np.arange(matrix_np.shape[0])
            X, Y = np.meshgrid(x, y)

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(
                X, Y, matrix_np, cmap='viridis', edgecolor='k', vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(surf, ax=ax, shrink=0.6)
            cbar.set_label('Element Value')
            ax.set_xlabel('Column Index')
            ax.set_ylabel('Row Index')
            ax.set_zlabel('Value')
            ax.set_title(f'3D Surface Plot | Max={vmax:.3g}, Min={vmin:.3g}')
            # plt.show()
    MatViz(Kxx)
    MatViz(kernel(Xtrain, Xtrain))
    plt.show()    
