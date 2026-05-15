from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt
import numpy as np
import warnings

from .kernels import Kernel, TwoPositiveParameterKernel, TwoParameterKernel
from .prior_means import MeanFunction, ZeroMean

class GPObservable(nn.Module):

    def __init__(self, d: int, Ns: int, 
                 kernel: Kernel, 
                 prior_mean: MeanFunction | None = None, 
                 noise: float = 1e-4, 
                 dtype : torch.dtype = torch.float32, 
                 device: str | torch.device = "cuda:0",
                 eps: float = 1e-8,
                 beta : float = 50.,
                 thresh : float = 1.0,):
        """
        Gaussian Process Observable with kernel and prior mean objects.

        Args:
            d (int): Dimensionality of the inputs.
            Ns (int): Number of training samples.
            kernel (kernels.Kernel): Covariance Kernel for the GP.
            prior_mean (prior_means.MeanFunction): 
                Fixed prior mean/basis function.
                If None is specified, ZeroMean is used
            noise (float) : GP Noise assumption
            dtype (torch.dtype) : 
                FP datatype for trainable parameters.
                Inputs and outputs are also in dtype.
            device (str OR torch.device) : Device on which object lives.
            eps (float) : Positive floor used for GP-Noise transform

        """
        super().__init__()

        # Core GP metadata ---------------------------------
        # --------------------------------------------------
        self.d = d
        self.Ns = Ns
        self.eps = eps
        self.dtype = dtype
        self.device = torch.device(device)
        self.beta = beta
        self.thresh = thresh

        # Kernel Object ------------------------------------
        if not isinstance(kernel, Kernel):
            raise TypeError(
                f"kernel must be an instance of Kernel. "
                f"Received type: {type(kernel)}")

        self.kernel = kernel.to(device=self.device, dtype=self.dtype)
        self.kernel.beta = self.beta
        self.kernel.thresh = self.thresh

        # Prior mean function ------------------------------
        if prior_mean is None:
            prior_mean = ZeroMean()

        if not isinstance(prior_mean, MeanFunction):
            raise TypeError(
                f"prior_mean must be None or an instance of MeanFunction. "
                f"Received type: {type(prior_mean)}")

        self.prior_mean = prior_mean.to(device=self.device)

        # Observation Noise --------------------------------
        #   - We store an unconstrained raw parameter and expose
        #     positive noise via noise property:
        #       noise = softplus(raw_noise) + eps
        # --------------------------------------------------
        noise_tensor = torch.as_tensor(
            noise, device=self.device, dtype=self.dtype,)

        if torch.any(noise_tensor <= 0):
            raise ValueError(f"noise must be strictly positive. Received {noise}.")

        self.raw_noise = nn.Parameter(
            self._inverse_softplus(
                torch.clamp(noise_tensor - self.eps, min=self.eps),
                beta=self.beta, thresh=self.thresh))

        # Training data placeholders -----------------------
        self.Xtrain = None
        self.ytrain = None

        # Cached kernel / GP matrices ----------------------
        self.invKxx = None
        self.alpha = None

    @staticmethod
    def _inverse_softplus(x: torch.Tensor,
                          beta : float = 10.0,
                          thresh : float = 1.0) -> torch.Tensor:
        """
        Numerically stable inverse softplus for x > 0.
        """
        return torch.where(
            beta * x > thresh,
            x,
            torch.log(torch.expm1(beta * x)) / beta
        )

    @property
    def noise(self) -> torch.Tensor:
        """
        Positive observation-noise standard deviation.
        """
        return F.softplus(self.raw_noise, beta=self.beta, threshold=self.thresh) + self.eps

    def set_hyperparameters(self, hp1_list=None, hp2_list=None):
        raise NotImplementedError
        # if hp1_list is not None:
        #     self.hp1_list = nn.ParameterList(
        #         [param if isinstance(param, nn.Parameter)
        #          else nn.Parameter(param.clone().detach() if isinstance(param, torch.Tensor)
        #                            else torch.tensor(param, device=self.device))
        #          for param in hp1_list]
        #     )

        # if hp2_list is not None:
        #     self.hp2_list = nn.ParameterList(
        #         [param if isinstance(param, nn.Parameter)
        #          else nn.Parameter(param.clone().detach() if isinstance(param, torch.Tensor)
        #                            else torch.tensor(param, device=self.device))
        #          for param in hp2_list]
        #     )

    def get_parameters(self) -> dict:
        """
        Returns a dictionary containing the current hyperparameters (hp1_list, hp2_list), noise,
        and attractor locations (mu_list) for this GPObservable.
        """
        raise NotImplementedError
        # params = {
        #     "hp1_list": [hp.detach() for hp in self.hp1_list],
        #     "hp2_list": [hp.detach() for hp in self.hp2_list],
        #     "noise": self.noise.detach(),
        #     "mu_list": [mu.detach() for mu in self.mu_list]
        # }
        # return params

    def trainGP(self, Xtrain : torch.Tensor, ytrain : torch.Tensor):
        if (Xtrain.shape[0] != self.d) or (Xtrain.shape[1] != self.Ns):
            raise ValueError(f'Xtrain must be of shape {tuple(self.d, self.Ns)}. '
                             f'Recieved Xtrain of shape {tuple(Xtrain.shape)}')
        if (ytrain.shape[0] != self.Ns) or (ytrain.shape[1] != 1):
            raise ValueError(f'ytrain must be of shape {tuple(self.Ns, 1)}. '
                             f'Recieved ytrain of shape {tuple(ytrain.shape)}')

        self.Xtrain = Xtrain.to(dtype=self.dtype, device=self.device)
        self.ytrain = ytrain.to(dtype=self.dtype, device=self.device)

        Kxx = self.kernel(self.Xtrain, self.Xtrain)

        try:
            L = torch.linalg.cholesky(Kxx + 
                    (self.noise ** 2) * torch.eye(self.Ns, dtype=self.dtype, device=self.device))
            self.invKxx = torch.cholesky_inverse(L)
        except RuntimeError:
            self.invKxx = torch.linalg.pinv((Kxx + 
                                (self.noise ** 2) * torch.eye(self.Ns, dtype=self.dtype, device=self.device)), 
                                hermitian=True)
            warnings.warn(f"Cholesky failed in trainGP. Used linalg.pinv with hermitian=True", RuntimeWarning)

        self.alpha = self.invKxx @ (self.ytrain - self.prior_mean(self.Xtrain))

    def predictMean(self, Xq : torch.Tensor) -> torch.Tensor:
        Xq = Xq.to(dtype=self.dtype, device=self.device)

        Kqx = self.kernel(Xq, self.Xtrain)
        return self.prior_mean(Xq) + Kqx @ self.alpha

    def predictCov(self, Xq : torch.Tensor) -> torch.Tensor:
        Xq = Xq.to(device=self.device, dtype=self.dtype)

        Kqx = self.kernel(Xq, self.Xtrain)
        Kqq = self.kernel(Xq, Xq)

        return Kqq - (Kqx @ self.invKxx @ (Kqx.mT))

    def predictGP(self, Xq : torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        Xq = Xq.to(self.device)

        mean = self.predictMean(Xq)
        cov = self.predictCov(Xq)

        return mean, cov
    
    def forward(self, Xq : torch.Tensor, ytrain : torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fully differentiable forward pass that computes predictions using the current hyperparameters.
        """
        Xq = Xq.to(device=self.device, dtype=self.dtype)
        ytrain = ytrain.to(device=self.device, dtype=self.dtype)

        Kxx = self.kernel(self.Xtrain, self.Xtrain)
        Kqx = self.kernel(Xq, self.Xq)
        Kqq = self.kernel(Xq, Xq)

        try:
            L = torch.linalg.cholesky(Kxx + 
                    (self.noise ** 2) * torch.eye(self.Ns, dtype=self.dtype, device=self.device))
            invKxx = torch.cholesky_inverse(L)
        except RuntimeError:
            invKxx = torch.linalg.pinv((Kxx + 
                        (self.noise ** 2) * torch.eye(self.Ns, dtype=self.dtype, device=self.device)), 
                        hermitian=True)
            warnings.warn(f"Cholesky failed in forward. Used linalg.pinv with hermitian=True", RuntimeWarning)

        alpha = invKxx @ ( ytrain - self.prior_mean(self.Xtrain) )
        mean = self.prior_mean(Xq) + Kqx @ alpha
        cov = Kqq - (Kqx @ invKxx @ (Kqx.mT))

        return mean, cov

    def forward_mean(self, Xq : torch.Tensor, ytrain : torch.Tensor) -> torch.Tensor:
        """
        Fully differentiable forward pass that computes predictive mean using current hyperparameters
        """
        Xq = Xq.to(device=self.device, dtype=self.dtype)
        ytrain = ytrain.to(device=self.device, dtype=self.dtype)

        Kxx = self.kernel(self.Xtrain, self.Xtrain)
        Kqx = self.kernel(Xq, self.Xtrain)

        try:
            L = torch.linalg.cholesky(Kxx + 
                    (self.noise ** 2) * torch.eye(self.Ns, dtype=self.dtype, device=self.device))
            invKxx = torch.cholesky_inverse(L)
        except RuntimeError:
            invKxx = torch.linalg.pinv((Kxx + 
                        (self.noise ** 2) * torch.eye(self.Ns, dtype=self.dtype, device=self.device)), 
                        hermitian=True)
            warnings.warn(f"Cholesky failed in forward_mean. Used linalg.pinv with hermitian=True", RuntimeWarning)

        alpha = invKxx @ ( ytrain - self.prior_mean(self.Xtrain) )

        return self.prior_mean(Xq) + Kqx @ alpha

    def forward_cov(self, Xq : torch.Tensor) -> torch.Tensor:
        """
        Fully differentiable forward pass that computes predictive covariance using current hyperparameters
        """
        Xq = Xq.to(device=self.device, dtype=self.dtype)
        
        Kxx = self.kernel(self.Xtrain, self.Xtrain)
        Kqx = self.kernel(Xq, self.Xtrain)
        Kqq = self.kernel(Xq, Xq)

        try:
            L = torch.linalg.cholesky(Kxx + 
                    (self.noise ** 2) * torch.eye(self.Ns, dtype=self.dtype, device=self.device))
            invKxx = torch.cholesky_inverse(L)
        except RuntimeError:
            invKxx = torch.linalg.pinv((Kxx + 
                        (self.noise ** 2) * torch.eye(self.Ns, dtype=self.self.dtype, device=self.device)), 
                        hermitian=True)
            warnings.warn(f"Cholesky failed in forward_cov. Used linalg.pinv with hermitian=True", RuntimeWarning)

        return (Kqq - (Kqx @ invKxx @ (Kqx.mT)))

    def forward_G(self, Xq: torch.Tensor) -> torch.Tensor:
        """
        Fully differentiable forward pass that computes the kernel covariance matrix at given query point(s)
        """
        Xq = Xq.to(device=self.device, dtype=self.dtype)    # (d, Nq)

        Kxx = self.kernel(self.Xtrain, self.Xtrain) # (Ns, Ns)
        Kqx = self.kernel(Xq, self.Xtrain)  # (Nq, Ns)

        try:
            L = torch.linalg.cholesky(Kxx + 
                    (self.noise ** 2) * torch.eye(self.Ns, dtype=self.dtype, device=self.device))
            invKxx = torch.cholesky_inverse(L)
        except RuntimeError:
            invKxx = torch.linalg.pinv((Kxx + 
                        (self.noise ** 2) * torch.eye(self.Ns, dtype=self.dtype, device=self.device)), 
                        hermitian=True)
            warnings.warn(f"Cholesky failed in forward_G. Used linalg.pinv with hermitian=True", RuntimeWarning)

        G = Kqx @ invKxx    # (Nq, Ns)
        return G

    def optimize_hyperparameters(
            self, num_iter : int = 100, 
            lr : float = 0.01, opt_noise : bool = False):
        """
        Optimize GPO kernel hyperparameters by maximizing log marginal likelihood

        Args
            num_iter
            lr
            opt_noise
        
        Notes
        -----
            - Kernel parameters are optimized through self.kernel.optimization_parameters().
            - The prior mean function is incorporated through the residual:
                y_centered = y - m(Xtrain)
            - At the end, self.trainGP(self.Xtrain, self.y) is called to refresh
            all cached kernel matrices and posterior quantities.
        """

        if self.Xtrain is None or self.ytrain is None:
            raise ValueError(
                "Training data not found. Please call trainGP first."
            )

        X = self.Xtrain # (d, Ns)
        y = self.ytrain # (Ns, 1)

        # --------------------------------------------------
        # Prior mean evaluated on training inputs
        residual = y - self.prior_mean(self.Xtrain) # (Ns, 1)

        # Collect optimization parameters
        params = self.kernel.optimization_parameters()  # returns [nn.Parameter, nn.Parameter]
        original_noise_requires_grad = self.raw_noise.requires_grad

        if not opt_noise:
            self.raw_noise.requires_grad_(False)
        else:
            params.append(self.raw_noise)

        # Remove duplicate parameter references, if any
        unique_params = []
        seen = set()
        for p in params:
            if id(p) not in seen:
                unique_params.append(p)
                seen.add(id(p))

        if len(unique_params) == 0:
            warnings.warn(
                "No hyperparameters were found for optimization.",
                RuntimeWarning,
            )
            self.trainGP(self.Xtrain, self.ytrain)
            return

        optimizer = torch.optim.Adam(unique_params, lr=lr)

        # Precompute identity matrix and constants
        eye = torch.eye(self.Ns, dtype=X.dtype, device=X.device)

        log_2pi = X.new_tensor(math.log(2.0 * math.pi))

        # Numerical jitter used only when Cholesky needs help
        jitter = 1e-8   # if X.dtype == torch.float64 else 1e-6

        # --------------------------------------------------
        # Negative Log Marginal-likelihood optimization loop
        # --------------------------------------------------
        for _ in range(num_iter):
            optimizer.zero_grad(set_to_none=True)

            Kxx = self.kernel(X, X)
            # Symmetrize against tiny floating-point asymmetries
            Kxx = 0.5 * (Kxx + Kxx.T)
            K_til = Kxx + (self.noise ** 2) * eye

            # Cholesky Attempts with increasing Jitter
            L = None
            for attempt in range(5):
                try:
                    L = torch.linalg.cholesky(K_til)
                    break

                except RuntimeError:
                    L = None
                    jitter = jitter * (10.0 ** attempt)
                    K_til += jitter * eye

            # Compute Cost using Cholesky Factor
            if L is not None:
                # alpha = K^{-1}(y - m(X))
                alpha = torch.cholesky_solve(residual, L)

                # Quadratic term
                quad_term = (residual.T @ alpha).squeeze()

                # Stable log determinant from Cholesky factor
                log_det = 2.0 * torch.log(torch.diagonal(L)).sum()
            # Cost Fallback if Cholesky Decomposition fails
            else:
                warnings.warn('Cholesky Decomposition failed for 5 attempts.'
                              'Using pinv with hermitian=True', RuntimeWarning)
                # Final fallback if Cholesky repeatedly fails
                # jitter = base_jitter * 1e4
                K_used = K_til + jitter * eye

                K_pinv = torch.linalg.pinv(K_used, hermitian=True)
                quad_term = (residual.T @ K_pinv @ residual).squeeze()

                sign, log_det = torch.linalg.slogdet(K_used)

                if torch.any(sign <= 0):
                    warnings.warn(
                        "Covariance matrix had non-positive determinant during "
                        "hyperparameter optimization. Using log|det(K)| fallback.",
                        RuntimeWarning)

            # Negative log marginal likelihood
            loss = -0.5 * ( quad_term + log_det + (self.Ns * log_2pi) )

            loss.backward()
            optimizer.step()

        # Clear gradients and refresh GP matrix cache
        optimizer.zero_grad(set_to_none=True)
        self.raw_noise.requires_grad_(original_noise_requires_grad)
        self.trainGP(self.Xtrain, self.ytrain)



class GPObservablesManager:
    def __init__(self):
        self.observables = {}

    @property
    def num_obs(self) -> int:
        return len(self.observables)

    def add_observable(
            self, 
            index : int, d : int, Ns : int, kernel : Kernel, 
            prior_mean : MeanFunction | None = None, 
            noise : float = 1e-4, 
            dtype : torch.dtype = torch.float32, 
            device : str | torch.device = 'cuda:0',
            eps : float = 1e-8,
            beta : float = 50.,
            thresh : float = 1.0):
        
        if index in self.observables:
            raise ValueError(f'Observable with index {index} already exists.')
        
        self.observables[index] = GPObservable(d, Ns, kernel, prior_mean, noise, dtype, device, eps, beta, thresh)

    def get_params(self, index):
        """
        Returns a dictionary of the current parameters for the observable with the given index.
        """
        if index not in self.observables:
            raise ValueError(f'Observable with index {index} does not exist.')
        return self.observables[index].get_parameters()

    def set_parameters(self, hp1_list=None, hp2_list=None, noise_list=None, mu_list=None):
        raise NotImplementedError
        # i = 0
        # for obs in self.observables.values():
        #     num_kernels = len(obs.kernel_types)

        #     if hp1_list is not None:
        #         for k in range(num_kernels):
        #             obs.hp1_list[k].data = hp1_list[i + k].data.to(obs.device)
        #     if hp2_list is not None:
        #         for k in range(num_kernels):
        #             obs.hp2_list[k].data = hp2_list[i + k].data.to(obs.device)
        #     if noise_list is not None:
        #         obs.noise.data = noise_list[i].data.to(obs.device)
        #     if mu_list is not None:
        #         for k in range(num_kernels):
        #             obs.mu_list[k].data = mu_list[i + k].data.to(obs.device)

        #     i += num_kernels

    def set_random_hyperparameters(
            self,
            seed: int = 42,
            scale: float | list | tuple = 1.0
        ):
        """
        Randomly initialize hp1 and hp2 for all kernel objects contained in all
        GPObservable instances.

        Parameters
        ----------
        seed : int
            Seed for reproducibility.

        scale : float, list, or tuple
            Controls the random initialization range.

            If a single float:
                hp1 ~ Uniform(0, scale)
                hp2 ~ Uniform(0, 2*scale)

            If a list/tuple of length 2:
                scale = [scale_hp1, scale_hp2]

                hp1 ~ Uniform(0, scale_hp1)
                hp2 ~ Uniform(0, 2*scale_hp2)

            If a list/tuple of length 3:
                scale = [scale_hp1, scale_hp2, _]

                The third entry is accepted for backward compatibility with the
                older mu-list-based implementation, but is ignored.

            Setting scale_hp1 or scale_hp2 to None leaves that parameter family
            unchanged.

        Notes
        -----
        - Only kernels inheriting from TwoPositiveParameterKernel are affected.
        - Attractor locations `mu` are intentionally not modified.
        - Other optional kernel parameters such as alpha or period are not modified.
        """

        torch.manual_seed(seed)

        # --------------------------------------------------
        # Parse scale argument
        # --------------------------------------------------
        if isinstance(scale, (list, tuple)):
            if len(scale) == 2:
                scale_hp1, scale_hp2 = scale
            elif len(scale) == 3:
                scale_hp1, scale_hp2, _ = scale # add mu_list later
            else:
                raise ValueError(
                    "scale must be either a single float, "
                    "a list/tuple of length 2 or 3 ")
        else:
            scale_hp1 = scale
            scale_hp2 = scale
        
        if scale_hp1 is not None and scale_hp1 <= 0:
            raise ValueError('hp1 Scale must be None or positive '
                             f'Recieved {scale_hp1}')
        if scale_hp2 is not None and scale_hp2 <= 0:
            raise ValueError('hp2 Scale must be None or positive '
                             f'Recieved {scale_hp2}')

        # --------------------------------------------------
        # Internal helper:
        # Convert a desired positive realized parameter value p
        # into the corresponding raw parameter value satisfying
        #
        #     p = softplus(raw_p) + eps
        # --------------------------------------------------
        def _raw_from_positive(
                positive_value: torch.Tensor,
                eps: float,
                beta : float = 10.0,
                thresh : float = 5.0,
            ) -> torch.Tensor:

            shifted = positive_value - eps

            # Clamp the softplus argument away from zero for numerical safety.
            tiny = torch.finfo(positive_value.dtype).tiny
            shifted = torch.clamp(shifted, min=tiny)

            return torch.where(
                beta * shifted > thresh,
                shifted,
                torch.log(torch.expm1(beta * shifted)) / beta
            )

        # --------------------------------------------------
        # Avoid randomizing the same shared kernel module multiple times
        # if a kernel object is shared across observables.
        # --------------------------------------------------
        visited_modules = set()

        with torch.no_grad():

            for obs in self.observables.values():

                # Recursively visits:
                #   - the outer kernel itself
                #   - child kernels inside SumKernel / ProductKernel
                for kernel_module in obs.kernel.modules():

                    if id(kernel_module) in visited_modules:
                        continue

                    visited_modules.add(id(kernel_module))

                    if not isinstance(kernel_module, TwoPositiveParameterKernel):
                        continue

                    # Randomize hp1
                    if scale_hp1 is not None:

                        hp1_rand = scale_hp1 * torch.rand_like(
                            kernel_module.raw_hp1).clamp(min=kernel_module.eps)

                        raw_hp1_rand = _raw_from_positive(hp1_rand, kernel_module.eps)

                        kernel_module.raw_hp1.copy_(raw_hp1_rand)

                    # Randomize hp2
                    if scale_hp2 is not None:
                        
                        hp2_rand = 2.0 * scale_hp2 * torch.rand_like(
                            kernel_module.raw_hp2).clamp(min=kernel_module.eps)

                        raw_hp2_rand = _raw_from_positive(hp2_rand, kernel_module.eps)

                        kernel_module.raw_hp2.copy_(raw_hp2_rand)

    def train_observable(self, index, Xtrain, ytrain):
        if index not in self.observables:
            raise ValueError(f'Observable with index {index} does not exist.')
        self.observables[index].trainGP(Xtrain, ytrain)

    def predict_mean(self, index : int, Xq : torch.Tensor) -> torch.Tensor:
        if index not in self.observables:
            raise ValueError(f'Observable with index {index} does not exist.')
        return self.observables[index].predictMean(Xq)

    def predict_covariance(self, index, Xq):
        if index not in self.observables:
            raise ValueError(f'Observable with index {index} does not exist.')
        return self.observables[index].predictCov(Xq)

    def optimize_hyperparameters(self, num_iter=100, lr=0.01, opt_noise=True):
        for obs in self.observables.values():
            obs.optimize_hyperparameters(
                num_iter, lr, opt_noise=opt_noise)

    def visualize2D(self, resolution=50, range_x=(-1, 1), range_y=(-1, 1)):
        """
        Generate surface plots for all 2D observables in the manager.

        Args:
            resolution (int): Number of points along each axis for the grid.
            range_x (tuple): Range of values for the first input dimension (min, max).
            range_y (tuple): Range of values for the second input dimension (min, max).
        """
        for idx, observable in self.observables.items():
            # Check if the observable input dimension is 2
            if observable.Xtrain.shape[0] != 2:
                raise ValueError(
                    f"Observable {idx} does not have 2D inputs and cannot be plotted.")

            # Create a meshgrid for plotting
            x = np.linspace(*range_x, resolution)
            y = np.linspace(*range_y, resolution)
            X, Y = np.meshgrid(x, y)
            grid_points = torch.tensor(
                np.vstack([X.ravel(), Y.ravel()]), dtype=torch.float32).to(observable.device)

            # Predict the mean values for the grid points
            Z = observable.predictMean(grid_points)
            Z = Z.cpu()
            Z = Z.detach().numpy().reshape(resolution, resolution)

            # Plot the surface
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.8)
            ax.set_title(f"Observable {idx+1} Surface Plot")
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.set_zlabel("Mean")
            plt.show()

    def print_parameters(self, indices=None, get_KTypes=True, get_noise=True, get_hp1=True, get_hp2=True, get_mu=True):
        """
        Prints a table with hyperparameter details for each GPObservable in scientific notation 
        with 3 significant digits.

        Each row corresponds to an observable (or only those specified via indices)
        and includes the following columns (if requested):
        - Observable index
        - Kernel types (comma-separated)
        - Noise value
        - hp1 value(s)
        - hp2 value(s)
        - mu value(s)
        """
        # Determine which observables to print.
        if indices is None:
            indices = sorted(self.observables.keys())
        else:
            indices = [idx for idx in indices if idx in self.observables]

        # Build the header based on requested columns.
        headers = ["Index"]
        if get_KTypes:
            headers.append("Kernel Types")
        if get_noise:
            headers.append("Noise")
        if get_hp1:
            headers.append("hp1")
        if get_hp2:
            headers.append("hp2")
        if get_mu:
            headers.append("mu")

        # Gather table rows.
        rows = []
        for idx in indices:
            obs = self.observables[idx]
            row = [str(idx)]
            if get_KTypes:
                row.append(", ".join(obs.kernel_types))
            if get_noise:
                row.append(f"{obs.noise.detach().cpu().item():.3e}")
            if get_hp1:
                hp1_str = ", ".join(
                    f"{p.detach().cpu().item():.3e}" if p.numel() == 1 else
                    "[" +
                    ", ".join(f"{x:.3e}" for x in p.detach(
                    ).cpu().view(-1).tolist()) + "]"
                    for p in obs.hp1_list
                )
                row.append(hp1_str)
            if get_hp2:
                hp2_str = ", ".join(
                    f"{p.detach().cpu().item():.3e}" if p.numel() == 1 else
                    "[" +
                    ", ".join(f"{x:.3e}" for x in p.detach(
                    ).cpu().view(-1).tolist()) + "]"
                    for p in obs.hp2_list
                )
                row.append(hp2_str)
            if get_mu:
                mu_str = ", ".join(
                    f"{p.detach().cpu().item():.3e}" if p.numel() == 1 else
                    "[" +
                    ", ".join(f"{x:.3e}" for x in p.detach(
                    ).cpu().view(-1).tolist()) + "]"
                    for p in obs.mu_list
                )
                row.append(mu_str)
            rows.append(row)

        # Determine the maximum width for each column (for neat printing).
        col_widths = [max(len(str(item)) for item in col)
                      for col in zip(headers, *rows)]

        # Print header.
        header_line = " | ".join(header.ljust(width)
                                 for header, width in zip(headers, col_widths))
        print(header_line)
        print("-" * len(header_line))

        # Print each row.
        for row in rows:
            print(" | ".join(cell.ljust(width)
                  for cell, width in zip(row, col_widths)))


def getKoopman(manager: GPObservablesManager,
               X :torch.Tensor, Xplus : torch.Tensor,
               nT : int, stateAug : bool = False) -> tuple[torch.Tensor]:
    """
    Compute Koopman A matrix using the manager for GPObservables.

    Args:
        manager (GPObservablesManager): Manager holding all GPObservable objects.
        indices (list): List of indices for observables to include.
        Xall (torch.Tensor): n x (N+1) matrix of state trajectory.
        nT (float): number of trajectories in training dataset

    Returns: tuple[torch.Tensor]
        A (torch.Tensor): p x p linear state transition matrix.
        C (torch.Tensor): n x p output matrix.
    """

    if not isinstance(manager, GPObservablesManager):
        raise ValueError(
            'Expected argument manager to be object of class GPObservablesManager')
    
    if X.shape != Xplus.shape:
        raise ValueError('X and Xplus should be the same shape. '
                         f'Recieved {X.shape} and {Xplus.shape} instead.')

    # n = Xall.shape[0]       # dimensionality of original system
    # N = (Xall.shape[1])//nT - 1  # Number of time steps in each trajectory
    # p = len(indices)        # number of observables
    nx = X.shape[0]
    N = X.shape[1]//nT
    nz = manager.num_obs
    device = manager.observables[0].device
    X, Xplus = X.to(device=device), Xplus.to(device=device)

    M = torch.zeros((nz, N*nT), device=device)
    Mplus = torch.zeros((nz, N*nT), device=device)
    for i in range(nz):
        M[i, :] = torch.transpose(manager.predict_mean(i, X), dim0=0, dim1=-1)
        Mplus[i, :] = torch.transpose(
            manager.predict_mean(i, Xplus), dim0=0, dim1=-1)

    if stateAug:
        M = torch.vstack((X, M))
        Mplus = torch.vstack((Xplus, Mplus))

    # Compute C(z) and A(z)
    try:
        L = torch.linalg.cholesky(M @ M.mT +
                (1e-8) * torch.eye(nz, device=M.device))
        M_pinv = torch.cholesky_solve(M.mT, L)
    except RuntimeError:
        M_pinv = torch.linalg.pinv(M)
    
    A = Mplus @ M_pinv

    if stateAug:
        C = torch.stack([torch.eye(nx), 
                         torch.zeros([nx, nz])], dim=1).to(device=M.device)
    else:
        C = X @ M_pinv

    return A, C


def getKoopman_control(manager, indices, X, Xplus, U, nT, stateAug=False):
    """
    Compute Koopman A matrix using the manager for GPObservables.

    Args:
        manager (GPObservablesManager): Manager holding all GPObservable objects.
        indices (list): List of indices for observables to include.
        X (torch.Tensor): n x N matrix of state trajectory.
        Xplus (torch.Tensor): n x N matrix of time-shifted state trajectories
        nT (float): number of trajectories in training dataset

    Returns:
        A (torch.Tensor): p x p linear state transition matrix.
        C (torch.Tensor): n x p output matrix.
    """

    if not isinstance(manager, GPObservablesManager):
        raise ValueError(
            'Expected argument manager to be object of class GPObservablesManager')

    n = X.shape[0]       # dimensionality of original system
    # N = (Xall.shape[1])//nT - 1  # Number of time steps in each trajectory
    N = X.shape[1]//nT
    p = len(indices)        # number of observables

    M = torch.zeros((p, N*nT))
    Mplus = torch.zeros((p, N*nT))
    for i in range(p):
        M[i, :] = torch.transpose(manager.predict_mean(i, X), dim0=0, dim1=-1)
        Mplus[i, :] = torch.transpose(
            manager.predict_mean(i, Xplus), dim0=0, dim1=-1)

    if stateAug:
        Mfull = torch.vstack((X, M, U))
        Mplus = torch.vstack((Xplus, Mplus))
    else:
        Mfull = torch.vstack((M, U))

    # Compute C(z) and A(z)
    Mf_pinv = torch.linalg.pinv(Mfull)
    K = Mplus @ Mf_pinv
    if stateAug:
        A = K[:, 0:(n+p)]
        B = K[:, (n+p):]
    else:
        A = K[:, 0:p]
        B = K[:, p:]

    if stateAug:
        C = torch.zeros((n, n+p))
        for i in range(n):
            C[i, i] = 1.
    else:
        C = X @ torch.linalg.pinv(M)

    return A, B, C


if __name__ == "__main__":
    from kernels import GaussianKernel
    from prior_means import MonomialMean
    # Create the GPObservablesManager
    manager = GPObservablesManager()

    # 1. Add Observables with multiple kernel types and combination methods
    Ns=50
    manager.add_observable(
        index=0, d=1, Ns=50,
        kernel=GaussianKernel(hp1=1.0, hp2=1.5), device='cpu'
    )

    manager.add_observable(
        index=1, d=1, Ns=50,
        kernel=GaussianKernel(hp1=2.0, hp2=0.5, device='cpu'),
        prior_mean=MonomialMean([2.0]), device='cpu'
    )

    # 2. Prepare synthetic training data (2D inputs, 5 samples)
    X_train = torch.linspace(0., 20., steps=Ns).unsqueeze(dim=0)
    # X_train = torch.vstack([X_train, 2*X_train])
    y_train = ((0.025 * X_train ** 2) + 2 * (torch.sin(1.1 * X_train)) + 2 * torch.randn_like(X_train)).mT

    # 3. Train the observables
    manager.train_observable(0, X_train, y_train)  # Train first observable
    manager.train_observable(1, X_train, y_train)  # Train second observable
    manager.optimize_hyperparameters(num_iter=20, lr=0.001)

    # 4. Predict mean and covariance for a query input (2D query, 3 samples)
    X_query = 15 * torch.rand(1, 10)
    mean_pred_0 = manager.predict_mean(
        0, X_query)  # Predict mean for observable 0
    # Predict covariance for observable 0
    cov_pred_0 = manager.predict_covariance(0, X_query)

    mean_pred_1 = manager.predict_mean(
        1, X_query)  # Predict mean for observable 1
    # Predict covariance for observable 1
    cov_pred_1 = manager.predict_covariance(1, X_query)

    # print(f"Mean prediction for Observable 0:\n{mean_pred_0}")
    # print(f"Covariance prediction for Observable 0:\n{cov_pred_0}")

    # print(f"Mean prediction for Observable 1:\n{mean_pred_1}")
    # print(f"Covariance prediction for Observable 1:\n{cov_pred_1}")

    # 5. Optimize hyperparameters for all observables
    
    plt.plot(X_train.squeeze().detach(), y_train.squeeze().detach(), label='Data')
    plt.plot(X_query.squeeze().detach(), mean_pred_0.squeeze().detach(), label='Naive', linestyle=None, marker='o')
    plt.plot(X_query.squeeze().detach(), mean_pred_1.squeeze().detach(), label='With prior', linestyle=None, marker='+')
    plt.grid()
    plt.show()

    # # 6. Get optimized hyperparameters for observable 0 and 1
    # params_0 = manager.get_params(0)
    # params_1 = manager.get_params(1)

    # print(f"Optimized hyperparameters for Observable 0: {params_0}")
    # print(f"Optimized hyperparameters for Observable 1: {params_1}")

    # # 7. Setting random hyperparameters for all observables
    # manager.set_random_hyperparameters()

    # # 7. Get all parameters for all observables in the manager
    # all_params = manager.get_all_params()
    # print(f"All randomized hyperparameters for all observables: {all_params}")

    # # 8. Count the number of observables
    # observable_count = GPObservable.count_Observables()
    # print(f"Total number of observables: {observable_count}")

    # manager.visualize2D()
