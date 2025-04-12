
import torch
import torch.nn as nn
import math
from matplotlib import pyplot as plt
import numpy as np

# Define individual kernel functions


def GaussianKernel(X1, X2, hp1, hp2, mu):
    dists = torch.cdist(X1.T, X2.T, p=2)**2
    return hp1 * torch.exp(-dists / (2 * hp2**2))


def ThinSplineKernel(X1, X2, hp1, hp2, mu):
    dists = torch.cdist(X1.T, X2.T, p=2)
    epsilon = 1e-8
    return hp1 * ((dists/hp2)**2) * torch.log((dists / hp2) + epsilon)


def InverseQuadraticKernel(X1, X2, hp1, hp2, mu):
    dists = torch.cdist(X1.T, X2.T, p=2)**2
    return hp1 * (1 / (1 + (dists / (hp2**2))))


def CosineKernel(X1, X2, hp1, hp2, mu):
    dists = torch.cdist(X1.T, X2.T, p=2)
    return hp1 * torch.cos(math.pi * dists / hp2) ** 2


def ExpSineSqrKernel(X1, X2, hp1, hp2, mu):
    dists = torch.cdist(X1.T, X2.T, p=2)
    hp2 = torch.clamp(hp2, min=1e-2)
    epsilon = 1e-4
    return hp1 * torch.exp(epsilon - 0.5 * ((torch.sin(math.pi * dists / hp1)) ** 2) / hp2)
    # return torch.exp((-2 * (torch.sin(math.pi * dists / hp1))**2) / hp2**2)


def GibbsExpAttractorKernel(X1, X2, hp1, hp2, mu):
    """
    Gibbs kernel with an attractor-dependent exponential decay length scale.
    The length scale decreases with distance from the attractor.

    Args:
        X1 (Tensor): First set of points (d, n1).
        X2 (Tensor): Second set of points (d, n2).
        hp1 (Tensor): Scaling factor for the length scale.
        hp2 (Tensor): Baseline length scale.
        mu (Tensor): Location of the attractor (d, 1).

    Returns:
        Tensor: Kernel matrix of shape (n1, n2).
    """
    # Shift inputs by the attractor location
    # X1_shifted = X1 - mu.unsqueeze(1)  # (d, n1)
    # X2_shifted = X2 - mu.unsqueeze(1)  # (d, n2)
    X1_shifted = X1 - mu
    X2_shifted = X2 - mu

    # Compute squared norm relative to attractor
    X1_norm2 = torch.sum(X1_shifted**2, dim=0)  # (n1,)
    X2_norm2 = torch.sum(X2_shifted**2, dim=0)  # (n2,)

    # Compute input-dependent length scales
    l_X1 = hp1 + hp2 * torch.exp(-X1_norm2)  # (n1,)
    l_X2 = hp1 + hp2 * torch.exp(-X2_norm2)  # (n2,)

    # Compute Gibbs kernel denominator
    L_sum = (l_X1**2).unsqueeze(1) + (l_X2**2).unsqueeze(0)  # (n1, n2)
    outer_l = l_X1.unsqueeze(1) * l_X2.unsqueeze(0)  # (n1, n2)
    factor = torch.sqrt(2.0 * outer_l / L_sum)

    # Compute squared Euclidean distances
    dists = torch.cdist(X1.T, X2.T, p=2)**2  # (n1, n2)

    # Compute kernel matrix
    return factor * torch.exp(-dists / L_sum)


def ExplicitAttractorKernel(X1, X2, hp1, hp2, mu):
    """
    Computes the Explicit Kernel K_Explicit(x, x') for attractor-based covariance.

    Args:
        X1 (Tensor): First set of points (d, n1).
        X2 (Tensor): Second set of points (d, n2).
        hp1 (Tensor): Scaling factor (theta_1).
        hp2 (Tensor): Length scale (theta_2).
        mu (Tensor): Location of the attractor (d, 1).

    Returns:
        Tensor: Kernel matrix (n1, n2).
    """
    # Compute squared distances from the attractor for X1 and X2
    X1_shifted_norm2 = torch.sum((X1 - mu) ** 2, dim=0)  # (n1,)
    X2_shifted_norm2 = torch.sum((X2 - mu) ** 2, dim=0)  # (n2,)

    # Compute the exponent term
    exponent = -(X1_shifted_norm2.unsqueeze(1) +
                 X2_shifted_norm2.unsqueeze(0)) / (2 * hp2**2)

    # Compute the kernel matrix
    return hp1 * torch.exp(exponent)


# Dictionary mapping kernel names to functions
KERNEL_FUNCTIONS = {
    'Gaussian': GaussianKernel,
    'ThinSpline': ThinSplineKernel,
    'InverseQuadratic': InverseQuadraticKernel,
    'ExpSineSqr': ExpSineSqrKernel,
    'Cosine': CosineKernel,
    'GibbsExpAttractor': GibbsExpAttractorKernel,
    'ExplicitAttractor': ExplicitAttractorKernel
}


def KernelFunction(X1, X2=None, kernel_types=['Gaussian'], hp1_list=None, hp2_list=None, mu_list=None, combination='sum'):
    """
    Computes a kernel matrix using multiple kernels, supporting attractor-based kernels.

    Args:
        X1 (Tensor): First set of points.
        X2 (Tensor, optional): Second set of points. Defaults to None.
        kernel_types (list): List of kernel function names.
        hp1_list (list, optional): List of first hyperparameters for each kernel.
        hp2_list (list, optional): List of second hyperparameters for each kernel.
        mu_list (list, optional): List of attractor locations for each kernel.
        combination (str): How to combine kernels ('sum' or 'product').

    Returns:
        Tensor: Kernel matrix.
    """
    if X2 is None:
        X2 = X1

    if hp1_list is None:
        hp1_list = [torch.tensor(1.0, device=X1.device)] * len(kernel_types)
    if hp2_list is None:
        hp2_list = [torch.tensor(1.0, device=X1.device)] * len(kernel_types)
    if mu_list is None:
        mu_list = [None] * len(kernel_types)

    if len(hp1_list) != len(kernel_types) or len(hp2_list) != len(kernel_types) or len(mu_list) != len(kernel_types):
        raise ValueError(
            "Number of hyperparameters and attractor locations must match the number of kernel types.")

    # Compute the kernel matrix
    if combination == 'sum':
        K = sum(KERNEL_FUNCTIONS[kernel](X1, X2, hp1, hp2, mu)
                for kernel, hp1, hp2, mu in zip(kernel_types, hp1_list, hp2_list, mu_list))
    elif combination == 'product':
        K = torch.ones((X1.shape[1], X2.shape[1]), device=X1.device)
        for kernel, hp1, hp2, mu in zip(kernel_types, hp1_list, hp2_list, mu_list):
            K *= KERNEL_FUNCTIONS[kernel](X1, X2, hp1, hp2, mu)
    else:
        raise ValueError(
            "Invalid combination method. Choose 'sum' or 'product'.")

    return K


class GPObservable:
    count = 0

    def __init__(self, d, ns, kernel_types=['Gaussian'], hp1_list=None, hp2_list=None,
                 mu_list=None, noise=2e-8, combination='sum', device='cuda:0', m=200):
        """
        Gaussian Process Observable with customizable kernel functions.

        Args:
            d (int): Dimensionality of the inputs.
            ns (int): Number of training samples.
            kernel_types (list): List of kernel names.
            hp1_list (list): List of first hyperparameters for each kernel.
            hp2_list (list): List of second hyperparameters for each kernel.
            mu_list (list): List of attractor locations for each kernel.
            noise (float): Observation noise.
            combination (str): Kernel combination method ('sum' or 'product').
            device (str): Compute device ('cuda:0' or 'cpu').
            m (int): Number of inducing points (or similar parameter).
        """
        self.device = torch.device(device)
        self.kernel_types = kernel_types  # List of kernel names
        self.combination = combination    # How kernels are combined

        # Create nn.ParameterLists for hyperparameters and mu
        if hp1_list is not None:
            self.hp1_list = nn.ParameterList(
                [param if isinstance(param, nn.Parameter)
                 else nn.Parameter(param.clone().detach() if isinstance(param, torch.Tensor)
                                   else torch.tensor(param, device=self.device))
                 for param in hp1_list]
            )
        else:
            self.hp1_list = nn.ParameterList([
                nn.Parameter(torch.tensor(1.0, device=self.device))
                for _ in kernel_types
            ])

        if hp2_list is not None:
            self.hp2_list = nn.ParameterList(
                [param if isinstance(param, nn.Parameter)
                 else nn.Parameter(param.clone().detach() if isinstance(param, torch.Tensor)
                                   else torch.tensor(param, device=self.device))
                 for param in hp2_list]
            )
        else:
            self.hp2_list = nn.ParameterList([
                nn.Parameter(torch.tensor(1.0, device=self.device))
                for _ in kernel_types
            ])

        if mu_list is not None:
            self.mu_list = nn.ParameterList(
                [param if isinstance(param, nn.Parameter)
                 else nn.Parameter(param.clone().detach() if isinstance(param, torch.Tensor)
                                   else torch.tensor(param, device=self.device))
                 for param in mu_list]
            )
        else:
            self.mu_list = nn.ParameterList([
                nn.Parameter(torch.zeros((d, 1), device=self.device))
                for _ in kernel_types
            ])

        if len(self.hp1_list) != len(kernel_types) or len(self.hp2_list) != len(kernel_types):
            raise ValueError(
                "Number of hyperparameters must match the number of kernel types.")

        self.noise = nn.Parameter(torch.tensor(noise, device=self.device))

        self.m = min(m, ns)
        self.idx_SOR = torch.linspace(0, ns-1, self.m).int()
        self.Xm = torch.empty((self.m, self.m), device=self.device)
        self.Knm = torch.empty((ns, self.m), device=self.device)
        self.y = torch.empty((ns, 1), device=self.device)  # Target values
        # SOR trained coefficient
        self.aSOR = torch.empty((ns, 1), device=self.device)
        GPObservable.count += 1

    def set_hyperparameters(self, hp1_list=None, hp2_list=None):
        # if hp1_list is not None:
        #     self.hp1_list = nn.ParameterList([
        #         nn.Parameter(torch.tensor(hp, device=self.device))
        #         for hp in hp1_list
        #     ])
        if hp1_list is not None:
            self.hp1_list = nn.ParameterList(
                [param if isinstance(param, nn.Parameter)
                 else nn.Parameter(param.clone().detach() if isinstance(param, torch.Tensor)
                                   else torch.tensor(param, device=self.device))
                 for param in hp1_list]
            )
        # if hp2_list is not None:
        #     self.hp2_list = nn.ParameterList([
        #         nn.Parameter(torch.tensor(hp, device=self.device))
        #         for hp in hp2_list
        #     ])
        if hp2_list is not None:
            self.hp2_list = nn.ParameterList(
                [param if isinstance(param, nn.Parameter)
                 else nn.Parameter(param.clone().detach() if isinstance(param, torch.Tensor)
                                   else torch.tensor(param, device=self.device))
                 for param in hp2_list]
            )

    def get_parameters(self):
        """
        Returns a dictionary containing the current hyperparameters (hp1_list, hp2_list), noise,
        and attractor locations (mu_list) for this GPObservable.
        """
        params = {
            "hp1_list": [hp.detach() for hp in self.hp1_list],
            "hp2_list": [hp.detach() for hp in self.hp2_list],
            "noise": self.noise.detach(),
            "mu_list": [mu.detach() for mu in self.mu_list]
        }
        return params

    def trainGP(self, Xtrain, ytrain):
        Xtrain = Xtrain.to(self.device)
        ytrain = ytrain.to(self.device)
        self.Xtrain = Xtrain
        self.y = ytrain

        self.Xm = Xtrain[:, self.idx_SOR]

        self.Kmm = KernelFunction(self.Xm, self.Xm, kernel_types=self.kernel_types,
                                  hp1_list=self.hp1_list, hp2_list=self.hp2_list,
                                  mu_list=self.mu_list, combination=self.combination)
        self.Kmn = KernelFunction(self.Xm, self.Xtrain, kernel_types=self.kernel_types,
                                  hp1_list=self.hp1_list, hp2_list=self.hp2_list,
                                  mu_list=self.mu_list, combination=self.combination)

        try:
            L = torch.linalg.cholesky(
                (self.Kmn @ self.Kmn.T) + (self.noise ** 2) * self.Kmm)
            self.invKmm = torch.cholesky_inverse(L)
        except RuntimeError:
            U, S, V = torch.linalg.svd(
                (self.Kmn @ self.Kmn.T) + (self.noise ** 2) * self.Kmm)
            S_inv = torch.diag(torch.where(
                S > 1e-6, 1.0 / S, torch.tensor(0.0, device=self.device)))
            self.invKmm = V.T @ S_inv @ U.T

        self.alpha = self.invKmm @ self.Kmn @ self.y

    def predictGP(self, Xq):
        Xq = Xq.to(self.device)
        Kqm = KernelFunction(Xq, self.Xm, kernel_types=self.kernel_types,
                             hp1_list=self.hp1_list, hp2_list=self.hp2_list,
                             mu_list=self.mu_list, combination=self.combination)
        mean = Kqm @ self.alpha
        CovMat = (Kqm @ self.invKmm @ Kqm.T) * (self.noise ** 2)
        return mean, CovMat

    def predictMean(self, Xq):
        Xq = Xq.to(self.device)
        Kqm = KernelFunction(Xq, self.Xm, kernel_types=self.kernel_types,
                             hp1_list=self.hp1_list, hp2_list=self.hp2_list,
                             mu_list=self.mu_list, combination=self.combination)
        return Kqm @ self.alpha

    def predictCov(self, Xq):
        Xq = Xq.to(self.device)
        Kqm = KernelFunction(Xq, self.Xm, kernel_types=self.kernel_types,
                             hp1_list=self.hp1_list, hp2_list=self.hp2_list,
                             mu_list=self.mu_list, combination=self.combination)
        return (Kqm @ self.invKmm @ Kqm.T) * (self.noise ** 2)

    def forward(self, Xq, ytrain):
        """
        Fully differentiable forward pass that computes predictions using the current hyperparameters.
        """
        Xq = Xq.to(self.device)
        ytrain = ytrain.to(self.device)

        # hp1_list = [torch.nn.functional.softplus(
        #     r, beta=10.) for r in self.hp1_list]
        # hp2_list = [torch.nn.functional.softplus(
        #     r, beta=10.) for r in self.hp2_list]
        hp1_list = self.hp1_list
        hp2_list = self.hp2_list

        Kmm = KernelFunction(self.Xm, self.Xm, kernel_types=self.kernel_types,
                             hp1_list=hp1_list, hp2_list=hp2_list,
                             mu_list=self.mu_list, combination=self.combination)
        Kmn = KernelFunction(self.Xm, self.Xtrain, kernel_types=self.kernel_types,
                             hp1_list=hp1_list, hp2_list=hp2_list,
                             mu_list=self.mu_list, combination=self.combination)
        # L = torch.linalg.cholesky(
        #     (Kmn @ Kmn.T) + (self.noise ** 2) * Kmm)
        # invKmm = torch.cholesky_inverse(L)
        try:
            L = torch.linalg.cholesky(
                (Kmn @ Kmn.T) + (self.noise ** 2) * Kmm)
            invKmm = torch.cholesky_inverse(L)
        except RuntimeError:
            U, S, V = torch.linalg.svd(
                (Kmn @ Kmn.T) + (self.noise ** 2) * Kmm)
            S_inv = torch.diag(torch.where(
                S > 1e-6, 1.0 / S, torch.tensor(0.0, device=self.device)))
            invKmm = V.T @ S_inv @ U.T
        alpha = invKmm @ Kmn @ ytrain
        Kqm = KernelFunction(Xq, self.Xm, kernel_types=self.kernel_types,
                             hp1_list=hp1_list, hp2_list=hp2_list,
                             mu_list=self.mu_list, combination=self.combination)
        mean = Kqm @ alpha
        cov = (Kqm @ invKmm @ Kqm.T) * (self.noise ** 2)
        return mean, cov

    def forward_mean(self, Xq, ytrain):
        """
        Fully differentiable forward pass that computes predictive mean using current hyperparameters
        """
        Xq = Xq.to(self.device)
        ytrain = ytrain.to(self.device)

        Kmm = KernelFunction(self.Xm, self.Xm, kernel_types=self.kernel_types,
                             hp1_list=self.hp1_list, hp2_list=self.hp2_list,
                             mu_list=self.mu_list, combination=self.combination)
        Kmn = KernelFunction(self.Xm, self.Xtrain, kernel_types=self.kernel_types,
                             hp1_list=self.hp1_list, hp2_list=self.hp2_list,
                             mu_list=self.mu_list, combination=self.combination)
        # L = torch.linalg.cholesky((Kmn @ Kmn.T) + (self.noise ** 2) * Kmm)
        # invKmm = torch.cholesky_inverse(L)
        try:
            L = torch.linalg.cholesky(
                (Kmn @ Kmn.T) + (self.noise ** 2) * Kmm)
            invKmm = torch.cholesky_inverse(L)
        except RuntimeError:
            U, S, V = torch.linalg.svd(
                (Kmn @ Kmn.T) + (self.noise ** 2) * Kmm)
            S_inv = torch.diag(torch.where(
                S > 1e-6, 1.0 / S, torch.tensor(0.0, device=self.device)))
            invKmm = V.T @ S_inv @ U.T
        alpha = invKmm @ Kmn @ ytrain
        Kqm = KernelFunction(Xq, self.Xm, kernel_types=self.kernel_types,
                             hp1_list=self.hp1_list, hp2_list=self.hp2_list,
                             mu_list=self.mu_list, combination=self.combination)
        mean = Kqm @ alpha
        return mean

    def forward_cov(self, Xq):
        """
        Fully differentiable forward pass that computes predictive covariance using current hyperparameters
        """
        Xq = Xq.to(self.device)
        Kmm = KernelFunction(self.Xm, self.Xm, kernel_types=self.kernel_types,
                             hp1_list=self.hp1_list, hp2_list=self.hp2_list,
                             mu_list=self.mu_list, combination=self.combination)
        Kmn = KernelFunction(self.Xm, self.Xtrain, kernel_types=self.kernel_types,
                             hp1_list=self.hp1_list, hp2_list=self.hp2_list,
                             mu_list=self.mu_list, combination=self.combination)
        # L = torch.linalg.cholesky((Kmn @ Kmn.T) + (self.noise ** 2) * Kmm)
        # invKmm = torch.cholesky_inverse(L)
        try:
            L = torch.linalg.cholesky(
                (Kmn @ Kmn.T) + (self.noise ** 2) * Kmm)
            invKmm = torch.cholesky_inverse(L)
        except RuntimeError:
            U, S, V = torch.linalg.svd(
                (Kmn @ Kmn.T) + (self.noise ** 2) * Kmm)
            S_inv = torch.diag(torch.where(
                S > 1e-6, 1.0 / S, torch.tensor(0.0, device=self.device)))
            invKmm = V.T @ S_inv @ U.T
        Kqm = KernelFunction(Xq, self.Xm, kernel_types=self.kernel_types,
                             hp1_list=self.hp1_list, hp2_list=self.hp2_list,
                             mu_list=self.mu_list, combination=self.combination)
        cov = (Kqm @ invKmm @ Kqm.T) * (self.noise ** 2)
        return cov

    def optimize_hyperparameters(self, max_iter=100, lr=0.01, opt_mu=False, opt_sigma=False):
        if not hasattr(self, 'Xtrain') or not hasattr(self, 'y'):
            raise ValueError(
                "Training data not found. Please call trainGP before optimizing hyperparameters.")

        # Store original references for hyperparameters and mu
        orig_hp1_list = self.hp1_list
        orig_hp2_list = self.hp2_list
        if opt_sigma:
            orig_noise = self.noise
        if opt_mu:
            orig_mu_list = self.mu_list

        # Create transformed versions for optimization:
        # Use softplus for hp1, hp2, and noise to ensure positivity.
        # (For mu, we allow negative values, so we don't use softplus.)
        opt_hp1_list = [torch.nn.Parameter(
            torch.nn.functional.softplus(hp.detach())) for hp in orig_hp1_list]
        opt_hp2_list = [torch.nn.Parameter(
            torch.nn.functional.softplus(hp.detach())) for hp in orig_hp2_list]
        if opt_sigma:
            opt_noise = torch.nn.Parameter(
                torch.nn.functional.softplus(orig_noise.detach()))
        if opt_mu:
            opt_mu_list = [torch.nn.Parameter(
                mu.detach()) for mu in orig_mu_list]

        # Include all parameters in the optimizer
        if opt_mu and opt_sigma:
            optimizer = torch.optim.Adam(
                [*opt_hp1_list, *opt_hp2_list, opt_noise, *opt_mu_list], lr=lr)
        elif opt_sigma and not opt_mu:
            optimizer = torch.optim.Adam(
                [*opt_hp1_list, *opt_hp2_list, opt_noise], lr=lr)
        elif opt_mu and not opt_sigma:
            optimizer = torch.optim.Adam(
                [*opt_hp1_list, *opt_hp2_list, *opt_mu_list], lr=lr)
        else:
            optimizer = torch.optim.Adam(
                [*opt_hp1_list, *opt_hp2_list], lr=lr)

        for _ in range(max_iter):
            optimizer.zero_grad()

            # Convert softplus-transformed parameters back:
            hp1_opt = [torch.nn.functional.softplus(hp) for hp in opt_hp1_list]
            hp2_opt = [torch.nn.functional.softplus(hp) for hp in opt_hp2_list]
            if opt_sigma:
                noise_opt = torch.nn.functional.softplus(opt_noise)
            else:
                noise_opt = self.noise
            if opt_mu:
                mu_opt = opt_mu_list  # no transformation for mu
            else:
                mu_opt = self.mu_list

            # Compute the kernel matrices using the optimized hyperparameters and mu values.
            jitter = 1e-6 * torch.eye(self.Xm.shape[1], device=self.device)
            try:
                iKmm = torch.linalg.inv(KernelFunction(self.Xm, self.Xm,
                                                       kernel_types=self.kernel_types,
                                                       hp1_list=hp1_opt, hp2_list=hp2_opt,
                                                       mu_list=mu_opt,
                                                       combination=self.combination))
            except RuntimeError:
                try:
                    iKmm = torch.linalg.pinv(KernelFunction(self.Xm, self.Xm,
                                                            kernel_types=self.kernel_types,
                                                            hp1_list=hp1_opt, hp2_list=hp2_opt,
                                                            mu_list=mu_opt,
                                                            combination=self.combination) + jitter)
                except RuntimeError:
                    jitter = 1e-5 * \
                        torch.eye(self.Xm.shape[1], device=self.device)
                    iKmm = torch.linalg.pinv(KernelFunction(self.Xm, self.Xm,
                                                            kernel_types=self.kernel_types,
                                                            hp1_list=hp1_opt, hp2_list=hp2_opt,
                                                            mu_list=mu_opt,
                                                            combination=self.combination) + jitter)

            Kmn = KernelFunction(self.Xm, self.Xtrain,
                                 kernel_types=self.kernel_types,
                                 hp1_list=hp1_opt, hp2_list=hp2_opt,
                                 mu_list=mu_opt,
                                 combination=self.combination)

            K_til = Kmn.T @ iKmm @ Kmn
            K_til += (noise_opt**2) * \
                torch.eye(K_til.shape[0], device=self.device)
            invK_til = torch.linalg.inv(K_til)

            log_det = torch.logdet(K_til)
            ll = -0.5 * (self.y.T @ invK_til @ self.y + log_det +
                         self.y.shape[0] * torch.log(torch.tensor(2 * torch.pi)))
            loss = -ll.squeeze()
            loss.backward()
            optimizer.step()

        # Restore optimized parameters back to the observable as nn.ParameterList objects.
        self.hp1_list = nn.ParameterList(
            [nn.Parameter(hp.detach()) for hp in opt_hp1_list])
        self.hp2_list = nn.ParameterList(
            [nn.Parameter(hp.detach()) for hp in opt_hp2_list])
        self.noise = nn.Parameter(noise_opt.detach())
        if opt_mu:
            self.mu_list = nn.ParameterList(
                [nn.Parameter(mu.detach()) for mu in opt_mu_list])

        # Retrain GP with the updated parameters.
        self.trainGP(self.Xtrain, self.y)


class GPObservablesManager:
    def __init__(self):
        self.observables = {}

    def add_observable(self, index, d, ns, kernel_types=['Gaussian'], hp1_list=None, hp2_list=None, mu_list=None, noise=2e-6, combination='sum', device='cuda:0', m=200):
        if index in self.observables:
            raise ValueError(f'Observable with index {index} already exists.')
        self.observables[index] = GPObservable(
            d, ns, kernel_types, hp1_list, hp2_list, mu_list, noise, combination, device, m)

    def get_params(self, index):
        """
        Returns a dictionary of the current parameters for the observable with the given index.
        """
        if index not in self.observables:
            raise ValueError(f'Observable with index {index} does not exist.')
        return self.observables[index].get_parameters()

    def get_all_params(self):
        """
        Returns a dictionary mapping each observable index to its parameters dictionary.
        """
        if not self.observables:
            raise ValueError('No observables available in manager.')
        return {idx: obs.get_parameters() for idx, obs in self.observables.items()}

    def parameters(self, idx=None, get_mu_only=False):
        """
        Returns a list of all optimizable parameters (hp1, hp2, mu, noise) from all GPObservable instances.
        """
        if idx is None:
            params = []
            if get_mu_only is True:
                for obs in self.observables.values():
                    params.extend(obs.mu_list)
            else:
                for obs in self.observables.values():
                    params.extend(obs.hp1_list)
                    params.extend(obs.hp2_list)
                    params.extend(obs.mu_list)
                    params.append(obs.noise)
        elif isinstance(idx, int):
            pass
        elif isinstance(idx, list) and all(isinstance(index, int) for index in idx):
            pass
        else:
            raise TypeError(
                'Argument idx can be None or object of class int or list')
        return params

    def set_parameters(self, hp1_list=None, hp2_list=None, noise_list=None, mu_list=None):
        i = 0
        for obs in self.observables.values():
            num_kernels = len(obs.kernel_types)

            if hp1_list is not None:
                for k in range(num_kernels):
                    obs.hp1_list[k].data = hp1_list[i + k].data.to(obs.device)
            if hp2_list is not None:
                for k in range(num_kernels):
                    obs.hp2_list[k].data = hp2_list[i + k].data.to(obs.device)
            if noise_list is not None:
                obs.noise.data = noise_list[i].data.to(obs.device)
            if mu_list is not None:
                for k in range(num_kernels):
                    obs.mu_list[k].data = mu_list[i + k].data.to(obs.device)

            i += num_kernels

    def set_random_hyperparameters(self, seed=42, scale=1.0):
        """
        Assigns random hyperparameters (hp1, hp2, and optionally mu_list) to all observables.

        Args:
            seed (int): Seed for reproducibility.
            scale (float or list of three floats): If a single float, the same scale is applied to hp1, hp2, and mu.
                If a list (or tuple) of three floats, they are used as the scales for hp1, hp2, and mu respectively.
        """
        torch.manual_seed(seed)  # For reproducibility

        # Determine scale factors for hp1, hp2, and mu
        if isinstance(scale, (list, tuple)):
            if len(scale) == 3:
                scale_hp1, scale_hp2, scale_mu = scale
            else:
                raise ValueError(
                    "Scale must be a single float or a list/tuple of three floats.")
        else:
            scale_hp1 = scale_hp2 = scale_mu = scale

        for obs in self.observables.values():
            num_kernels = len(obs.kernel_types)
            if scale_hp1 is not None:
                obs.hp1_list = nn.ParameterList([
                    nn.Parameter(scale_hp1 * torch.rand(1,
                                                        device=obs.device, requires_grad=True))
                    for _ in range(num_kernels)
                ])

            if scale_hp2 is not None:
                obs.hp2_list = nn.ParameterList([
                    nn.Parameter(scale_hp2 * torch.rand(1,
                                                        device=obs.device, requires_grad=True))
                    for _ in range(num_kernels)
                ])

            # Randomize mu_list if it exists (i.e. is not None)
            if obs.mu_list is not None and scale_mu is not None:
                obs.mu_list = nn.ParameterList([
                    nn.Parameter((-scale_mu/2) + scale_mu * torch.rand(*p.shape,
                                 device=obs.device, requires_grad=True))
                    for p in obs.mu_list
                ])

    def train_observable(self, index, Xtrain, ytrain):
        if index not in self.observables:
            raise ValueError(f'Observable with index {index} does not exist.')
        self.observables[index].trainGP(Xtrain, ytrain)

    def predict_mean(self, index, Xq):
        if index not in self.observables:
            raise ValueError(f'Observable with index {index} does not exist.')
        return self.observables[index].predictMean(Xq)

    def predict_covariance(self, index, Xq):
        if index not in self.observables:
            raise ValueError(f'Observable with index {index} does not exist.')
        return self.observables[index].predictCov(Xq)

    def optimize_hyperparameters(self, max_iter=100, lr=0.01, opt_mu=False, opt_sigma=False):
        for obs in self.observables.values():
            obs.optimize_hyperparameters(
                max_iter, lr, opt_mu=opt_mu, opt_sigma=opt_sigma)

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


def getKoopman(manager:GPObservablesManager, indices:list, Xall:torch.tensor, nT, stateAug=False):
    """
    Compute Koopman A matrix using the manager for GPObservables.

    Args:
        manager (GPObservablesManager): Manager holding all GPObservable objects.
        indices (list): List of indices for observables to include.
        Xall (torch.Tensor): n x (N+1) matrix of state trajectory.
        nT (float): number of trajectories in training dataset

    Returns:
        A (torch.Tensor): p x p linear state transition matrix.
        C (torch.Tensor): n x p output matrix.
    """

    if not isinstance(manager, GPObservablesManager):
        raise ValueError(
            'Expected argument manager to be object of class GPObservablesManager')

    n = Xall.shape[0]       # dimensionality of original system
    N = (Xall.shape[1])//nT - 1  # Number of time steps in each trajectory
    p = len(indices)        # number of observables

    X = torch.cat([Xall[:, j*(N+1):j*(N+1)+N] for j in range(nT)],
                  dim=1)         # Data matrix from original system
    Xplus = torch.cat([Xall[:, j*(N+1)+1:j*(N+1)+N+1]
                      for j in range(nT)], dim=1)  # Time-shifted data matrix

    M = torch.empty((p, N*nT), device=Xall.device)
    Mplus = torch.empty((p, N*nT), device=Xall.device)
    for i in range(p):
        M[i, :] = torch.transpose(manager.predict_mean(i, X), dim0=0, dim1=-1)
        Mplus[i, :] = torch.transpose(
            manager.predict_mean(i, Xplus), dim0=0, dim1=-1)

    if stateAug:
        M = torch.vstack((X, M))
        Mplus = torch.vstack((Xplus, Mplus))

    # Compute C(z) and A(z)
    M_pinv = torch.linalg.pinv(M)
    A = Mplus @ M_pinv

    if stateAug:
        C = torch.zeros((n, n+p))
        for i in range(n):
            C[i, i] = 1.
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

    M = torch.empty((p, N*nT))
    Mplus = torch.empty((p, N*nT))
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
    # Create the GPObservablesManager
    manager = GPObservablesManager()

    # 1. Add Observables with multiple kernel types and combination methods
    manager.add_observable(
        index=0, d=2, ns=5,
        kernel_types=['Gaussian', 'ThinSpline'],
        hp1_list=[torch.tensor(1.0), torch.tensor(0.5)],
        hp2_list=[torch.tensor(0.5), torch.tensor(2.0)],
        combination='sum', noise=2e-1
    )

    manager.add_observable(
        index=1, d=2, ns=5,
        kernel_types=['Gaussian', 'InverseQuadratic'],
        hp1_list=[torch.tensor(0.8), torch.tensor(1.2)],
        hp2_list=[torch.tensor(1.0), torch.tensor(0.7)],
        combination='product'
    )

    # 2. Prepare synthetic training data (2D inputs, 5 samples)
    X_train = torch.linspace(0., 1., steps=5).view(1, 5)
    X_train = torch.vstack([X_train, 2*X_train])
    y_train = torch.sin(torch.linspace(0., 1., steps=5).view(5, 1))

    # 3. Train the observables
    manager.train_observable(0, X_train, y_train)  # Train first observable
    manager.train_observable(1, X_train, y_train)  # Train second observable

    # 4. Predict mean and covariance for a query input (2D query, 3 samples)
    X_query = torch.randn(2, 3)
    mean_pred_0 = manager.predict_mean(
        0, X_query)  # Predict mean for observable 0
    # Predict covariance for observable 0
    cov_pred_0 = manager.predict_covariance(0, X_query)

    mean_pred_1 = manager.predict_mean(
        1, X_query)  # Predict mean for observable 1
    # Predict covariance for observable 1
    cov_pred_1 = manager.predict_covariance(1, X_query)

    print(f"Mean prediction for Observable 0:\n{mean_pred_0}")
    print(f"Covariance prediction for Observable 0:\n{cov_pred_0}")

    print(f"Mean prediction for Observable 1:\n{mean_pred_1}")
    print(f"Covariance prediction for Observable 1:\n{cov_pred_1}")

    # 5. Optimize hyperparameters for all observables
    manager.optimize_hyperparameters(max_iter=10, lr=0.001)

    # 6. Get optimized hyperparameters for observable 0 and 1
    params_0 = manager.get_params(0)
    params_1 = manager.get_params(1)

    print(f"Optimized hyperparameters for Observable 0: {params_0}")
    print(f"Optimized hyperparameters for Observable 1: {params_1}")

    # 7. Setting random hyperparameters for all observables
    manager.set_random_hyperparameters()

    # 7. Get all parameters for all observables in the manager
    all_params = manager.get_all_params()
    print(f"All randomized hyperparameters for all observables: {all_params}")

    # 8. Count the number of observables
    observable_count = GPObservable.count_Observables()
    print(f"Total number of observables: {observable_count}")

    manager.visualize2D()
