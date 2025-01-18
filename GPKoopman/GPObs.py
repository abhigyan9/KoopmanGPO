
import math
import torch
import numpy as np
from matplotlib import pyplot as plt


def KernelFunction(X1, X2=None, kernel_type='Gaussian', hp1=torch.tensor([1.0]), hp2=torch.tensor([1.0])):
    """
    Computes the kernel matrix based on the given kernel type.

    Args:
        Y1: Tensor of shape (n, a), first set of points.
        Y2: Tensor of shape (n, b), second set of points.
        kernel_type: String, either 'Gaussian' or 'ThinSpline'.
        hp1: float, 1st hyperparameter
        hp2: float, 2nd hyperparameter

    Returns:
        K: Tensor of shape (a, b), the evaluation of the kernel function on the inputs.
    """
    if not torch.is_tensor(X1):
        raise TypeError(f"Expected X1 to be a torch.Tensor, but got {
                        type(X1).__name__}.")

    n1 = X1.shape[0]

    # If X2 is provided, check that it is a PyTorch tensor; otherwise, default to a zero tensor
    if X2 is not None:
        if not torch.is_tensor(X2):
            raise TypeError(f"Expected X2 to be a torch.Tensor, but got {
                            type(X2).__name__}.")
        else:
            n2 = X2.shape[0]
            if n1 != n2:
                raise ValueError(
                    f'Mismatch in input dimensions. Tensor 1 and 2 have different number of rows, {n1} and {n2}.')
    else:
        # Default X2 to a zero tensor of correct dimension
        X2 = torch.zeros(n1, 1)
    cuda0 = torch.device('cuda:0')
    X1 = X1.to(cuda0)
    X2 = X2.to(cuda0)
    hp1 = hp1.to(cuda0)
    hp2 = hp2.to(cuda0)
    if kernel_type == 'Gaussian':
        # Gaussian RBF kernel
        # hp1 == sigma^2 : Variance
        # hp2 == l : Lengthscale
        # Add method to use symmetric matrix property to reduce computation for X1 == X2
        # Pairwise squared Euclidean distances |
        dists = torch.cdist(X1.T, X2.T, p=2)**2
        # Apply Gaussian kernel formula
        K = hp1 * torch.exp(-dists / (2 * hp2**2))

    elif kernel_type == 'ThinSpline':
        # Thin-Plate Spline kernel
        # Does NOT use hyperparameters
        dists = torch.cdist(X1.T, X2.T, p=2)  # Pairwise Euclidean distances
        epsilon = 1e-8  # Small value to avoid log(0)
        K = hp1 * ((dists/hp2)**2) * torch.log((dists / hp2) +
                                               epsilon)  # Apply Thin-Plate Spline formula

    elif kernel_type == 'InverseQuadratic':
        # Inverse Quadratic Kernel function
        # hp1 == sigma^2 : Variance
        # hp2 == l : Lengthscale
        dists = torch.cdist(X1.T, X2.T, p=2)**2
        K = hp1 * (1 / (1 + (dists / (hp2**2))))

    elif kernel_type == 'ExpSineSqr':
        # Exponential Sine Squared Kernel function
        # hp1 == p : periodicity
        # hp2 == l : lengthscale
        dists = torch.cdist(X1.T, X2.T, p=2)
        K = torch.exp(
            (-2 * (torch.sin(torch.tensor(math.pi, device='cuda:0') * dists / hp1))**2) / hp2**2)

    else:
        raise ValueError(
            "Invalid kernel_type. Choose 'Gaussian' or 'ThinSpline'.")

    return K


class GPObservable:

    count = 0

    def __init__(self, d, ns, kernel_type='Gaussian', hp1=1.0, hp2=1.0, noise=2e-8, device='cuda:0'):
        # d is the dimensionality of the inputs
        # ns is the number of training samples
        self.device = torch.device(device)
        self.kernel_type = kernel_type  # KernelType
        # First kernel hyperparameter
        self.hp1 = torch.tensor(hp1, requires_grad=True, device=self.device)
        # Second kernel hyperparameter
        self.hp2 = torch.tensor(hp2, requires_grad=True, device=self.device)
        # Noise in Observations/Training Data
        self.noise = torch.tensor(
            noise, requires_grad=True, device=self.device)
        self.Kxx = torch.empty((ns, ns), device=self.device)
        self.invKxx = torch.empty((ns, ns), device=self.device)
        # target variable is always 1-dimensional, but with ns samples in training dataset
        self.y = torch.empty((ns, 1), device=self.device)
        GPObservable.count += 1

    def set_hyperparameters(self, hp1=None, hp2=None):
        if hp1 is not None:
            self.hp1 = torch.tensor(
                hp1, requires_grad=True, device=self.device)
        if hp2 is not None:
            self.hp2 = torch.tensor(
                hp2, requires_grad=True, device=self.device)

    def trainGP(self, Xtrain, ytrain):
        Xtrain = Xtrain.to(self.device)
        ytrain = ytrain.to(self.device)
        self.Xtrain = Xtrain
        self.y = ytrain

        self.Kxx = KernelFunction(
            Xtrain, Xtrain, kernel_type=self.kernel_type, hp1=self.hp1, hp2=self.hp2)
        # self.invKxx = torch.linalg.inv(self.Kxx + ((self.noise)**2)*torch.eye(self.Kxx.shape[0], device=self.device))
        self.invKxx = torch.cholesky_inverse(torch.linalg.cholesky(
            self.Kxx + ((self.noise)**2)*torch.eye(self.Kxx.shape[0], device=self.device)))

    def predictGP(self, Xq):
        Kqx = KernelFunction(
            Xq, self.Xtrain, kernel_type=self.kernel_type, hp1=self.hp1, hp2=self.hp2)
        Kqq = KernelFunction(
            Xq, Xq, kernel_type=self.kernel_type, hp1=self.hp1, hp2=self.hp2)
        mean = Kqx @ self.invKxx @ self.y
        CovMat = Kqq - Kqx @ self.invKxx @ torch.t(Kqx)
        return mean, CovMat

    def predictMean(self, Xq):
        Kqx = KernelFunction(
            Xq, self.Xtrain, kernel_type=self.kernel_type, hp1=self.hp1, hp2=self.hp2)
        return Kqx @ self.invKxx @ self.y   # returns the mean of the prediction

    def predictCov(self, Xq):
        Kqx = KernelFunction(
            Xq, self.Xtrain, kernel_type=self.kernel_type, hp1=self.hp1, hp2=self.hp2)
        Kqq = KernelFunction(
            Xq, Xq, kernel_type=self.kernel_type, hp1=self.hp1, hp2=self.hp2)
        return Kqq - Kqx @ self.invKxx @ torch.t(Kqx)

    def optimize_hyperparameters(self, max_iter=100, lr=0.01):
        """
        Optimizes the hyperparameters (hp1, hp2, noise) to maximize the log likelihood of the training data.

        Args:
            max_iter (int): Maximum number of optimization iterations.
            lr (float): Learning rate for the optimizer.
        """
        if not hasattr(self, 'Xtrain') or not hasattr(self, 'y'):
            raise ValueError(
                "Training data not found. Please call trainGP before optimizing hyperparameters.")

        # Define optimizer
        optimizer = torch.optim.Adam([self.hp1, self.hp2, self.noise], lr=lr)

        for i in range(max_iter):
            optimizer.zero_grad()

            # Compute kernel matrix with current hyperparameters
            Kxx = KernelFunction(
                self.Xtrain, self.Xtrain, kernel_type=self.kernel_type, hp1=self.hp1, hp2=self.hp2)
            Kxx += (self.noise**2) * \
                torch.eye(Kxx.shape[0], device=self.device)

            # Compute log likelihood
            invKxx = torch.linalg.inv(Kxx)
            y = self.y
            n = y.shape[0]
            log_det = torch.logdet(Kxx)
            ll = -0.5 * (y.t() @ invKxx @ y + log_det + n *
                         torch.log(torch.tensor(2 * torch.pi)))

            # Negate log likelihood to minimize
            loss = -ll.squeeze()
            loss.backward()
            optimizer.step()

            # Display progress
            # print(f"Iteration {i + 1}/{max_iter}, Log Likelihood: {-loss.item()}")

        self.hp1 = self.hp1
        self.hp2 = self.hp2
        self.noise = self.noise

        # Re-evaluate Training Covariance matrices for ease of use
        # This way, I don't have to call trainGP again after optimize_hyperparameters
        self.trainGP(self.Xtrain, self.y)

        # print(f"Optimization complete. Final Log Likelihood: {-loss.item()}")
        # print(f"Optimized hyperparameters: hp1={self.hp1.item()}, hp2={self.hp2.item()}, noise={self.noise.item()}")

    @classmethod
    def count_Observables(cls):
        return cls.count
    # idk what else

# Managing Class


class GPObservablesManager:

    def __init__(self):
        # Initialize a manager for handling multiple GPObservable objects
        self.observables = {}

    def add_observable(self, index, d, ns, kernel_type='Gaussian', hp1=1.0, hp2=1.0, noise=2e-6):
        # Adds a new Observable
        # Modify to add multiple observables in a single line with vector inputs
        if index in self.observables:
            raise ValueError(f'Observable with index {index} already exists.')
        self.observables[index] = GPObservable(
            d, ns, kernel_type, hp1, hp2, noise)

    def set_random_hyperparameters(self, seed=42, scale=1.0):
        torch.manual_seed(seed)
        for obs in self.observables.values():
            obs.hp1 = scale*torch.rand((1, 1))
            obs.hp2 = scale*torch.rand((1, 1))

    def train_observable(self, index, Xtrain, ytrain):
        # Train the specified Observable
        if index not in self.observables:
            raise ValueError(f'Observable with index {index} does not exist.')
        self.observables[index].trainGP(Xtrain, ytrain)

    def predict_mean(self, index, Xq):
        # predict the mean vector based on query predictor variable
        # Xq must be 2D torch.tensor of shape d-rows by nq columns, where nq is the number of query points
        # returns 2D torch.tensor of shape nq-rows by 1 column
        if index not in self.observables:
            raise ValueError(f'Observable with index {index} does not exist.')
        return self.observables[index].predictMean(Xq)

    def predict_covariance(self, index, Xq):
        # predict the covariance matrix based on query predictor variables
        if index not in self.observables:
            raise ValueError(f'Observable with index {index} does not exist.')
        return self.observables[index].predictCov(Xq)

    def optim_GP_hyperparams(self, max_iter=100, lr=0.01):
        # Optimize Hyperparameters of all GP Observables by Maximizing the Log-Likelihood
        # of available training data
        for obs in self.observables.values():
            obs.optimize_hyperparameters(max_iter, lr)

    def get_params(self, index):
        if index not in self.observables:
            raise ValueError(f'Observable with index {index} does not exist.')
        hp1 = self.observables[index].hp1
        hp2 = self.observables[index].hp2
        return torch.tensor([hp1, hp2])

    def get_all_params(self):
        if not self.observables:
            raise ValueError('No observables available in manager.')

        params = []
        for idx, obs in self.observables.items():
            params.append([obs.hp1, obs.hp2])

        return torch.tensor(params, dtype=torch.float32)

    def plot_observables(self, resolution=50, range_x=(-1, 1), range_y=(-1, 1)):
        """
        Generate surface plots for all 2D observables in the manager.

        Args:
            resolution (int): Number of points along each axis for the grid.
            range_x (tuple): Range of values for the first input dimension (min, max).
            range_y (tuple): Range of values for the second input dimension (min, max).

        Raises:
            ValueError: If any observable does not have a 2D input dimension.
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
            # ensure predictMean gets a 2-row, 'res*res'-column input
            grid_points = torch.tensor(
                np.vstack([X.ravel(), Y.ravel()]), dtype=torch.float32)

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


def getKoopmanMulti(manager, indices, Xall, nT):
    """
    Compute Koopman Matrices A & C using the manager for GPObservables.

    Args:
        manager (GPObservablesManager): Manager holding all GPObservable objects.
        indices (list): List of indices for observables to include.
        Xall (torch.Tensor): n x (N+1) matrix of state trajectory.
        nT (float): number of trajectories in training dataset

    Returns:
        A (torch.Tensor): p x p linear state transition matrix.
        C (torch.Tensor): n x p output matrix.
    """

    n = Xall.shape[0]       # dimensionality of original system
    N = (Xall.shape[1])//nT - 1  # Number of time steps in each trajectory
    p = len(indices)        # number of observables

    X = torch.cat([Xall[:, j*(N+1):j*(N+1)+N] for j in range(nT)],
                  dim=1)         # Data matrix from original system
    Xplus = torch.cat([Xall[:, j*(N+1)+1:j*(N+1)+N+1]
                      for j in range(nT)], dim=1)  # Time-shifted data matrix

    M = torch.empty((p, N*nT))
    Mplus = torch.empty((p, N*nT))
    for i in range(p):
        M[i, :] = torch.transpose(manager.predict_mean(i, X), dim0=0, dim1=-1)
        Mplus[i, :] = torch.transpose(
            manager.predict_mean(i, Xplus), dim0=0, dim1=-1)

    # Compute C(z) and A(z)
    M_pinv = torch.linalg.pinv(M)
    C = X @ M_pinv
    A = Mplus @ M_pinv

    return A, C


def getKoopman(manager, indices, Xall, nT, stateAug=False):
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

    M = torch.empty((p, N*nT))
    Mplus = torch.empty((p, N*nT))
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
