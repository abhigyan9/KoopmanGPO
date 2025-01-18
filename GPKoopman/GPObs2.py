
import torch
import math
from matplotlib import pyplot as plt
import numpy as np

# Define individual kernel functions


def GaussianKernel(X1, X2, hp1, hp2):
    dists = torch.cdist(X1.T, X2.T, p=2)**2
    return hp1 * torch.exp(-dists / (2 * hp2**2))


def ThinSplineKernel(X1, X2, hp1, hp2):
    dists = torch.cdist(X1.T, X2.T, p=2)
    epsilon = 1e-8
    return hp1 * ((dists/hp2)**2) * torch.log((dists / hp2) + epsilon)


def InverseQuadraticKernel(X1, X2, hp1, hp2):
    dists = torch.cdist(X1.T, X2.T, p=2)**2
    return hp1 * (1 / (1 + (dists / (hp2**2))))


def CosineKernel(X1, X2, hp1, hp2):
    dists = torch.cdist(X1.T, X2.T, p=2)
    return hp1 * torch.cos(math.pi * dists / hp2) ** 2


def ExpSineSqrKernel(X1, X2, hp1, hp2):
    dists = torch.cdist(X1.T, X2.T, p=2)
    hp2 = torch.clamp(hp2, min=1e-2)
    epsilon = 1e-4
    return hp1 * torch.exp(epsilon - 0.5 * ((torch.sin(math.pi * dists / hp1)) ** 2) / hp2)
    # return torch.exp((-2 * (torch.sin(math.pi * dists / hp1))**2) / hp2**2)


# Dictionary mapping kernel names to functions
KERNEL_FUNCTIONS = {
    'Gaussian': GaussianKernel,
    'ThinSpline': ThinSplineKernel,
    'InverseQuadratic': InverseQuadraticKernel,
    'ExpSineSqr': ExpSineSqrKernel,
    'Cosine': CosineKernel
}


def KernelFunction(X1, X2=None, kernel_types=['Gaussian'], hp1_list=None, hp2_list=None, combination='sum'):
    """
    Computes a kernel matrix using a combination of multiple kernels.

    Args:
        X1 (Tensor): First set of points.
        X2 (Tensor, optional): Second set of points. Defaults to None.
        kernel_types (list): List of kernel function names.
        hp1_list (list, optional): List of first hyperparameters corresponding to each kernel. Defaults to None.
        hp2_list (list, optional): List of second hyperparameters corresponding to each kernel. Defaults to None.
        combination (str): How to combine kernels ('sum' or 'product'). Defaults to 'sum'.

    Returns:
        Tensor: Kernel matrix.
    """
    if not torch.is_tensor(X1):
        raise TypeError(f"Expected X1 to be a torch.Tensor, but got {
                        type(X1).__name__}.")

    n1 = X1.shape[0]
    if X2 is not None:
        if not torch.is_tensor(X2):
            raise TypeError(f"Expected X2 to be a torch.Tensor, but got {
                            type(X2).__name__}.")
        n2 = X2.shape[0]
        if n1 != n2:
            raise ValueError(
                f'Mismatch in input dimensions. Tensor 1 and 2 have different rows: {n1}, {n2}.')
    else:
        X2 = torch.zeros(n1, 1)

    l1, l2 = X1.shape[1], X2.shape[1]

    cuda0 = torch.device('cuda:0')
    X1, X2 = X1.to(cuda0), X2.to(cuda0)

    # Ensure hyperparameters are lists and match the number of kernels
    if hp1_list is None:
        hp1_list = [torch.tensor(1.0, device=cuda0)] * len(kernel_types)
    if hp2_list is None:
        hp2_list = [torch.tensor(1.0, device=cuda0)] * len(kernel_types)

    if len(hp1_list) != len(kernel_types) or len(hp2_list) != len(kernel_types):
        raise ValueError(
            "Number of hyperparameters must match the number of kernel types.")

    # Compute combined kernel
    if combination == 'sum':
        K = sum(KERNEL_FUNCTIONS[kernel](X1, X2, hp1, hp2)
                for kernel, hp1, hp2 in zip(kernel_types, hp1_list, hp2_list))
    elif combination == 'product':
        K = torch.ones((l1, l2), device=cuda0)  # Initialize product kernel
        for kernel, hp1, hp2 in zip(kernel_types, hp1_list, hp2_list):
            K *= KERNEL_FUNCTIONS[kernel](X1, X2, hp1, hp2)
    else:
        raise ValueError(
            "Invalid combination method. Choose 'sum' or 'product'.")

    return K


class GPObservable:
    count = 0

    def __init__(self, d, ns, kernel_types=['Gaussian'], hp1_list=None, hp2_list=None,
                 noise=2e-8, combination='sum', device='cuda:0'):
        """
        Gaussian Process Observable with customizable kernel functions.

        Args:
            d (int): Dimensionality of the inputs.
            ns (int): Number of training samples.
            kernel_types (list): List of kernel names.
            hp1_list (list): List of first hyperparameters for each kernel.
            hp2_list (list): List of second hyperparameters for each kernel.
            noise (float): Observation noise.
            combination (str): Kernel combination method ('sum' or 'product').
            device (str): Compute device ('cuda:0' or 'cpu').
        """
        self.device = torch.device(device)
        self.kernel_types = kernel_types  # List of kernel functions
        self.combination = combination  # How kernels are combined

        # Ensure hyperparameter lists are set correctly
        self.hp1_list = hp1_list if hp1_list is not None else [torch.tensor(
            1.0, requires_grad=True, device=self.device)] * len(kernel_types)
        self.hp2_list = hp2_list if hp2_list is not None else [torch.tensor(
            1.0, requires_grad=True, device=self.device)] * len(kernel_types)

        if len(self.hp1_list) != len(kernel_types) or len(self.hp2_list) != len(kernel_types):
            raise ValueError(
                "Number of hyperparameters must match the number of kernel types.")

        self.noise = torch.tensor(
            noise, requires_grad=True, device=self.device)
        self.Kxx = torch.empty((ns, ns), device=self.device)
        self.invKxx = torch.empty((ns, ns), device=self.device)
        self.y = torch.empty((ns, 1), device=self.device)  # Target values

        GPObservable.count += 1

    def set_hyperparameters(self, hp1_list=None, hp2_list=None):
        if hp1_list is not None:
            self.hp1_list = [torch.tensor(
                hp, requires_grad=True, device=self.device) for hp in hp1_list]
        if hp2_list is not None:
            self.hp2_list = [torch.tensor(
                hp, requires_grad=True, device=self.device) for hp in hp2_list]

    def trainGP(self, Xtrain, ytrain):
        Xtrain = Xtrain.to(self.device)
        ytrain = ytrain.to(self.device)
        self.Xtrain = Xtrain
        self.y = ytrain

        self.Kxx = KernelFunction(Xtrain, Xtrain, kernel_types=self.kernel_types,
                                  hp1_list=self.hp1_list, hp2_list=self.hp2_list,
                                  combination=self.combination)
        # self.Kxx += 1e-6 * torch.eye(self.Kxx.shape[0], device=self.device)
        self.invKxx = torch.linalg.inv(
            self.Kxx + ((self.noise)**2)*torch.eye(self.Kxx.shape[0], device=self.device))
        # self.invKxx = torch.cholesky_inverse(torch.linalg.cholesky(
        #    self.Kxx + ((self.noise)**2) * torch.eye(self.Kxx.shape[0], device=self.device)))

        try:
            L = torch.linalg.cholesky(
                self.Kxx + ((self.noise)**2)*torch.eye(self.Kxx.shape[0], device=self.device))
            self.invKxx = torch.cholesky_inverse(L)
        except RuntimeError:
            U, S, V = torch.linalg.svd(
                self.Kxx + ((self.noise)**2)*torch.eye(self.Kxx.shape[0], device=self.device))
            S_inv = torch.diag(torch.where(
                S > 1e-6, 1.0 / S, torch.tensor(0.0, device=self.device)))
            self.invKxx = V.T @ S_inv @ U.T

    def predictGP(self, Xq):
        Kqx = KernelFunction(Xq, self.Xtrain, kernel_types=self.kernel_types,
                             hp1_list=self.hp1_list, hp2_list=self.hp2_list,
                             combination=self.combination)
        Kqq = KernelFunction(Xq, Xq, kernel_types=self.kernel_types,
                             hp1_list=self.hp1_list, hp2_list=self.hp2_list,
                             combination=self.combination)
        mean = Kqx @ self.invKxx @ self.y
        CovMat = Kqq - Kqx @ self.invKxx @ torch.t(Kqx)
        return mean, CovMat

    def predictMean(self, Xq):
        Kqx = KernelFunction(Xq, self.Xtrain, kernel_types=self.kernel_types,
                             hp1_list=self.hp1_list, hp2_list=self.hp2_list,
                             combination=self.combination)
        return Kqx @ self.invKxx @ self.y

    def predictCov(self, Xq):
        Kqx = KernelFunction(Xq, self.Xtrain, kernel_types=self.kernel_types,
                             hp1_list=self.hp1_list, hp2_list=self.hp2_list,
                             combination=self.combination)
        Kqq = KernelFunction(Xq, Xq, kernel_types=self.kernel_types,
                             hp1_list=self.hp1_list, hp2_list=self.hp2_list,
                             combination=self.combination)
        return Kqq - Kqx @ self.invKxx @ torch.t(Kqx)

    def optimize_hyperparameters(self, max_iter=100, lr=0.01):
        if not hasattr(self, 'Xtrain') or not hasattr(self, 'y'):
            raise ValueError(
                "Training data not found. Please call trainGP before optimizing hyperparameters.")

        optimizer = torch.optim.Adam(
            self.hp1_list + self.hp2_list + [self.noise], lr=lr)

        for _ in range(max_iter):
            optimizer.zero_grad()
            Kxx = KernelFunction(self.Xtrain, self.Xtrain, kernel_types=self.kernel_types,
                                 hp1_list=self.hp1_list, hp2_list=self.hp2_list,
                                 combination=self.combination)
            Kxx += (self.noise**2) * \
                torch.eye(Kxx.shape[0], device=self.device)

            invKxx = torch.linalg.inv(Kxx)
            y = self.y
            log_det = torch.logdet(Kxx)
            ll = -0.5 * (y.t() @ invKxx @ y + log_det +
                         y.shape[0] * torch.log(torch.tensor(2 * torch.pi)))

            loss = -ll.squeeze()
            loss.backward()
            optimizer.step()

        # Update with optimized hyperparameters
        self.trainGP(self.Xtrain, self.y)

    @classmethod
    def count_Observables(cls):
        return cls.count


class GPObservablesManager:
    def __init__(self):
        self.observables = {}

    def add_observable(self, index, d, ns, kernel_types=['Gaussian'], hp1_list=None, hp2_list=None, noise=2e-6, combination='sum'):
        if index in self.observables:
            raise ValueError(f'Observable with index {index} already exists.')
        self.observables[index] = GPObservable(
            d, ns, kernel_types, hp1_list, hp2_list, noise, combination)

    def set_random_hyperparameters(self, seed=42, scale=1.0):
        """
        Assigns random hyperparameters (hp1 and hp2) to all observables.

        Args:
            seed (int): Seed for reproducibility.
            scale (float): Scaling factor for the generated hyperparameters.
        """
        torch.manual_seed(seed)  # Set random seed for reproducibility

        for obs in self.observables.values():
            num_kernels = len(obs.kernel_types)

            # Assign random values to hp1 and hp2 lists
            obs.hp1_list = [
                scale * torch.rand(1, device=obs.device, requires_grad=True) for _ in range(num_kernels)]
            obs.hp2_list = [
                scale * torch.rand(1, device=obs.device, requires_grad=True) for _ in range(num_kernels)]

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

    def optimize_hyperparameters(self, max_iter=100, lr=0.01):
        for obs in self.observables.values():
            obs.optimize_hyperparameters(max_iter, lr)

    def get_params(self, index):
        if index not in self.observables:
            raise ValueError(f'Observable with index {index} does not exist.')
        return torch.tensor(self.observables[index].hp1_list + self.observables[index].hp2_list)

    def get_all_params(self):
        if not self.observables:
            raise ValueError('No observables available in manager.')
        return torch.vstack([self.get_params(idx) for idx in self.observables])

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
