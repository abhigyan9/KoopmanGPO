from typing import Literal
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import GPKoopman as gpk
import warnings

warnings.filterwarnings(
    "ignore",
    message=(
        r"The use of `x\.T` on tensors of dimension other than 2 "
        r"to reverse their shape is deprecated.*"
    ),
    category=UserWarning
)

# This script runs vanilla iGPK to analyze trends and behavior of the algorithm

############################
#   FUNCTION DEFINITIONS   #
############################


# Cost Function
def get_cost_AC(Z, X, Xplus, Xtrain, manager, nT=1, lambda1=1.0, lambda2=1.0, lambda3=1.0):
    """
    Computes the cost function using a single differentiable GP forward pass per observable,
    merging the training and prediction steps by passing Z[:, i] directly to the forward method.

    Args:
        Z: Tensor of shape (r**n, p), decision variable (requires grad).
        X: Tensor of shape (n, nT*N), dataset of N steps per trajectory.
        Xplus: Tensor of shape (n, nT*N), time-shifted dataset.
        Xtrain: Tensor of shape (n, r**n), gridpoints for training.
        manager: GPObservablesManager.
        nT: Number of trajectories.
        lambda1: Weighting for NLPD.
        lambda2: Weighting for linearity enforcement.
        lambda3: Weighting for prediction error minimization.
    """
    N = X.shape[1] // nT    # Number of time steps per trajectory
    p = Z.shape[1]          # Number of observables
    l = Z.shape[0] // nT    # Decision horizon
    n = X.shape[0]          # State dimension

    # For each observable, call forward once on the full dataset X (and Xplus)
    M = torch.empty((p, N * nT), device=Z.device)
    cov_all = [None] * p  # store full covariance matrices for X
    Mplus = torch.empty((p, N * nT), device=Z.device)
    # cov_all_plus = [None] * p  # store full covariance matrices for Xplus

    for i in range(p):
        mean_i, cov_i = manager.observables[i].forward(X, Z[:, i])
        M[i, :] = torch.transpose(mean_i, 0, -1)
        cov_all[i] = cov_i

        mean_plus_i, _ = manager.observables[i].forward(Xplus, Z[:, i])
        Mplus[i, :] = torch.transpose(mean_plus_i, 0, -1)

    # Compute the pseudo-inverse lifting operator and the corresponding matrices Cz and Az.
    M_pinv = torch.linalg.pinv(M)
    Cz = X @ M_pinv
    Az = Mplus @ M_pinv

    # Cost Term 1: Negative Log Predictive Density (NLPD)
    NormNLPD = 0.0
    if not math.isclose(lambda1, 0):    # NLPD Cost
        for j in range(nT):
            TrajNLPD = 0.0
            # Define the number of time steps for NLPD computation:
            num_steps = N - 2 - l
            vz_k = torch.empty((p, num_steps), device=X.device)
            for i in range(p):
                # For trajectory j, determine the indices for the NLPD slice.
                start = j * N + l + 1
                end = (j + 1) * N - 1  # end is exclusive
                # Extract the covariance for the slice from the precomputed full covariance.
                cov_sub = cov_all[i][start:end, start:end]
                # Get the diagonal elements (predictive variances) and clamp them.
                vz_k[i, :] = torch.clamp(torch.diag(cov_sub), min=1e-3)
            for k in range(num_steps):
                vx_next = torch.abs(torch.diag(
                    Cz @ Az @ torch.diag(vz_k[:, k]) @ Az.T @ Cz.T).view(n, 1))
                error_term = ((X[:, j * N + l + 1 + k + 1] - Cz @
                              Az @ M[:, j * N + l + 1 + k]) ** 2) / vx_next
                log_term = torch.log(vx_next)
                TrajNLPD += torch.sum(error_term + log_term)
            NormNLPD += TrajNLPD

    # Cost Term 2: Linearity Enforcement
    NormLEP = 0.0
    if not math.isclose(lambda2, 0):    # Linearity Cost
        for j in range(nT):
            TrajLEP = 0.0
            for k in range(l - 1):
                lin_error = torch.transpose(
                    Z[j * l + k + 1, :], 0, -1) - Az @ torch.transpose(Z[j * l + k, :], 0, -1)
                TrajLEP += torch.norm(lin_error)
            NormLEP += TrajLEP

    # Cost Term 3: Prediction Error Minimization
    NormPEM = 0.0
    if not math.isclose(lambda3, 0):    # Prediction Error
        for j in range(nT):
            TrajPEM = 0.0
            for k in range(N - 1):
                pred_error = X[:, j * N + (k + 1)] - Cz @ (
                    torch.linalg.matrix_power(Az, k + 1)) @ M[:, j * N]
                TrajPEM += torch.norm(pred_error)
            NormPEM += TrajPEM

    cost = (lambda1 * NormNLPD / ((N - l) * nT)) + (lambda2 *
                                                    NormLEP / (l * nT)) + (lambda3 * NormPEM / (N * nT))
    return cost


# Load trajectory data from files
def load_data(system_name: str, nTrain, nTest, Nmax=None):
    """
    Load data using particular system name

    Returns:
        X
        Xplus
        Xall
        num_trajectories
        N
        n
        ts
        ICsetTrain
        ICsetTest
    """
    data = torch.load(f"Data/DataAuto_{system_name}.pt", weights_only=True)

    # Shape: (num_trajectories, state_dim, num_steps)
    SimData = data["trajectories"].float()
    ts = data["sample_time"]
    num_trajectories = data["num_trajectories"]
    N = data["num_steps"]
    n = SimData.shape[1]

    nTrain, nTest = math.floor(
        num_trajectories * nTrain), math.floor(num_trajectories * nTest)

    # Clip Training Data steps
    if Nmax is not None:
        N, SimData = Nmax, SimData[:, :, :Nmax+1]

    Xall = torch.cat([SimData[j, :, :] for j in range(nTrain)],
                     dim=1)      # Concatenated total matrix
    X = torch.cat([SimData[j, :, 0:N] for j in range(nTrain)],
                  dim=1)       # Concatenated Data matrix
    # Time-shifted Data matrix
    Xplus = torch.cat([SimData[j, :, 1:] for j in range(nTrain)], dim=1)

    ICsetTrain = torch.cat([SimData[j, :, 0].view(n, 1) for j in range(
        nTrain)], dim=1)    # Random IC set for training
    ICsetTest = torch.cat([SimData[j, :, 0].view(n, 1) for j in range(
        nTrain, nTrain + nTest)], dim=1)  # Random IC set for testing

    return X, Xplus, Xall, num_trajectories, N, n, ts, ICsetTrain, ICsetTest


# Implement iGPK algorithm
def iGPK_vanilla(system_name: str, nTrain: float, p: int, l: int, max_iter: int, opt_algo: Literal['Adam', 'SGD']):
    """
    Returns GP-Koopman model using the vanilla iGPK algorithm

    Args:
        system_name: String, system to find GP-Koopman model of
        nTrain: Float, fraction of trajectories to utilize for training
        p: Integer, number of lifted states
        l: Integer, horizon for virtual targets
        max_iter: Integer, maximum iterations for optimization
        opt_algo: String, Choice of optimization algorithm - Adam or SGD

    Returns:
        ObsManager: GPObservablesManager, learned GPOs
        A: Tensor of shape (p,p), learned linear state transition matrix
        C: Tensor of shape (n,p), learned output matrix
    """

    X, Xplus, Xall, num_trajectories, N, n, _, _, _ = load_data(
        system_name, nTrain, 1-nTrain, Nmax=150)
    nTrain = math.floor(nTrain * num_trajectories)
    Xtrain = torch.cat([X[:, j*N:j*N+l] for j in range(nTrain)], dim=1)
    Z = torch.nn.Parameter(torch.rand(Xtrain.shape[1], p, device='cuda:0'))
    X, Xplus, Xall, Xtrain = X.to(device='cuda:0'), Xplus.to(
        device='cuda:0'), Xall.to(device='cuda:0'), Xtrain.to(device='cuda:0')
    ObsManager = gpk.GPObservablesManager()
    for i in range(p):  # add observables
        ObsManager.add_observable(
            index=i, d=n, ns=l*nTrain, kernel_types=['Gaussian'], combination='sum', noise=1e-4, m=500)
    for i in range(p):  # initial training of observables
        ObsManager.train_observable(i, Xtrain, Z[:, i])
    ObsManager.set_random_hyperparameters(scale=[1., 0.01, None])

    cost_history, iter, count_insignificant = [], 0, 0
    lambda1, lambda2, lambda3 = 1e-2, 10., 0.

    if opt_algo == 'Adam':  # Non-Stochastic | Faster Convergence, lesser exploration
        optimizer = torch.optim.Adam([Z], lr=0.01)
    elif opt_algo == 'SGD':  # More Exploration
        optimizer = torch.optim.SGD([Z], lr=0.002, momentum=0.9, nesterov=True)
    else:
        raise ValueError('Only supports Adam and SGD')

    while iter < max_iter:  # Optimization loop
        optimizer.zero_grad()  # Clear gradients
        cost = get_cost_AC(Z, X, Xplus, Xtrain, ObsManager, nT=nTrain,
                           lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)   # compute cost
        cost_history.append(cost.item())    # add to cost history
        cost.backward()    # backpropagate
        optimizer.step()    # gradient descent step
        if iter % 10 == 0:
            print(f"Iter: {iter}/{max_iter} || Cost: {cost.item()}")

        # Increment iteration
        iter += 1

    optimal_Z = Z.detach()
    print(f'Optimization Completed')
    print(f'Final Cost: {cost.item()}')

    for i in range(p):
        # train GP Observables with Optimal Z outputs
        ObsManager.train_observable(i, Xtrain, optimal_Z[:, i])

    # Optimize Kernel hyperparameters for Optimal training data
    ObsManager.optimize_hyperparameters(opt_mu=False, opt_sigma=True)
    print(f'GPO Hyperparameters have been optimized.')
    ObsManager.print_parameters()

    ObsList = [i for i in range(p)]
    A, C = gpk.getKoopman(ObsManager, ObsList, Xall, nTrain, stateAug=False)

    A, C = A.detach().to(device='cpu'), C.detach().to(device='cpu')
    return ObsManager, A, C


#############################
#   1. EXPERIMENTAL SETUP   #
#############################
SYSTEM_NAME = 'Unforced Duffing'
P, L, MAX_ITER = 30, 10, 500
OPT_ALGO = 'Adam'

TRAINING_FRACTIONS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

results = {}


##########################
#   2. RUN EXPERIMENTS   #
##########################

for frac in TRAINING_FRACTIONS:  # iterate over fractions
    print(f"\n=== Training with {int(frac*100)}% of trajectories ===")

    # run vanilla iGPK:
    ObsMgr, A, C = iGPK_vanilla(
        system_name=SYSTEM_NAME,
        nTrain=frac,
        p=P,
        l=L,
        max_iter=MAX_ITER,
        opt_algo=OPT_ALGO
    )

    # compute eigenvalues of A
    eigs = torch.linalg.eigvals(A).cpu().numpy()
    results[frac] = eigs


###########################
#   3. PLOTTING RESULTS   #
###########################

# (3a) Magnitude vs. index
plt.figure(figsize=(8, 4))
for frac, eigs in results.items():
    # compute magnitude and sort indices by descending mag
    mags = np.abs(eigs)
    order = np.argsort(-mags)
    mags_sorted = mags[order]

    plt.plot(
        np.arange(len(mags_sorted)),
        mags_sorted,
        marker='o',
        label=f"{int(frac*100)}"
    )
plt.title("Eigenvalue Magnitudes (sorted) vs. Index")
plt.xlabel("Sorted eigenvalue index")
plt.ylabel("Magnitude |λ|")
plt.legend(title="Trajectories")
plt.grid(True)

plt.figure(figsize=(8, 4))
for frac, eigs in results.items():
    # same ordering:
    mags = np.abs(eigs)
    order = np.argsort(-mags)
    # compute phase then sort
    phase_sorted = np.arctan2(eigs.imag, eigs.real)[order]

    plt.plot(
        np.arange(len(phase_sorted)),
        phase_sorted,
        marker='o',
        label=f"{int(frac*100)}"
    )
plt.title("Eigenvalue Phase (sorted by magnitude) vs. Index")
plt.xlabel("Sorted eigenvalue index")
plt.ylabel("Phase arg(λ) [rad]")
plt.legend(title="Trajectories")
plt.grid(True)

plt.show()
