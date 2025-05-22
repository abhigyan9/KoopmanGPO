import torch
import math
# Employs state-augmentation and uses multi-step PEM and linearity costs


def get_cost(Z, X, Xplus, Xtrain, manager, nT=1, lambda1=1.0, lambda2=1.0, lambda3=1.0):
    """
    Computes the cost function as defined in the Word doc, with different GP Kernel hyperparameters for
    each observable.

    Args:
        Z: Tensor of shape (r**n, p), decision variable, required grad
        X: Tensor of shape (n, nT*N), dataset of N steps from all trajectories
        Xall: Tensor of shape (n, nT*(N+1)), complete training dataset
        Xtrain: Tensor of shape (n, r**n), flattened set of gridpoints for training GPOs
        manager: Object of class GPObservablesManager (manager for all Gaussian Process based Observable functions)
        nT: float, number of trajectories in training dataset
        lambda1: float, Weighting for prediction error minimization term
        lambda2: float, Weightining for Reconstruction Error penalty term
    """

    N = (X.shape[1])//nT    # Number of time steps in each trajectory
    p = Z.shape[1]          # Number of Observables
    l = Z.shape[0]//nT      # Decision Horizon
    n = X.shape[0]          # Dimensionality of original system

    for i in range(p):
        manager.train_observable(i, Xtrain, Z[:, i])

    # For current definition of GPOs
    # Training: Xtrain = dimensions x samples
    # Training: Ytrain = samples x (dimensions=1)
    # Prediction: Xquery = dimensions x num-query = Input
    # Prediction: Yquery = num-query x (dimensions=1) = Output

    # Lifting X and Xplus to higher dimension using trained GPOs
    M = torch.empty((p, N*nT))
    Mplus = torch.empty((p, N*nT))

    # Mall = torch.empty((p,(N+1)*nT))
    for i in range(p):
        M[i, :] = torch.transpose(manager.predict_mean(i, X), dim0=0, dim1=-1)
        Mplus[i, :] = torch.transpose(
            manager.predict_mean(i, Xplus), dim0=0, dim1=-1)

    # Compute C(z) and A(z)
    Mfull = torch.vstack((X, M))
    Mplusfull = torch.vstack((Xplus, Mplus))

    M_pinv = torch.linalg.pinv(Mfull)
    # Cz = X @ M_pinv
    Az = Mplusfull @ M_pinv
    C = torch.zeros((n, n+p))
    for i in range(n):
        C[i, i] = 1.

    # Cost term 1: Multi-Trajectory Prediction Error Minimization
    NormPEM = 0.0
    for j in range(nT):
        TrajPEM = 0.0
        for k in range(N - 1):
            # multi-step at X (with Cz)
            pred_error = X[:, j*N + (k+1)] - \
                C @ (torch.linalg.matrix_power(Az, k+1)) @ Mfull[:, j*N]
            TrajPEM += torch.norm(pred_error)
        NormPEM += TrajPEM

    # Linearity Enforcement
    NormLEP = 0.0
    for j in range(nT):
        TrajLEP = 0.0
        for k in range(l-1):
            Zk = torch.vstack(
                [X[:, j*N + k].view(n, 1), torch.transpose(Z[j*l + k, :], dim0=0, dim1=-1).view(p, 1)])
            Zkplus = torch.vstack(
                [X[:, j*N + k + 1].view(n, 1), torch.transpose(Z[j*l + k + 1, :], dim0=0, dim1=-1).view(p, 1)])
            lin_error = Zkplus - Az @ Zk
            TrajLEP += torch.norm(lin_error)
        NormLEP += TrajLEP

    # Weighted sum of terms
    cost = (lambda1 * NormPEM / (N * nT)) + (lambda2 * NormLEP / (l * nT))
    return cost


# No State-Augmentation and uses NLPD and linearity costs
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
    M = torch.empty((p, N * nT), device=X.device)
    cov_all = [None] * p  # store full covariance matrices for X
    Mplus = torch.empty((p, N * nT), device=X.device)
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
    if not math.isclose(lambda1, 0):
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
    if not math.isclose(lambda2, 0):
        for j in range(nT):
            TrajLEP = 0.0
            for k in range(l - 1):
                lin_error = torch.transpose(
                    Z[j * l + k + 1, :], 0, -1) - Az @ torch.transpose(Z[j * l + k, :], 0, -1)
                TrajLEP += torch.norm(lin_error)
            NormLEP += TrajLEP

    # Cost Term 3: Prediction Error Minimization
    NormPEM = 0.0
    if not math.isclose(lambda3, 0):
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
