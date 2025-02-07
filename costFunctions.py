import torch

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
    Computes the cost function as defined in the Word doc, with different GP Kernel hyperparameters for
    each observable.

    Args:
        Z: Tensor of shape (r**n, p), decision variable, required grad
        X: Tensor of shape (n, nT*N), dataset of N steps from all trajectories
        Xall: Tensor of shape (n, nT*(N+1)), complete training dataset
        Xtrain: Tensor of shape (n, r**n), flattened set of gridpoints for training GPOs
        manager: Object of class GPObservablesManager (manager for all Gaussian Process based Observable functions)
        nT: float, number of trajectories in training dataset
        lambda1: float, Weighting for NLPD
        lambda2: float, Weightining for Linearity enforcement in lifted state
    """

    N = (X.shape[1])//nT    # Number of time steps in each trajectory
    p = Z.shape[1]          # Number of Observables
    l = Z.shape[0]//nT      # Decision Horizon
    n = X.shape[0]          # Dimensionality of original system

    for i in range(p):
        manager.train_observable(i, Xtrain, Z[:, i])

    # Lifting X and Xplus to higher dimension using trained GPOs
    M = torch.empty((p, N*nT))
    Mplus = torch.empty((p, N*nT))

    # Mall = torch.empty((p,(N+1)*nT))
    for i in range(p):
        M[i, :] = torch.transpose(manager.predict_mean(i, X), dim0=0, dim1=-1)
        Mplus[i, :] = torch.transpose(
            manager.predict_mean(i, Xplus), dim0=0, dim1=-1)

    # Compute C(z) and A(z)
    M_pinv = torch.linalg.pinv(M)
    Cz = X @ M_pinv
    Az = Mplus @ M_pinv

    # Cost term 1: Negative Log Predictive Density (penalize uncertainty and confidently wrong models)
    NormNLPD = 0.0
    for j in range(nT):
        TrajNLPD = 0.0
        vz_k = torch.empty((p, N-1 - l), device=M.device)
        for i in range(p):
            vz_k[i, :] = torch.diag(manager.predict_covariance(
                i, X[:, j*N + l:(j+1)*N-1])).view(1, N-1-l)

        for k in range(N-1-l):
            vx_next = torch.diag(
                Cz @ Az @ torch.diag(vz_k[:, k]) @ Az.T @ Cz.T).view(n, 1)
            TrajNLPD += torch.norm(((X[:, j*N + k+1] - Cz @ Az @ M[:, j*N + k]) ** 2 / (
                lambda3 + vx_next) + torch.log(lambda3 + vx_next)))

        NormNLPD += TrajNLPD

    # Cost Term 2: Linearity Enforcement
    NormLEP = 0.0
    for j in range(nT):
        TrajLEP = 0.0
        for k in range(l-1):
            lin_error = torch.transpose(
                Z[j*l + k + 1, :], dim0=0, dim1=-1) - Az @ torch.transpose(Z[j*l + k, :], dim0=0, dim1=-1)
            TrajLEP += torch.norm(lin_error)
        NormLEP += TrajLEP

    # Weighted sum of terms
    cost = (lambda1 * NormNLPD / ((N-l) * nT)) + (lambda2 * NormLEP / (l * nT))
    return cost
