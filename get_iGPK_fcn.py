## --- IMPORTS --- ###
import GPKoopman as gpk
import torch
import matplotlib.pyplot as plt
import time
import math
# from botorch.models.transforms import Normalize, Standardize
# from botorch.fit import fit_gpytorch_mll
# from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
# # from botorch.acquisition.monte_carlo import qLogExpectedImprovement
# from botorch.acquisition import qLogExpectedImprovement
# from botorch.sampling import SobolQMCNormalSampler
# from botorch.optim import optimize_acqf

# from botorch.models import SingleTaskGP
# from botorch.acquisition import qLogNoisyExpectedImprovement
# from gpytorch.kernels import ScaleKernel, MaternKernel
# from gpytorch.priors import GammaPrior, LogNormalPrior


## --- COST FUNCTION --- ##


def get_cost_simple(Z, X, Xplus, manager, nT=1, lambda1=1.0, lambda2=1.0, lambda3=1.0):
    """
    Computes the cost function using a single differentiable GP forward pass per observable,
    merging the training and prediction steps by passing Z[:, i] directly to the forward method.

    Args:
        Z: Tensor of shape (nT*l, p), decision variable (requires grad).
        X: Tensor of shape (n, nT*N), dataset of N steps per trajectory.
        Xplus: Tensor of shape (n, nT*N), time-shifted dataset.
        Xtrain: Tensor of shape (n, r**n), gridpoints for training.
        manager: GPObservablesManager.
        nT: Number of trajectories.
        lambda1: Weighting for multi-variate NLPD
        lambda2: Weighting for Lifting Accuracy (Bhattacharyya Distance)
        lambda3: Weighting for Reconstruction
    """
    N = X.shape[1] // nT    # Number of time steps per trajectory
    p = Z.shape[1]          # Number of observables
    l = Z.shape[0] // nT    # Decision horizon
    # n = X.shape[0]          # State dimension

    # For each observable, call forward once on the full dataset X (and Xplus)
    M = torch.empty((p, N * nT), device=X.device)
    Mplus = torch.empty((p, N * nT), device=X.device)
    # diag_all = torch.empty((p, N * nT), device=X.device)
    # diag_all_plus = torch.empty((p, N * nT), device=X.device)
    # cov_all_plus = [None] * p  # store full covariance matrices for Xplus

    for i in range(p):
        mean_i, _ = manager.observables[i].forward(X, Z[:, i])
        M[i, :] = torch.transpose(mean_i, 0, -1)
        # diag_all[i] = torch.clamp(torch.diagonal(cov_i), min=1e-3)

        mean_plus_i, _ = manager.observables[i].forward(
            Xplus, Z[:, i])
        Mplus[i, :] = torch.transpose(mean_plus_i, 0, -1)
        # diag_all_plus[i] = torch.clamp(torch.diagonal(cov_i_plus), min=1e-3)

    # Compute the pseudo-inverse lifting operator and the corresponding matrices Cz and Az.

    try:
        L = torch.linalg.cholesky(M @ M.mT)
        M_pinv = torch.cholesky_solve(M.mT, L)
    except RuntimeError:
        M_pinv = torch.linalg.pinv(M)

    M_pinvM = M_pinv @ M

    cost1 = torch.linalg.matrix_norm(Mplus - (Mplus @ M_pinvM))
    cost2 = torch.linalg.matrix_norm(X - (X @ M_pinvM))

    return (lambda1 * cost1) + (lambda2 * cost2)


def get_cost_ACnew2(Z, X, Xplus, manager, nT=1, lambda1=1.0, lambda2=1.0, lambda3=1.0):
    """
    Computes the cost function using a single differentiable GP forward pass per observable,
    merging the training and prediction steps by passing Z[:, i] directly to the forward method.

    Args:
        Z: Tensor of shape (nT*l, p), decision variable (requires grad).
        X: Tensor of shape (n, nT*N), dataset of N steps per trajectory.
        Xplus: Tensor of shape (n, nT*N), time-shifted dataset.
        Xtrain: Tensor of shape (n, r**n), gridpoints for training.
        manager: GPObservablesManager.
        nT: Number of trajectories.
        lambda1: Weighting for multi-variate NLPD
        lambda2: Weighting for Lifting Accuracy (Bhattacharyya Distance)
        lambda3: Weighting for Reconstruction
    """
    N = X.shape[1] // nT    # Number of time steps per trajectory
    p = Z.shape[1]          # Number of observables
    l = Z.shape[0] // nT    # Decision horizon
    # n = X.shape[0]          # State dimension

    # For each observable, call forward once on the full dataset X (and Xplus)
    M = torch.empty((p, N * nT), device=X.device)
    Mplus = torch.empty((p, N * nT), device=X.device)
    diag_all = torch.empty((p, N * nT), device=X.device)
    diag_all_plus = torch.empty((p, N * nT), device=X.device)
    # cov_all_plus = [None] * p  # store full covariance matrices for Xplus

    for i in range(p):
        mean_i, cov_i = manager.observables[i].forward(X, Z[:, i])
        M[i, :] = torch.transpose(mean_i, 0, -1)
        diag_all[i] = torch.clamp(torch.diagonal(cov_i), min=1e-3)

        mean_plus_i, cov_i_plus = manager.observables[i].forward(
            Xplus, Z[:, i])
        Mplus[i, :] = torch.transpose(mean_plus_i, 0, -1)
        diag_all_plus[i] = torch.clamp(torch.diagonal(cov_i_plus), min=1e-3)

    # Compute the pseudo-inverse lifting operator and the corresponding matrices Cz and Az.
    M_pinv = torch.linalg.pinv(M)
    Cz = X @ M_pinv
    Az = Mplus @ M_pinv

    # Cost Term 1: 1-step Prediction Euclid + Trace
    NormNLPD = 0.0
    if lambda1 != 0.0:
        # --- after you have diag_all (p × NT), M (p × NT), X (n × NT), Cz (n×p), Az (p×p) ---
        num_steps = N - 2 - l
        device = X.device

        # 1) build indices for all (j,k) at once
        offsets = torch.arange(nT, device=device) * \
            N                          # (nT,)
        # (num_steps,)
        idx_base = torch.arange(num_steps, device=device)
        idx_M = (offsets[:, None] + (l+1) + idx_base[None, :]
                 ).reshape(-1)     # (nT*num_steps,)
        # shift by one for X
        idx_X = idx_M + 1

        # 2) gather into big “batch” of size B = nT*num_steps
        diag_batch = diag_all[:, idx_M]         # (p, B)
        M_batch = M[:, idx_M]           # (p, B)
        X_batch = X[:, idx_X]           # (n, B)

        # 3) compute A_z @ diag @ A_z^T in batch
        #    diag_batch.T has shape (B,p), so
        AzD = Az[None, :, :] * diag_batch.T[:,
                                            :, None]  # (B, p, p)  = Az @ diag
        A_batch = AzD @ Az.T                           # (B, p, p)

        # 4) push through Cz to get vx_batch = Cz * A_batch * Cz^T
        Cz_exp = Cz[None, :, :]                       # (1, n, p)
        B1_batch = Cz_exp @ A_batch                   # (B, n, p)
        vx_batch = (B1_batch @ Cz.T[None, :, :]).abs()  # (B, n, n)

        # 5) batched Cholesky & inverse
        # L_batch      = torch.linalg.cholesky(vx_batch)        # (B, n, n)
        # vx_inv_batch = torch.cholesky_inverse(L_batch)        # (B, n, n)

        # 6) batched error vectors
        #    first compute Cz @ Az @ M_batch  → shape (n, B)
        CtM = Cz @ (Az @ M_batch)                          # (n, B)
        err = (X_batch - CtM).T                            # (B, n)

        # 7) batched quadratic form + logdet
        #    a) quadratic form: eᵢᵀ V⁻¹ eᵢ for each i
        # qf     = torch.einsum('bi,bij,bj->b', err, vx_inv_batch, err)  # (B,)
        # (B,) | un-normalized
        qf = (err ** 2).sum(dim=1)

        #    b) log-det: use slogdet for stability
        # sign, logdet = torch.linalg.slogdet(vx_batch)        # both (B,)
        trace_batch = torch.diagonal(
            vx_batch, dim1=-2, dim2=-1).sum(dim=-1)  # (B,)
        #    (sign should be all +1 if SPD)

        NormNLPD = (qf + trace_batch).sum()

    # Cost Term 2: Lifting Accuracy (Bhattacharyya Distance)
    NormLift = 0.0
    if lambda2 != 0.0:
        # B = total (trajectory, step) pairs
        B = nT * N

        # Gather per-step variances
        # diag_all: (p, B), diag_all_plus: (p, B)
        d_k = diag_all.permute(1, 0)        # (B, p)
        d_kp = diag_all_plus.permute(1, 0)   # (B, p)

        # Dvkp_k = Az @ diag(d_k[b]) @ Az.T   (batched)
        Az_exp = Az.unsqueeze(0)                                   # (1, p, p)
        # (B, p, p) == Az @ diag(d_k)
        A_d = Az_exp * d_k.unsqueeze(1)
        Dvkp_k = A_d @ Az_exp.transpose(-1, -2)                    # (B, p, p)

        # Dvkp_kp = diag(d_kp[b]) (batched)
        Dvkp_kp = torch.diag_embed(d_kp)                            # (B, p, p)

        # σ = 0.5 * (Dvkp_k + Dvkp_kp)
        sigma = 0.5 * (Dvkp_k + Dvkp_kp)                          # (B, p, p)

        # Cholesky + inverse of σ
        # L         = torch.linalg.cholesky(sigma)                     # (B, p, p)
        # sigma_inv = torch.cholesky_inverse(L)                        # (B, p, p)

        # err = Mplus - Az @ M
        pred = Az @ M                                                # (p, B)
        err = (Mplus - pred).permute(1, 0)                          # (B, p)

        # Quadratic term: (errᵀ σ⁻¹ err) / 8  (matches your loop)
        # qf = torch.einsum('bi,bij,bj->b', err, sigma_inv, err) / 8.0 # (B,)
        qf = (err ** 2).sum(dim=1)

        # --- Diagonal-only logdet for Dvkp_k ---
        # diag(Dvkp_k)_i = sum_j Az[i,j]^2 * d_k[b,j]  ⇒ diag = d_k @ (Az⊙Az)^T
        eps = 1e-12
        Az_sq = Az.pow(2)                                     # (p, p)
        diag_prop = d_k @ Az_sq.T                                 # (B, p)
        logdet_Dvkp_k_diag = torch.log(
            diag_prop.clamp_min(eps)).sum(dim=1)  # (B,)

        # logdet(Dvkp_kp) for diagonal matrix = sum(log d_kp)
        logdet_Dvkp_kp = torch.log(d_kp.clamp_min(eps)).sum(dim=1)   # (B,)

        # logdet(σ) via slogdet (stable; same math as logdet for SPD)
        _, logdet_sigma = torch.linalg.slogdet(sigma)                 # (B,)

        # 0.5 * [ logdet(σ) - 0.5*( logdet_diag(Dvkp_k) + logdet(Dvkp_kp) ) ]
        log_term = 0.5 * (logdet_sigma - 0.5 *
                          (logdet_Dvkp_k_diag + logdet_Dvkp_kp))  # (B,)

        NormLift = (qf + log_term).sum()

    # Cost Term 3: Reconstruction Euclid + Trace
    NormRecon = 0.0
    if lambda3 != 0.0:
        # Total number of time-points across all trajectories
        B = nT * N          # B = number of (j,k) pairs

        # 1) Build batch of weighted C_z matrices: (B, n, p)
        #    Cz_exp: (1, n, p)  broadcast to (B, n, p)
        #    d_exp : (B, 1, p)  broadcast to (B, n, p)
        Cz_exp = Cz.unsqueeze(0)                          # (1, n, p)
        d_exp = diag_all.T.unsqueeze(1)                  # (B, 1, p)
        # (B, n, p) = C_z * diag(vz)
        CzD = Cz_exp * d_exp

        # 2) Form the full covariances: (B, n, n)
        vx_batch = CzD @ Cz.T                             # (B, n, n)

        # 3) Batched Cholesky + inverse
        # L_batch     = torch.linalg.cholesky(vx_batch)     # (B, n, n)
        # inv_batch   = torch.cholesky_inverse(L_batch)     # (B, n, n)

        # 4) Gather all errors in one go
        pred_batch = Cz @ M                              # (n,  B)
        err_batch = (X - pred_batch).T                   # (B,  n)

        # 5) Batched quadratic form eᵀ V⁻¹ e  →  (B,)
        # qf_batch    = torch.einsum('bi,bij,bj->b', err_batch, inv_batch, err_batch)
        # (B,) | un-normalized
        qf_batch = (err_batch ** 2).sum(dim=1)

        # 6) Batched log-determinant
        # _, logdet_batch = torch.linalg.slogdet(vx_batch)
        trace_batch = torch.diagonal(
            vx_batch, dim1=-2, dim2=-1).sum(dim=-1)  # (B,)
        # (we expect sign all +1 if SPD)

        # 7) Sum everything
        NormRecon = (qf_batch + trace_batch).sum()

    cost = (lambda1 * NormNLPD / ((N - l) * nT)) + (lambda2 *
                                                    NormLift / (N * nT)) + (lambda3 * NormRecon / (N * nT))
    return cost


def cost_prop_theta_fast(X, Xplus, manager, nz=1, nT=1):
    """
    Fully vectorized across trajectories (no Python loops over j),
    returns sqrt(nz) * mean_j [ ||I - P_j|| * ||Gplus_j|| ].
    """
    nT, nz = int(nT), int(nz)
    N = X.shape[1] // nT
    device = manager.observables[0].device

    # 1) Precompute G_full and Gplus_full for each observable on the *entire* data
    G_full_blocks = []
    Gp_full_blocks = []
    for obs in manager.observables.values():
        # obs.forward_G returns shape (nT*N, n_i)
        G_full = obs.forward_G(X)   # (nT*N, n_i)
        Gp_full = obs.forward_G(Xplus)   # (nT*N, n_i)

        # reshape into a batch of per-trajectory feature maps (nT, N, n_i)
        G_full_blocks .append(G_full .view(nT, N, -1))
        Gp_full_blocks.append(Gp_full.view(nT, N, -1))

    # 2) Concatenate features from *all* observables into (nT, N, F)
    G_batched = torch.cat(G_full_blocks,  dim=2)  # (nT, N, F)
    Gp_batched = torch.cat(Gp_full_blocks, dim=2)  # (nT, N, F)

    # 3) Build the batch of Gram matrices: (nT, F, F)
    #    and invert each one via Cholesky (or fallback to full inverse).
    Gram = G_batched.transpose(1, 2) @ G_batched    # (nT, F, F)
    try:
        # batched Cholesky & inverse
        L = torch.linalg.cholesky(Gram)               # (nT, F, F)
        invGram = torch.cholesky_inverse(L)                 # (nT, F, F)
    except RuntimeError:
        invGram = torch.linalg.inv(Gram)                    # (nT, F, F)

    # 4) Compute the batch of projection matrices P_j = G_j invGram_j G_j^T → (nT, N, N)
    P = G_batched @ invGram @ G_batched.transpose(1, 2)    # (nT, N, N)

    # 5) Compute per-trajectory norms:
    I_N = torch.eye(N, device=device).unsqueeze(0)         # (1, N, N)
    # Frobenius norm over last two dims for each batch element
    cost1 = torch.linalg.vector_norm(I_N - P,    dim=(1, 2))  # (nT,)
    cost2 = torch.linalg.vector_norm(Gp_batched, dim=(1, 2))  # (nT,)

    # 6) average over trajectories
    return math.sqrt(nz) * (cost1 * cost2).mean()


def cost_recon_Z(Zcal, manager, X, nT=1, Ngp=None):
    """
    Calculate the reconstruction based cost for virtual target selection
    """
    nz = Zcal.shape[0]  # Zcal = \mathcal{Z}, constructed as block-diagonal of different Z.T
    N = X.shape[1] / nT
    if Ngp is None:
        Ngp = int(0.1 * N)
    nT, N, Ngp, nz = int(nT), int(N), int(Ngp), int(nz)
    G = torch.empty((nT * N, nT * Ngp * nz),
                    device=manager.observables[0].device)
    for i in range(nz):
        G[:, i*Ngp*nT:(i+1)*nT*Ngp] = manager.observables[i].forward_G(X)
    Phi = Zcal @ torch.transpose(G, 0, 1)
    PhiT = torch.transpose(Phi, 0, 1)
    try:
        L = torch.linalg.cholesky(Phi @ PhiT)
        invPPt = torch.cholesky_inverse(L)
    except RuntimeError:
        invPPt = torch.linalg.inv(Phi @ PhiT)

    eye_N = torch.eye(N * nT, device=manager.observables[0].device)
    Zerr = X @ (eye_N - PhiT @ invPPt @ Phi)
    Zcost = torch.linalg.matrix_norm(Zerr)
    return Zcost


def get_iGPK(
    # normalized (and optionally noised) data, (num_traj, n, N+1)
    SimData,
    nTrain, nTest,
    lifting_order,
    # [max_iter, (optional phase_len_hp), (optional phase_len_z), (optional reserve_final_z)]
    iters_list,
    learn_rate,
    opt_weights,                 # [lambda1, lambda2, lambda3]
    routine="Z_only",            # "Z_only" or "SpacedOpt"
    train_method="Horizon",      # "Horizon" or "K-Means"
    device="cuda:0",
    seed=1234
):
    """
    Train iGPK, build Koopman (A, C), simulate train/test, and return predictions, covariances, NRMSE.

    NOTE: Data loading & noise addition remain outside. Pass prepped SimData in.
    """
    torch.manual_seed(seed)
    SimData = SimData.float().to(device)

    # Shapes & basic splits
    n = SimData.shape[1]
    N = SimData.shape[2] - 1
    p = lifting_order
    l = 1  # lifting horizon fixed to 1 (kept from your script)

    # Build concatenated matrices and ICs from SimData
    Xall = torch.cat([SimData[j, :, :]
                     for j in range(nTrain)], dim=1)     # n x (nTrain*(N+1))
    X = torch.cat([SimData[j, :, 0:N]
                  for j in range(nTrain)], dim=1)     # n x (nTrain*N)
    Xplus = torch.cat([SimData[j, :, 1:]
                      for j in range(nTrain)], dim=1)     # n x (nTrain*N)

    ICsetTrain = torch.cat([SimData[j, :, 0].view(n, 1)
                           for j in range(nTrain)], dim=1)
    ICsetTest = torch.cat([SimData[j, :, 0].view(n, 1)
                          for j in range(nTrain, nTrain + nTest)], dim=1)

    # Initialize manager & decision variable Z (training grid)
    if train_method == "Horizon":
        Xtrain = torch.cat([X[:, j*N: j*N + l]
                           for j in range(nTrain)], dim=1)  # n x (nTrain*l)
        Z = torch.nn.Parameter(torch.rand(Xtrain.shape[1], p, device=device))
        ObsManager = gpk.GPObservablesManager()
        for i in range(p):
            ObsManager.add_observable(
                index=i, d=n, ns=l*nTrain, kernel_types=['Gaussian'],
                combination='sum', noise=1e-4, m=500, device=device
            )
        for i in range(p):
            ObsManager.train_observable(i, Xtrain, Z[:, i])
        ObsManager.set_random_hyperparameters(scale=[1.0, 1.0, None])

    elif train_method == "K-Means":
        Xtrain = torch.cat([X[:, j*N: j*N + l] for j in range(nTrain)], dim=1)
        Z = torch.nn.Parameter(torch.rand(Xtrain.shape[1], p, device=device))
        ObsManager = gpk.GPObservablesManager()
        centroids = gpk.get_kmeans(X, num_centers=p)
        for i in range(p):
            ObsManager.add_observable(
                index=i, d=n, ns=l*nTrain,
                kernel_types=['ExplicitAttractor', 'Gaussian'],
                combination='sum', noise=1e-4, m=500, device=device
            )
        for i in range(p):
            ObsManager.train_observable(i, Xtrain, Z[:, i])
        ObsManager.set_random_hyperparameters(scale=[1.0, 2.0, None])
        mu_centroids = [centroids[:, i:i+1] for i in range(centroids.shape[1])]
        mu_centroids.extend(mu_centroids)
        ObsManager.set_parameters(mu_list=mu_centroids)
    else:
        raise ValueError(f"Unrecognized train_method: {train_method}")

    # === Optimization ===
    max_iter = iters_list[3]
    lam1, lam2, lam3 = opt_weights
    iter = 0
    cost_history = []

    if routine == "Z_only":
        optimizer = torch.optim.SGD(
            [Z], lr=learn_rate, momentum=0.75, nesterov=True)
        while iter < max_iter:
            optimizer.zero_grad()
            cost = get_cost_simple(Z, X, Xplus, ObsManager,
                                   nT=nTrain, lambda1=lam1, lambda2=lam2, lambda3=lam3)
            cost.backward()
            optimizer.step()
            cost_history.append(cost.item())
            iter += 1
    else:
        raise ValueError(f"Invalid routine: {routine}")

    # Plot Cost History
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('log(Cost)', color=color)
    ax1.plot(torch.log10(torch.abs(torch.tensor(cost_history))), color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Cost', color=color)
    ax2.plot(cost_history, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()

    # === Retrain GPs at optimal Z & (optionally) optimize hp ===
    optimal_Z = Z.detach()
    for i in range(p):
        ObsManager.train_observable(i, Xtrain, optimal_Z[:, i])

    ObsManager.optimize_hyperparameters(
        opt_mu=False, opt_sigma=True, max_iter=250, lr=0.01)

    # === Koopman A, C ===
    ObsList = [i for i in range(p)]
    A, C = gpk.getKoopman(ObsManager, ObsList, Xall, nTrain, stateAug=False)

    # === Simulate & evaluate ===
    #   Train split (offset 0), Test split (offset nTrain)
    XhatTrain, XcvTrain, TrainNRMSE = gpk.sim_and_eval(
        ObsManager, A, C, ICsetTrain, SimData, traj_offset=0)
    XhatTest,  XcvTest,  TestNRMSE = gpk.sim_and_eval(
        ObsManager, A, C, ICsetTest,  SimData, traj_offset=nTrain)

    # === Package results ===
    return {
        "ObsManager": ObsManager,
        "A": A, "C": C,
        "ICsetTrain": ICsetTrain.detach().cpu(),
        "ICsetTest":  ICsetTest.detach().cpu(),
        "Train": {
            "Xhat": XhatTrain,           # (nTrain, n, N)
            "Xcv":  XcvTrain,            # (nTrain, n, n, N)
            "NRMSE": TrainNRMSE          # (nTrain, n)
        },
        "Test": {
            "Xhat": XhatTest,            # (nTest, n, N)
            "Xcv":  XcvTest,             # (nTest, n, n, N)
            "NRMSE": TestNRMSE           # (nTest, n)
        },
        "history": {
            "cost": torch.tensor(cost_history).cpu()
        }
    }


if __name__ == "__main__":
    system_name = 'Cart_data'
    train_frac, test_frac = 0.4, 0.2
    clip = None
    lifted_order = 20
    noise_type = 'gaussian'
    # unused, samples, iterations, inner iterations
    iters_list = [0, 8, 20, 100]
    routine = "BO_hp_and_GD_Z"  # BO_Z | BO_ZnHP | BO_hp_and_GD_Z
    # 1) Load + normalize
    SimData_raw, ts, num_traj, N, nTrain, nTest = gpk.load_SimData(
        system_name, train_frac, test_frac, clip=clip)
    SimData_clean, mu_vec, std_vec = gpk.normalize_data(
        SimData_raw, nTrain, N)

    # 2) Noise
    SimData = gpk.add_noise(SimData_clean, noise_type=noise_type,
                            intensity=0., seed=1234)

    print(f'==== Starting iGPK Model Identification ====')
    t0 = time.perf_counter()
    results = get_iGPK(SimData, nTrain, nTest, lifted_order,
                       iters_list, learn_rate=0.04,
                       opt_weights=[1.0, 1.0, 1.0], routine=routine,
                       train_method="Horizon")
    t_BO = time.perf_counter() - t0
    print(
        f'Bayesian Optimization with {iters_list[1]}-iterations, {iters_list[2]}-samples, finished in {t_BO:.2f} seconds.')

    # unpack iGPK
    A_igpk, C_igpk = results["A"], results["C"]
    # ICsetTrain, ICsetTest = results["ICsetTrain"], results["ICsetTest"]
    XhatTrain, XcvhatTrain, TrainNRMSE = results["Train"][
        "Xhat"], results["Train"]["Xcv"], results["Train"]["NRMSE"]
    XhatTest,  XcvhatTest,  TestNRMSE = results["Test"][
        "Xhat"],  results["Test"]["Xcv"],  results["Test"]["NRMSE"]

    gpk.plot_eigen(A_igpk)

    gpk.plot_NRMSE_metrics([TrainNRMSE*100], [TestNRMSE*100], ['iGPK-BO'])

    # 6) indices + timebase
    idx_trainMIN = torch.argmin(TrainNRMSE.mean(dim=1))
    idx_testMIN = torch.argmin(TestNRMSE.mean(dim=1))
    idx_testMAX = torch.argmax(TestNRMSE.mean(dim=1))
    # same shape you used (see your callsite) :contentReference[oaicite:11]{index=11}
    time_arr = torch.arange(0., ts * (SimData.shape[2] - 1), ts)

    # 7) pack models for overlay plot
    models = [
        {"name": "iGPK-BO", "train": {"Xhat": XhatTrain},
            "test": {"Xhat": XhatTest}}
    ]

    # a) 3 trajectory overlays
    for (which, idx, split, sim_offset, suffix) in [
        ("best-train", idx_trainMIN, "train", 0,         "Best_Train"),
        ("best-test",  idx_testMIN,  "test",  nTrain,    "Best_Test"),
        ("worst-test", idx_testMAX,  "test",  nTrain,    "Worst_Test"),
    ]:
        gpk.compare_model_predictions(
            time=time_arr, models=models, SimData=SimData, idx=idx, N=(
                SimData.shape[2]-1),
            system_name=system_name, title_suffix=suffix, split=split, sim_offset=sim_offset,
            compare_to="SimData_clean", SimData_clean=SimData_clean, sigma=1.0
        )

    plt.show()
