## --- IMPORTS --- ###
import GPKoopman as gpk
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
import time

## --- COST FUNCTION --- ##


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
    n = X.shape[0]          # State dimension

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

## --- HELPER FUNCTIONS --- ##
# --- Noise Injection Utility ---


def add_noise(SimData_norm, noise_type="gaussian", intensity=0.05, seed=None):
    """
    Add noise to normalized simulation data.

    Args:
        SimData_norm (torch.Tensor): Normalized trajectories,
            shape (num_traj, state_dim, num_steps).
        noise_type (str): 'gaussian' or 'uniform'.
        intensity (float): Noise strength (default 0.05 = 5% of 1 std).
                          Since data is normalized, this is relative to std=1.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        torch.Tensor: Noisy version of SimData_norm.
    """
    if seed is not None:
        torch.manual_seed(seed)

    if noise_type == "gaussian":
        noise = torch.randn_like(SimData_norm) * intensity
    elif noise_type == "uniform":
        # Uniform noise in [-intensity, intensity]
        noise = (torch.rand_like(SimData_norm) * 2 - 1) * intensity
    elif noise_type == "linear_gaussian":
        # Gaussian noise with intensity varying linearly with state value
        var_intensity = intensity * SimData_norm
        noise = torch.randn_like(SimData_norm) * var_intensity
    elif noise_type == "quadratic_gaussian":
        # Gaussian noise with intensity varying as square of state value
        var_intensity = intensity * (SimData_norm ** 2)
        noise = torch.randn_like(SimData_norm) * var_intensity
    elif noise_type == "linear_gaussian":
        # Uniform noise with linearly varying intensity
        var_intensity = intensity * SimData_norm
        noise = (torch.rand_like(SimData_norm) * 2 - 1) * var_intensity
    else:
        raise ValueError(
            f"Unsupported noise_type {noise_type}. Choose 'gaussian' or 'uniform'.")

    return SimData_norm + noise


def load_SimData(system_name, trainFrac, testFrac, clip=None):
    data = torch.load(f"Data/DataAuto_{system_name}.pt", weights_only=True)
    # Shape: (num_trajectories, state_dim, num_steps)
    SimData = data["trajectories"]
    ts = data["sample_time"]
    num_trajectories, N = data["num_trajectories"], data["num_steps"]

    nTrain, nTest = math.floor(
        num_trajectories * trainFrac), math.floor(num_trajectories * testFrac)
    if clip is not None:
        SimData = SimData[:, :, :clip+1]
        N = clip

    return SimData, ts, num_trajectories, N, nTrain, nTest


def normalize_data(SimData_raw, nTrain, N):
    # Compute normalization stats over training split only
    # SimData shape: (num_traj, state_dim, num_steps)
    mu_vec = SimData_raw[:nTrain, :, :N].mean(
        dim=(0, 2))                                # (n,)
    std_vec = SimData_raw[:nTrain, :, :N].std(
        dim=(0, 2), unbiased=False).clamp_min(1e-8)  # (n,)

    # Apply normalization to ALL trajectories (train+test); keep everything on CPU for now
    SimData = (SimData_raw - mu_vec.view(1, -1, 1)) / std_vec.view(1, -1, 1)
    return SimData, mu_vec, std_vec


def set_requires(params, flag: bool):
    for p in params:
        if p is not None:
            p.requires_grad_(flag)


def get_iGPK(lifting_order, lifting_horizon, N, nTrain, nTest, SimData, iters_list, learn_rate, opt_weights):
    SimData = SimData.float().to(device='cuda:0')
    # mu_vec_cpu, std_vec_cpu = mu_vec.clone().cpu(), std_vec.clone().cpu()

    ## OPTIMIZATION AND MODEL EXTRACTION --- ##
    # Original State Dim | Lifted State Dim | Learning Horizon | Resolution
    n, p, l = SimData.shape[1], lifting_order, lifting_horizon
    stop = None
    torch.manual_seed(1234)

    if stop is None:    # Setup Optimization Variables
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

    # Options: 'Horizon' | 'K-Means'
        trainMethod = 'Horizon'

    # Initialize GP training-grid and decision variables
        if trainMethod == 'Horizon':
            Xtrain = torch.cat([X[:, j*N:j*N+l] for j in range(nTrain)], dim=1)
        # Z = torch.rand(Xtrain.shape[1], p, requires_grad=True)
            Z = torch.nn.Parameter(torch.rand(
                Xtrain.shape[1], p, device='cuda:0'))
            ObsManager = gpk.GPObservablesManager()
            for i in range(p):
                ObsManager.add_observable(
                    index=i, d=n, ns=l*nTrain, kernel_types=['Gaussian'], combination='sum', noise=1e-4, m=500)
            for i in range(p):
                ObsManager.train_observable(i, Xtrain, Z[:, i])
            ObsManager.set_random_hyperparameters(scale=[1., 1.0, None])
            print('Observable Hyperparameters have been randomized:')
            ObsManager.print_parameters()

        elif trainMethod == 'K-Means':
            Xtrain = torch.cat([X[:, j*N:j*N+l] for j in range(nTrain)], dim=1)
            Z = torch.nn.Parameter(torch.rand(
                Xtrain.shape[1], p, device='cuda:0'))
            ObsManager = gpk.GPObservablesManager()
            centroids = gpk.get_kmeans(X, num_centers=p)
            for i in range(p):
                ObsManager.add_observable(index=i, d=n, ns=l*nTrain, kernel_types=[
                    'ExplicitAttractor', 'Gaussian'], combination='sum', noise=1e-4, m=500)

            for i in range(p):
                ObsManager.train_observable(i, Xtrain, Z[:, i])
            ObsManager.set_random_hyperparameters(scale=[1., 5.0, None])
        # hp1_val, hp2_val = 1.0, 0.5
        # hp1_val = [torch.tensor([hp1_val], device='cuda:0') for _ in range(2*p)]
        # hp2_val = [torch.tensor([hp2_val], device='cuda:0') for _ in range(p)]
        # hp2_val.extend([torch.tensor([0.001], device='cuda:0') for _ in range(p)])
        # ObsManager.set_parameters(hp1_list=hp1_val, hp2_list=hp2_val)
            mu_centroids = [centroids[:, i:i+1]
                            for i in range(centroids.shape[1])]
            mu_centroids.extend(mu_centroids)
            ObsManager.set_parameters(mu_list=mu_centroids)
            print('Observable Hyperparameters have been randomized:')
            ObsManager.print_parameters()

            for i in range(p):
                plt.plot(centroids[0, i].cpu(),
                         centroids[1, i].cpu(), marker='o')
            plt.grid()
            plt.title(
                'Centroids of K-Means Clusters'), plt.xlabel('X1'), plt.ylabel('X2')

        else:
            raise ValueError(f'Unrecognized GP Training method {trainMethod}')

    # Optimization Parameters: Maximum Iterations | Learning Rate
    max_iter = iters_list[0]
    lambda1, lambda2, lambda3 = opt_weights[0], opt_weights[1], opt_weights[2]
    cost_history, iter = [], 0
    # Optimization Options: "Spaced HPOpt" | "Opt all" | "Z only"
    routine_name = "SpacedOpt"

    if routine_name == 'SpacedOpt':  # Main Optimization Loop
        print('Starting Iteration Loop!')
    # ---- configuration ----
        phase_len_hp = iters_list[1]    # steps of HP optimization per phase
        phase_len_z = iters_list[2]     # steps of Z optimization per phase
        reserve_final_z = iters_list[3]  # last iterations dedicated to Z-only

    # ---- collect hyperparameters from the manager ----
        hp_params = []
        for obs in ObsManager.observables.values():
            hp_params += list(getattr(obs, "hp1_list", []))
            hp_params += list(getattr(obs, "hp2_list", []))
        # noise = getattr(obs, "noise", None)
        # if isinstance(noise, torch.nn.Parameter):
        #     hp_params.append(noise)

    # ---- two optimizers ----
    # a bit smaller for HPs is often more stable
        lr_Z, lr_HP = learn_rate, learn_rate * 0.5
        optZ = torch.optim.Adam([Z],        lr=lr_Z)
        optHP = torch.optim.Adam(hp_params,  lr=lr_HP) if hp_params else None
        iter, cost_history = 0, []

    # We will alternate HP -> Z until 'switch_point', then Z-only for the last 200 iters.
        switch_point = max(0, max_iter - reserve_final_z)

    # ---------- Alternating phases (HP first) ----------
        while iter < switch_point:
            # ---- Phase: HP ----
            if optHP and phase_len_hp > 0:
                set_requires([Z], False)
                set_requires(hp_params, True)
                steps = min(phase_len_hp, switch_point - iter)
                for _ in range(steps):
                    optHP.zero_grad(set_to_none=True)
                    cost = get_cost_ACnew2(
                        Z, X, Xplus, ObsManager,
                        nT=nTrain, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3
                    )
                    cost.backward()
                # optional: torch.nn.utils.clip_grad_norm_(hp_params, max_norm=1.0)
                    optHP.step()
                    cost_history.append(cost.item())
                    iter += 1
                    print(
                        f"[{iter}/{max_iter}] Phase=HP | Cost={cost.item():.6e}")
                    if iter >= switch_point:
                        break

            if iter >= switch_point:
                break

        # ---- Phase: Z ----
            if phase_len_z > 0:
                set_requires([Z], True)
                set_requires(hp_params, False)
                steps = min(phase_len_z, switch_point - iter)
                for _ in range(steps):
                    optZ.zero_grad(set_to_none=True)
                    cost = get_cost_ACnew2(
                        Z, X, Xplus, ObsManager,
                        nT=nTrain, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3
                    )
                    cost.backward()
                # optional: torch.nn.utils.clip_grad_norm_([Z], max_norm=5.0)
                    optZ.step()
                    cost_history.append(cost.item())
                    iter += 1
                    print(f"[{iter}/{max_iter}] Phase=Z | Cost={cost.item():.6e}")
                    if iter >= switch_point:
                        break

    # ---------- Final Z-only phase (last 200 iters) ----------
        set_requires([Z], True)
        set_requires(hp_params, False)
        while iter < max_iter:
            optZ.zero_grad(set_to_none=True)
            cost = get_cost_ACnew2(
                Z, X, Xplus, ObsManager,
                nT=nTrain, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3
            )
            cost.backward()
        # optional: torch.nn.utils.clip_grad_norm_([Z], max_norm=5.0)
            optZ.step()
            cost_history.append(cost.item())
            iter += 1
            print(f"[{iter}/{max_iter}] Phase=Z(final) | Cost={cost.item():.6e}")

    if iter == max_iter:    # Print Stopping Criterion AND Plots
        print(f'Stopping: Reached maximum number of iterations = {iter}.')
        optimal_Z = Z.detach()
        print('Optimization Complete.')
        print("Final Cost:", cost.item())

    # Plot cost history
        plt.figure(figsize=(6, 4))
        plt.plot(cost_history, label="Cost")
        plt.title("Cost History")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.legend(), plt.grid()
        # plt.show()

    # Logarithmic cost history
        plt.figure(figsize=(6, 4))
        plt.plot(torch.log10(torch.abs(torch.tensor(cost_history))),
                 label="log(Cost)")
        plt.title("Logarithmic Cost History")
        plt.xlabel("Iteration")
        plt.ylabel("log10(Cost)")
        plt.legend(), plt.grid()
        # plt.show()

    for i in range(p):  # Re-Train on Optimal Z
        # train GP Observables with Optimal Z outputs
        ObsManager.train_observable(i, Xtrain, optimal_Z[:, i])

    # Optimize Kernel hyperparameters for Optimal training data
    ObsManager.optimize_hyperparameters(
        opt_mu=False, opt_sigma=True, max_iter=25)
    print(f'GPO Hyperparameters have been optimized.')
    ObsManager.print_parameters()

    ObsList = [i for i in range(p)]
    A, C = gpk.getKoopman(ObsManager, ObsList, Xall, nTrain, stateAug=False)
    return ICsetTrain, ICsetTest, ObsManager, A, C


def plot_eigen(A):  # Eigen value plot of Koopman Matrices
    A = A.detach().cpu()
    eigval = torch.linalg.eigvals(A)

    eigreal, eigimag = eigval.real, eigval.imag
    eigreal, eigimag = eigreal.detach().numpy(), eigimag.detach().numpy()
    eig_mag = np.sqrt(eigreal**2 + eigimag**2)

    theta = np.linspace(0, 2*np.pi, 500)
    unitCirclex, unitCircley = np.cos(theta), np.sin(theta)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    # First subplot: Eigenvalues plot
    axes[0].plot(unitCirclex, unitCircley, color='orange', label='Unit Circle')
    for i in range(np.size(eig_mag)):
        if eig_mag[i] <= 1:
            axes[0].scatter(eigreal, eigimag, color='green',
                            label='Eigenvalues')
        else:
            axes[0].scatter(eigreal, eigimag, color='red', label='Eigenvalues')

    axes[0].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes[0].axvline(0, color='black', linewidth=0.5, linestyle='--')
    axes[0].set_title(f"Eigenvalues of A Matrix with {A.shape[0]} Observables")
    axes[0].set_xlabel("Real Part")
    axes[0].set_ylabel("Imaginary Part")
    axes[0].grid(True)
    axes[0].legend(labels=['Unit Circle', 'Eignevalues'], loc='upper right')

    # Second subplot: Heatmap of matrix A
    im = axes[1].imshow(A.detach().numpy(), cmap='viridis', aspect='auto')
    fig.colorbar(im, ax=axes[1], label="Value")
    axes[1].set_title(f'{A.shape[0]}-D Koopman Matrix')
    axes[1].set_xlabel("Columns")
    axes[1].set_ylabel("Rows")
    plt.tight_layout()
    # plt.show()


def sim_and_eval(ObsManager, A, C, ICsetTrain, SimData):
    A, C = A.to(device='cpu'), C.to(device='cpu')
    ICsetTrain, SimData = ICsetTrain.to(device='cpu'), SimData.to(device='cpu')
    nTrain, p, n, N = ICsetTrain.shape[1], A.shape[0], C.shape[0], SimData.shape[2]-1
    # Evaluation on training set
    ZmeanTrain, ZcvTrain = torch.empty(
        (nTrain, p, N)), torch.empty((nTrain, p, p, N))
    XhatTrain, XcvhatTrain = torch.empty(
        (nTrain, n, N)), torch.empty((nTrain, n, n, N))

    # -- new: containers for normalized RMSE
    TrainNRMSE = torch.empty((nTrain, n))

    for j in range(nTrain):
        # predict initial lifted states
        for i in range(p):
            ZmeanTrain[j, i, 0] = ObsManager.predict_mean(
                i, ICsetTrain[:, j].view(n, 1))
            ZcvTrain[j, i, i, 0] = ObsManager.predict_covariance(
                i, ICsetTrain[:, j].view(n, 1))

        # simulate
        ZmeanTrain[j], ZcvTrain[j], XhatTrain[j], XcvhatTrain[j] = gpk.sim_LTI(
            ZmeanTrain[j, :, 0], A, C, num_steps=N, ts=None, x0cv=ZcvTrain[j, :, :, 0]
        )

        # absolute RMSE per state
        errors = XhatTrain[j] - SimData_clean[j, :, :N]               # (n, N)
        rmse = torch.sqrt(torch.mean(errors**2, dim=1))       # (n,)

        # true‐value range per state
        true_vals = SimData[j, :, :N]                          # (n, N)
        max_vals = true_vals.max(dim=1).values                # (n,)
        min_vals = true_vals.min(dim=1).values                # (n,)
        range_vals = max_vals - min_vals                        # (n,)

        # avoid division by zero
        range_vals = torch.where(range_vals == 0,
                                 torch.ones_like(range_vals),
                                 range_vals)

        # normalized RMSE
        TrainNRMSE[j] = rmse / range_vals

    XhatTrain, XcvhatTrain = XhatTrain.detach(), XcvhatTrain.detach()
    TrainNRMSE = TrainNRMSE.detach()
    return XhatTrain, XcvhatTrain, TrainNRMSE


def plot_NRMSE_metrics(TrainNRMSE, TestNRMSE):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Training set plot
    axes[0].plot(range(TrainNRMSE.shape[0]), TrainNRMSE[:, 0].numpy(),
                 marker='o', linestyle='-', label='NRMSE X1')
    axes[0].plot(range(TrainNRMSE.shape[0]), TrainNRMSE[:, 1].numpy(),
                 marker='o', linestyle='-', label='NRMSE X2')
    axes[0].set_title('Training Metrics')
    axes[0].set_xlabel("Trajectory Index")
    axes[0].set_ylabel("Metric Value")
    axes[0].legend()
    axes[0].grid()

    # Test set plot
    axes[1].plot(range(TestNRMSE.shape[0]), TestNRMSE[:, 0].numpy(),
                 marker='o', linestyle='-', label='NRMSE X1')
    axes[1].plot(range(TestNRMSE.shape[0]), TestNRMSE[:, 1].numpy(),
                 marker='o', linestyle='-', label='NRMSE X2')
    axes[1].set_title('Test Metrics')
    axes[1].set_xlabel("Trajectory Index")
    axes[1].set_ylabel("Metric Value")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    # plt.show()


## --- DATA LOADING --- ##
# Allowed system names - "Unforced Duffing_right" | "van der Pol" | "Simple Pendulum" | "Lorenz" | "Lotka Volterra"
# Load Data
system_name = 'Simple Pendulum'
SimData_raw, ts, num_trajectories, N, nTrain, nTest = load_SimData(
    system_name, 0.4, 0.2, clip=100)
# Normalization
SimData_clean, mu_vec, std_vec = normalize_data(SimData_raw, nTrain, N)
# Add Noise
SimData = add_noise(SimData_clean, noise_type="gaussian",
                    intensity=0.02, seed=100)

iters_list = [500, 50, 50, 100]
opt_weights = [10., 1., 10.]

ICsetTrain, ICsetTest, ObsManager, A, C = get_iGPK(
    10, 1, N, nTrain, nTest, SimData, iters_list=iters_list, learn_rate=0.0025, opt_weights=opt_weights)

## --- PLOTTING --- ##
plot_eigen(A)

# SIMULATE
XhatTrain, XcvhatTrain, TrainNRMSE = sim_and_eval(
    ObsManager, A, C, ICsetTrain, SimData_clean)
XhatTest, XcvhatTest, TestNRMSE = sim_and_eval(
    ObsManager, A, C, ICsetTest, SimData_clean)

idx_trainMIN = torch.argmin(TrainNRMSE.mean(dim=1))
idx_testMIN = torch.argmin(TestNRMSE.mean(dim=1))
idx3_testMAX = torch.argmax(TestNRMSE.mean(dim=1))
time = torch.arange(0., ts * N, ts)

# PLOT TRAJECTORIES
# gpk.plot_phase(XhatTrain, SimData, ICsetTrain, idx_trainMIN, N, system_name, 'Training Trajectory')
gpk.plot_time_series_with_bounds(time, XhatTrain, XcvhatTrain, SimData,
                                 idx_trainMIN, N, system_name, title_suffix='Best Train Trajectory')

# gpk.plot_phase(XhatTest, SimData, ICsetTest, idx_testMIN, N, system_name, 'Best Test Trajectory', sim_offset=nTrain)
gpk.plot_time_series_with_bounds(time, XhatTest, XcvhatTest, SimData, idx_testMIN,
                                 N, system_name, title_suffix='Best Test Trajectory', sim_offset=nTrain)

# gpk.plot_phase(XhatTest, SimData, ICsetTest, idx3_testMAX, N, system_name, 'Worst Test Trajectory', sim_offset=nTrain)
gpk.plot_time_series_with_bounds(time, XhatTest, XcvhatTest, SimData, idx3_testMAX,
                                 N, system_name, title_suffix='Worst Test Trajectory', sim_offset=nTrain)


plot_NRMSE_metrics(TrainNRMSE, TestNRMSE)

plt.show()
### --- MODEL SAVING --- ###


# ===== END ===== #
