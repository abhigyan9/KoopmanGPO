## --- IMPORTS --- ###
import GPKoopman as gpk
import torch
import matplotlib.pyplot as plt
import time
import numpy as np
import math

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
    M = torch.zeros((p, N * nT), device=X.device)
    Mplus = torch.zeros((p, N * nT), device=X.device)
    # diag_all = torch.zeros((p, N * nT), device=X.device)
    # diag_all_plus = torch.zeros((p, N * nT), device=X.device)
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
        L = torch.linalg.cholesky(
            M @ M.mT + (1e-10 * torch.eye(p, device=X.device)))
        M_pinv = torch.cholesky_solve(M.mT, L)
    except RuntimeError:
        M_pinv = torch.linalg.pinv(M)

    M_pinvM = M_pinv @ M

    cost1 = torch.linalg.matrix_norm(Mplus - (Mplus @ M_pinvM))
    cost2 = torch.linalg.matrix_norm(X - (X @ M_pinvM))
    # cost3 = torch.linalg.matrix_norm(Z)

    return (lambda1 * cost1) + (lambda2 * cost2)

import torch


def _standardize_G_shape(G, S, r):
    """
    Ensure G has shape (S, r), where:
        S = number of query snapshots
        r = number of virtual targets = Z.shape[0]
    """
    if G.shape == (S, r):
        return G
    elif G.shape == (r, S):
        return G.mT
    else:
        raise ValueError(
            f"Unexpected G shape {tuple(G.shape)}. Expected {(S, r)} or {(r, S)}."
        )


@torch.no_grad()
def build_G_cache(manager, X, Xplus):
    """
    Build detached G caches for Z-only optimization.

    Returns
    -------
    G_X      : (p, S, r)
    G_Xplus  : (p, S, r)

    where:
        p = number of observables
        S = N*nT
        r = number of virtual targets
    """
    p = len(manager.observables)
    r = manager.observables[0].y.shape[0] # Z.shape[0]
    S = X.shape[1]

    G_X_list = []
    G_Xplus_list = []

    for i in range(p):
        obs = manager.observables[i]

        Gi = obs.forward_G(X)
        Gpi = obs.forward_G(Xplus)

        Gi = _standardize_G_shape(Gi, S, r)
        Gpi = _standardize_G_shape(Gpi, S, r)

        G_X_list.append(Gi)
        G_Xplus_list.append(Gpi)

    G_X = torch.stack(G_X_list, dim=0).contiguous()
    G_Xplus = torch.stack(G_Xplus_list, dim=0).contiguous()

    return G_X, G_Xplus


def get_cost_simple_fast(
    Z, X, Xplus, manager,
    G_X, G_Xplus, nT=1,
    lambda1=1.0, lambda2=1.0, lambda3=1.0,
    jitter=1e-6,
    squared=True,
):
    """
    Fast version of get_cost_simple.

    Avoids explicitly forming:
        M_pinvM = M_pinv @ M

    This prevents construction of an (N*nT) x (N*nT) matrix.
    """

    p = Z.shape[1]
    dtype = X.dtype
    device = X.device

    # ------------------------------------------------------------
    # 1. Build lifted mean matrices M and Mplus
    # ------------------------------------------------------------
    M = torch.einsum("isr,ri->is", G_X, Z)
    Mplus = torch.einsum("isr,ri->is", G_Xplus, Z)

    # ------------------------------------------------------------
    # 2. Compute Gram matrix in lifted space
    # ------------------------------------------------------------
    eye_p = torch.eye(p, dtype=dtype, device=device)
    # Gram = M @ M.mT + jitter * eye_p        # (p, p)

    # Use Cholesky solve, not pinv.
    # If this fails often, increase jitter rather than falling back to pinv.
    try:
        L = torch.linalg.cholesky(M @ M.mT + jitter * eye_p)
    except RuntimeError:
        try:
            L = torch.linalg.cholesky(M @ M.mT + (10*jitter) * eye_p)
        except RuntimeError:
            L = torch.linalg.cholesky(M @ M.mT + (100*jitter) * eye_p)

    # ------------------------------------------------------------
    # 3. Compute B P_M without forming P_M
    # ------------------------------------------------------------
    # B contains both terms whose projection residual we need:
    #   Mplus P_M and X P_M
    B = torch.cat([Mplus, X], dim=0)         # (p+n, S)

    # coeff = (B M.T) (M M.T + eps I)^(-1)
    # Use cholesky_solve for stability:
    BMt = B @ M.mT                          # (p+n, p)
    coeff = torch.cholesky_solve(BMt.mT, L).mT  # (p+n, p)

    residual = B - coeff @ M                # (p+n, S)

    R1 = residual[:p, :]                    # Mplus projection residual
    R2 = residual[p:, :]                    # X projection residual

    # ------------------------------------------------------------
    # 4. Cost
    # ------------------------------------------------------------
    # if squared:
    #     # Faster and usually better conditioned than Frobenius norm.
    #     # This corresponds to the usual squared-Frobenius objective.
    #     cost1 = R1.square().sum()
    #     cost2 = R2.square().sum()
    # else:
        # Preserves your original objective exactly.
    cost1 = torch.linalg.matrix_norm(R1, ord="fro")
    cost2 = torch.linalg.matrix_norm(R2, ord="fro")
    # cost3 = torch.

    return (lambda1 * cost1) + (lambda2 * cost2)

def get_iGPK(
    SimData: torch.tensor,          # (num_traj, n, N+1)
    nTrain: int, nTest: int,
    lifting_order: int = 10,
    max_iter: int = 100,
    learn_rate: float = 0.01,
    opt_weights: list[float] = [1., 1., 0.01],
    routine: str = "Z_only",        # "Z_only" or "SpacedOpt"
    train_method: str = "Horizon",  # "Horizon" or "K-Means"
    hp_scale: list = [None, 1.0, None],  # [hp1, hp2, mu]
    device: str = "cuda:0",
    seed_z: int = 1234,
    seed_hp: int = 1234
):
    """
    Train iGPK, build Koopman (A, C), simulate train/test, and return predictions, covariances, NRMSE.

    NOTE: Data loading & noise addition remain outside. Pass prepped SimData in.
    """
    torch.manual_seed(seed_z)
    SimData = SimData.float().to(device)

    # Shapes & basic splits
    n = SimData.shape[1]
    N = SimData.shape[2] - 1
    p = lifting_order

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
        Xtrain = torch.cat([X[:, j*N: j*N + 1]
                           for j in range(nTrain)], dim=1)  # n x (nTrain)
        Z = torch.nn.Parameter(torch.rand(
            Xtrain.shape[1], p, device=device))   # Virtual Targets
        ObsManager = gpk.GPObservablesManager()
        for i in range(int(p)):
            ObsManager.add_observable(
                index=i, d=n, ns=nTrain, kernel_types=[
                    'Gaussian'],
                combination='sum', noise=1e-4, device=device
            )
        # for i in range(int(p/2), p):
        #     ObsManager.add_observable(
        #         index=i, d=n, ns=nTrain, kernel_types=[
        #             'Gaussian', 'ExpSineSqr'],
        #         combination='product', noise=1e-4, m=500, device=device
        #     )
        for i in range(p):
            ObsManager.train_observable(i, Xtrain, Z[:, i])
        torch.manual_seed(seed_hp)
        ObsManager.set_random_hyperparameters(
            scale=hp_scale, seed=seed_hp)

    elif train_method == "K-Means":
        Xtrain = torch.cat([X[:, j*N: j*N + 1] for j in range(nTrain)], dim=1)
        Z = torch.nn.Parameter(torch.rand(
            Xtrain.shape[1], p, device=device))   # Virtual Targets
        ObsManager = gpk.GPObservablesManager()
        centroids = gpk.get_kmeans(X, num_centers=p)
        for i in range(p):
            ObsManager.add_observable(
                index=i, d=n, ns=nTrain,
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
    lam1, lam2, lam3 = opt_weights
    iter = 0
    cost_history = []
    hp_opt_iter = int(0.0001*max_iter)
    num_hpopt = 0
    G_X, G_Xplus = build_G_cache(ObsManager, X, Xplus)

    optimizer = torch.optim.SGD(
        [Z], lr=learn_rate, momentum=0.85, nesterov=True)
    while iter < max_iter:
        optimizer.zero_grad(set_to_none=True)
        cost = get_cost_simple_fast(Z, X, Xplus, ObsManager, G_X, G_Xplus,
                               nT=nTrain, lambda1=lam1, lambda2=lam2, lambda3=lam3)
        cost.backward()
        optimizer.step()
        cost_history.append(cost.item())
        iter += 1

        if iter > 1000:
            rel_change = float((cost_history[-50] - cost_history[-1]) / cost_history[-50])
            if rel_change > 0 and rel_change < 1e-5:
                break
        if (routine == 'alternating') and (iter < max_iter) and ((iter % 1000) == 0):
            ObsManager.optimize_hyperparameters(
                opt_mu=False, opt_sigma=False, max_iter=2, lr=learn_rate)
            num_hpopt += 1
            G_X, G_Xplus = build_G_cache(ObsManager, X, Xplus)

    # === Retrain GPs at optimal Z & (optionally) optimize hp ===
    optimal_Z = Z.detach()
    for i in range(p):
        ObsManager.train_observable(i, Xtrain, optimal_Z[:, i])

    ObsManager.optimize_hyperparameters(
        opt_mu=False, opt_sigma=True, max_iter=250, lr=0.01)
    ObsManager.print_parameters(get_mu=False)

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
            "Xhat": XhatTrain,      # (nTrain, n, N)
            "Xcv":  XcvTrain,       # (nTrain, n, n, N)
            "NRMSE": TrainNRMSE     # (nTrain, n)
        },
        "Test": {
            "Xhat": XhatTest,       # (nTest, n, N)
            "Xcv":  XcvTest,        # (nTest, n, n, N)
            "NRMSE": TestNRMSE      # (nTest, n)
        },
        "history": {
            "cost": torch.tensor(cost_history).detach().cpu(),
            "iters": iter
        },
    }


def find_hp_init(SimData: torch.tensor, nTrain: int) -> float:
    def _stack_snapshot_pairs(batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        batch: (nB, n, T+1)
        Returns:
            X: (nB*T, n), Y: (nB*T, n)
        """
        n = batch.shape[1]
        X = batch[:, :, :-1].permute(0, 2, 1).reshape(-1,
                                                      n).detach().cpu().numpy()
        Y = batch[:, :,  1:].permute(
            0, 2, 1).reshape(-1, n).detach().cpu().numpy()
        return X, Y
    # ---------- build stacked (X,Y) ----------
    train_batch = SimData[:nTrain]
    Xtr, _ = _stack_snapshot_pairs(train_batch)
    Npts = Xtr.shape[0]

    max_pairs_to_store = 5_000_000  # ~5 million floats ~ 40MB
    num_pairs = Npts * (Npts - 1) // 2

    if num_pairs <= max_pairs_to_store:
        # Store all distances (exact median).
        dists = np.empty(num_pairs, dtype=np.float32)
        k = 0
        for i in range(Npts - 1):
            diff = Xtr[i + 1:] - Xtr[i]                 # (Npts-i-1, n)
            di = np.sqrt(np.sum(diff * diff, axis=1))    # (Npts-i-1,)
            dists[k: k + di.size] = di
            k += di.size
        hp_init = float(np.median(dists))
        return hp_init
    else:  # fallback for huge datasets
        rng = np.random.default_rng(0)
        # sample up to the cap
        sample_pairs = min(max_pairs_to_store, num_pairs)
        idx_i = rng.integers(0, Npts, size=sample_pairs, endpoint=False)
        idx_j = rng.integers(0, Npts, size=sample_pairs, endpoint=False)

        # Ensure i != j (resample conflicts)
        mask = idx_i == idx_j
        while np.any(mask):
            idx_j[mask] = rng.integers(
                0, Npts, size=int(mask.sum()), endpoint=False)
            mask = idx_i == idx_j

        diffs = Xtr[idx_i] - Xtr[idx_j]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        hp_init = float(np.median(dists))
        return hp_init


if __name__ == "__main__":
    system_name = 'Cart_data'
    train_frac, test_frac = 0.6, 0.4
    clip = None
    lifted_order = 35
    noise_type = 'uniform'
    # unused, samples, iterations, inner iterations
    MAX_ITER = 100000
    routine = "Z-only"
    # 1) Load + normalize
    SimData_raw, ts, num_traj, N, nTrain, nTest = gpk.load_SimData(
        system_name, train_frac, test_frac, clip=clip)
    SimData_clean, mu_vec, std_vec = gpk.normalize_data(
        SimData_raw, nTrain, N)

    # 2) Find Initial Hyperparameter
    HP_INIT = find_hp_init(SimData_clean, nTrain)
    print(f'Heuristic Kernel-lengthscale param found to be {HP_INIT:.3e}')

    # 2) Noise
    SimData = gpk.add_noise(SimData_clean, noise_type=noise_type,
                            intensity=0.0, seed=1234)

    print(f'==== Starting iGPK Model Identification ====')
    t0 = time.perf_counter()
    results = get_iGPK(SimData, nTrain, nTest, lifted_order,
                       MAX_ITER, learn_rate=0.0001,
                       opt_weights=[1.0, 1.0, 0.0], routine=routine,
                       train_method="Horizon", hp_scale=[None, HP_INIT, None])
    t_iGPK = time.perf_counter() - t0
    print(
        f'{lifted_order}-D iGPK model-ID with {results["history"]["iters"]}-epochs, finished in {t_iGPK:.2f} seconds')

    # unpack iGPK
    ObsManager = results["ObsManager"]
    A_igpk, C_igpk = results["A"], results["C"]
    XhatTrain, XcvhatTrain, TrainNRMSE = results["Train"][
        "Xhat"], results["Train"]["Xcv"], results["Train"]["NRMSE"]
    XhatTest,  XcvhatTest,  TestNRMSE = results["Test"][
        "Xhat"],  results["Test"]["Xcv"],  results["Test"]["NRMSE"]

    z_norm = torch.tensor([ObsManager.observables[i].y.norm() for i in range(lifted_order)])
    print(f'Min Norm of Z vectors:      {z_norm.min():.3e}')
    print(f'Median Norm of Z vectors:   {z_norm.median():.3e}')
    print(f'Mean Norm of Z vectors:     {z_norm.mean():.3e}')
    print(f'Max Norm of Z vectors:      {z_norm.max():.3e}')
    
    gpk.plot_eigen(A_igpk)

    gpk.plot_NRMSE_metrics([TrainNRMSE*100], [TestNRMSE*100], ['iGPK'])

    # 6) indices + timebase
    idx_trainMIN = torch.argmin(TrainNRMSE.mean(dim=1))
    idx_testMIN = torch.argmin(TestNRMSE.mean(dim=1))
    idx_testMAX = torch.argmax(TestNRMSE.mean(dim=1))
    time_arr = torch.arange(0., ts * (SimData.shape[2] - 1), ts)
    print(f'Median Test NMRSE:          {100*TestNRMSE.mean(dim=1).median():.2f}%')
    print(f'Mean Test NMRSE:            {100*TestNRMSE.mean(dim=1).mean():.2f}%')
    print(f'Example Initial Covariance {XcvhatTest[5, :, :, 0]}')

    # 7) pack models for overlay plot
    models = [
        {"name": "iGPK", "train": {"Xhat": XhatTrain, "Xcvhat": XcvhatTrain},
            "test": {"Xhat": XhatTest, "Xcvhat": XcvhatTest}}
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

    cost_history = results["history"].get("cost", None)
    # Plot Cost History
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('log(Cost)', color=color)
    ax1.plot(torch.log10(torch.abs(cost_history)), color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Cost', color=color)
    ax2.plot(cost_history, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()

    ### NLPD Calulation
    def _nlpd_one(y, mu, S, jitter=1e-8):
        """
        NLPD for a single multivariate Gaussian y~N(mu,S).
        y, mu: (n,)
        S: (n,n) covariance
        Returns scalar (float)
        """
        n = y.numel()
        S = 0.5 * (S + S.T)  # symmetrize
        S = S + jitter * torch.eye(n, dtype=S.dtype)
        try:
            L = torch.linalg.cholesky(S)
            logdet = 2.0 * torch.log(torch.diag(L)).sum()
            diff = (y - mu).view(n, 1)
            sol = torch.cholesky_solve(diff, L)
            quad = float((diff.T @ sol).item())
            return 0.5 * (n * math.log(2.0 * math.pi) + float(logdet) + quad)
        except Exception:
            # Diagonal fallback
            diag = torch.clamp(torch.diagonal(S), min=jitter)
            logdet = torch.log(diag).sum()
            quad = ((y - mu) ** 2 / diag).sum().item()
            return 0.5 * (n * math.log(2.0 * math.pi) + float(logdet) + quad)

    def _nlpd_per_traj(Xhat, Xcv, GT):
        """
        Average NLPD per trajectory across time-steps.
        returns (nTraj,) tensor
        """
        nTraj, n, N = Xhat.shape
        traj_vals = torch.zeros(nTraj, dtype=Xhat.dtype)
        for j in range(nTraj):
            acc = 0.0
            for k in range(N):
                acc += _nlpd_one(GT[j, :, k], Xhat[j, :, k],
                                torch.clamp(torch.abs(Xcv[j, :, :, k]), min=1e-6))
            traj_vals[j] = acc / N
        return traj_vals

    def _ms(x):
        return float(x.mean()), float(x.std(unbiased=False))

    GT_test = SimData[nTrain:nTrain+nTest, :, :N-1]  # (nTest, n, N)

    nlpd_traj_test_igpk = _nlpd_per_traj(
        XhatTest[:, :, :N-1],       XcvhatTest[:, :, :, :N-1],       GT_test).detach().cpu()

    # Print summary
    m, s = _ms(nlpd_traj_test_igpk)
    print(f"Test  NLPD iGPK:     mean={m:.4f}, std={s:.4f}")

    plt.show()
