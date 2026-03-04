## --- IMPORTS --- ###
import GPKoopman as gpk
import torch
import matplotlib.pyplot as plt
import time
import numpy as np

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


def get_iGPK(
    SimData: torch.tensor,          # (num_traj, n, N+1)
    nTrain: int, nTest: int,
    lifting_order: int = 10,
    max_iter: int = 100,
    learn_rate: float = 0.01,
    opt_weights: list[float] = [1., 1., 1.],
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
        for i in range(p):
            ObsManager.add_observable(
                index=i, d=n, ns=nTrain, kernel_types=['Gaussian'],
                combination='sum', noise=1e-4, m=500, device=device
            )
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
    lam1, lam2, _ = opt_weights
    iter = 0
    cost_history = []
    hp_opt_iter = int(0.02*max_iter)
    num_hpopt = 0

    optimizer = torch.optim.SGD(
        [Z], lr=learn_rate, momentum=0.7, nesterov=True)
    while iter < max_iter:
        optimizer.zero_grad()
        cost = get_cost_simple(Z, X, Xplus, ObsManager,
                               nT=nTrain, lambda1=lam1, lambda2=lam2)
        cost.backward()
        optimizer.step()
        cost_history.append(cost.item())
        iter += 1
        if (routine == 'alternating') and (iter < max_iter) and ((iter % 25) == 0):
            ObsManager.optimize_hyperparameters(
                opt_mu=False, opt_sigma=True, max_iter=hp_opt_iter, lr=learn_rate)
            num_hpopt += 1

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

    with torch.no_grad():
        final_train_cost = get_cost_simple(
            optimal_Z, X, Xplus, ObsManager, nTrain)

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
            "cost": torch.tensor(cost_history).detach().cpu()
        },
        "final_train_cost": final_train_cost.detach().cpu()
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
    system_name = 'Lorenz'
    train_frac, test_frac = 0.4, 0.6
    clip = None
    lifted_order = 40
    noise_type = 'gaussian'
    # unused, samples, iterations, inner iterations
    iters_list = [0, 8, 20, 500]
    routine = "alternating"
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
                            intensity=0., seed=1234)

    print(f'==== Starting iGPK Model Identification ====')
    t0 = time.perf_counter()
    results = get_iGPK(SimData, nTrain, nTest, lifted_order,
                       iters_list, learn_rate=0.001,
                       opt_weights=[1.0, 1.0, 1.0], routine=routine,
                       train_method="Horizon", hp_scale=[None, HP_INIT, None])
    t_iGPK = time.perf_counter() - t0
    print(
        f'{lifted_order}-D iGPK model identification with {iters_list[3]}-iterations, finished in {t_iGPK:.2f} seconds')

    # unpack iGPK
    A_igpk, C_igpk = results["A"], results["C"]
    XhatTrain, XcvhatTrain, TrainNRMSE = results["Train"][
        "Xhat"], results["Train"]["Xcv"], results["Train"]["NRMSE"]
    XhatTest,  XcvhatTest,  TestNRMSE = results["Test"][
        "Xhat"],  results["Test"]["Xcv"],  results["Test"]["NRMSE"]

    gpk.plot_eigen(A_igpk)

    gpk.plot_NRMSE_metrics([TrainNRMSE*100], [TestNRMSE*100], ['iGPK'])

    # 6) indices + timebase
    idx_trainMIN = torch.argmin(TrainNRMSE.mean(dim=1))
    idx_testMIN = torch.argmin(TestNRMSE.mean(dim=1))
    idx_testMAX = torch.argmax(TestNRMSE.mean(dim=1))
    time_arr = torch.arange(0., ts * (SimData.shape[2] - 1), ts)

    # 7) pack models for overlay plot
    models = [
        {"name": "iGPK", "train": {"Xhat": XhatTrain},
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

    cost_history = results["history"].get("cost", None)
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
    plt.close(fig)
