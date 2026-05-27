## --- IMPORTS --- ###
import GPKoopman as gpk
import torch
import matplotlib.pyplot as plt
import time
import numpy as np
import math

from itertools import combinations_with_replacement

def generate_monomial_powers(nx: int, total_orders=(2, 3)):
    """
    Return exponent tuples for all unique monomials whose total degree
    is in `total_orders`.

    Example for nx=2:
        degree 2: (2,0), (1,1), (0,2)
        degree 3: (3,0), (2,1), (1,2), (0,3)
    """
    power_list = []

    for order in total_orders:
        for combo in combinations_with_replacement(range(nx), order):
            powers = [0] * nx
            for idx in combo:
                powers[idx] += 1
            power_list.append(tuple(powers))

    return power_list

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


@torch.no_grad()
def build_G_cache(manager, X, Xplus) -> tuple[torch.Tensor]:
    """
    Build detached G caches for Z-only optimization.

    Returns
    -------
    G_X      : (nz, S, r)
    G_Xplus  : (nz, S, r)

    where:
        nz = number of observables
        S = N*nT
        r = number of virtual targets
    """
    nz = manager.num_obs
    r = manager.observables[0].Ns # Z.shape[0]
    S = X.shape[1]

    G_X_list, G_Xplus_list = [], []

    for i in range(nz):
        obs = manager.observables[i]

        Gi = obs.forward_G(X)       # X : (nx, N*nT) -> Gi : (N*nT, r)
        Gpi = obs.forward_G(Xplus)  # (N*nT, r)

        G_X_list.append(Gi)
        G_Xplus_list.append(Gpi)

    G_X = torch.stack(G_X_list, dim=0).contiguous()
    G_Xplus = torch.stack(G_Xplus_list, dim=0).contiguous()

    return G_X, G_Xplus


def get_cost_simple_fast(
    Z, X, G_X, G_Xplus,
    lambda1=1.0, lambda2=1.0, lambda3=1.0,
    jitter=1e-6, Mp_X0=None, Mp_X=None, Mp_Xplus=None):
    """
    Fast version of get_cost_simple.

    Avoids explicitly forming:
        M_pinvM = M_pinv @ M

    """
    nz = Z.shape[1]
    nT = Z.shape[0]
    dtype = G_X.dtype
    device = G_X.device
    # ------------------------------------------------------------
    # 1. Build lifted mean matrices M and Mplus
    if Mp_X0 is not None:
        lifted_residual = Z - Mp_X0
        M = Mp_X + torch.einsum("isr,ri->is", G_X, lifted_residual)
        Mplus = Mp_Xplus + torch.einsum("isr,ri->is", G_Xplus, lifted_residual)
    else:
        M = torch.einsum("isr,ri->is", G_X, Z)
        Mplus = torch.einsum("isr,ri->is", G_Xplus, Z)

    # ------------------------------------------------------------
    # 2. Compute Gram matrix in lifted space
    eye_p = torch.eye(nz, dtype=dtype, device=device)
    Gram = M @ M.mT # (p, p)

    try:
        L = torch.linalg.cholesky(Gram + jitter * eye_p)
    except RuntimeError:
        try:
            L = torch.linalg.cholesky(Gram + (10*jitter) * eye_p)
        except RuntimeError:
            L = torch.linalg.cholesky(Gram + (100*jitter) * eye_p)

    # ------------------------------------------------------------
    # 3. Compute B P_M without forming P_M
    # B contains both terms whose projection residual we need:
    #   Mplus P_M and X P_M
    B = torch.cat([Mplus, X], dim=0)         # (p+n, S)

    # Use cholesky_solve for stability:
    BMt = B @ M.mT                          # (p+n, p)
    coeff = torch.cholesky_solve(BMt.mT, L).mT  # (p+n, p)

    residual = B - coeff @ M                # (p+n, S)

    R1 = residual[:nz, :]       # Mplus projection residual
    R2 = residual[nz:, :]       # X projection residual

    # ------------------------------------------------------------
    # 4. Cost
    cost1 = torch.linalg.matrix_norm(R1, ord="fro")
    cost2 = torch.linalg.matrix_norm(R2, ord="fro")

    return ((lambda1 * cost1) + (lambda2 * cost2)) / (nz * nT)

def get_iGPK(
    SimData: torch.tensor,          # (num_traj, n_x, N+1)
    nTrain: int, nTest: int,
    lifting_order: int = 10,
    max_iter: int = 100,
    sgd_lr : float = 0.01, sgd_m : float = 0.8, stop_tol : float = 1e-6,
    opt_weights: list[float] = [1., 1., 0.01],
    routine: str = "Z_only",        # "Z_only" or "SpacedOpt"
    train_method: str = "Zero-Mean",  # Zero-Mean | Monomials
    hp_scale: list = [None, 1.0, None],  # [hp1, hp2, _]
    device: str | torch.device = "cuda:0",
    seed_z: int = 1234,
    seed_hp: int = 1234
):
    """
    Train iGPK, build Koopman (A, C), simulate train/test, and return predictions, covariances, NRMSE.

    NOTE: Data loading & noise addition remain outside. Pass prepped SimData in.
    """
    SimData = SimData.to(dtype=torch.float32)
    import warnings
    warnings.filterwarnings("ignore")
    # Shapes & basic splits
    nx = SimData.shape[1]
    N = SimData.shape[2] - 1
    nz = int(lifting_order)
 
    # Build concatenated matrices and ICs from SimData
    # Xall = torch.cat([SimData[j, :, :]
    #                  for j in range(nTrain)], dim=1)    # (nx, (N+1)*nTrain)
    X = torch.cat([SimData[j, :, 0:N]
                  for j in range(nTrain)], dim=1)       # (nx, N*nTrain)
    Xplus = torch.cat([SimData[j, :, 1:]
                      for j in range(nTrain)], dim=1)   # (n, N*nTrain)

    ICsetTrain = torch.cat([SimData[j, :, 0].view(nx, 1)
                           for j in range(nTrain)], dim=1)
    ICsetTest = torch.cat([SimData[j, :, 0].view(nx, 1)
                          for j in range(nTrain, nTrain + nTest)], dim=1)

    t0 = time.perf_counter()
    ObsManager = gpk.GPObservablesManager()

    # Initialize manager & decision variable Z (training grid)
    if train_method == "Zero-Mean":
        Xtrain = torch.cat([X[:, j*N : j*N+1]
                            for j in range(nTrain)], dim=1)  # (nx, nTrain)
        torch.manual_seed(seed=seed_z)
        # Z = torch.nn.Parameter(torch.rand(
        #     nTrain, nz, device=device))   # Virtual Targets, (nTrain, nz)

        Z_raw = torch.zeros((nTrain, nz))
        monomial_powers = generate_monomial_powers(nx, total_orders=(1, 2, 3))
        num_monomial_means = min(nz, len(monomial_powers))
        for i in range(nz):
            if i < num_monomial_means:
                monomial = gpk.MonomialMean(powers=monomial_powers[i])
                Z_raw[:, i] = monomial(Xtrain).squeeze(dim=1)
            else:
                monomial = None
                Z_raw[:, i] = hp_scale[1] * torch.rand(nTrain, 1).squeeze(dim=1)
            
        Z = torch.nn.Parameter(Z_raw.to(device=device))

        for i in range(nz):
            kernel = gpk.GaussianKernel()
            ObsManager.add_observable(
                index=i, d=nx, Ns=nTrain, kernel=kernel,
                prior_mean=None, noise=1e-6, device=device,
                beta=20.0, thresh=20.0, eps=1e-12)
        ObsManager.set_random_hyperparameters(seed=seed_hp, scale=hp_scale)
        for i in range(nz):
            ObsManager.train_observable(i, Xtrain, Z[:, i].unsqueeze(dim=1))

    elif train_method == "Monomials":
        Xtrain = torch.cat([X[:, j*N: j*N + 1]
                            for j in range(nTrain)], dim=1)  # (nx, nTrain)
        torch.manual_seed(seed=seed_z)
        Z = torch.nn.Parameter(torch.rand(
            nTrain, nz, device=device))   # Virtual Targets, (nTrain, nz)
        
        monomial_powers = generate_monomial_powers(nx, total_orders=(1, 2, 3))
        num_monomial_means = min(nz, len(monomial_powers))
        for i in range(nz):
            kernel = gpk.GaussianKernel()

            if i < num_monomial_means:
                prior_mean = gpk.MonomialMean(powers=monomial_powers[i])
            else:
                prior_mean = None
            
            ObsManager.add_observable(
                index=i, d=nx, Ns=nTrain, kernel=kernel,
                prior_mean=prior_mean, noise=1e-6, device=device,
                beta=20.0, thresh=20.0, eps=1e-12)

        ObsManager.set_random_hyperparameters(seed=seed_hp, scale=hp_scale)
            
        for i in range(nz):
            ObsManager.train_observable(i, Xtrain, Z[:, i].unsqueeze(dim=1))

    else:
        raise ValueError(f"Unrecognized train_method: {train_method}")

    # === Optimization ===
    lam1, lam2, lam3 = opt_weights
    iter = 0
    cost_history, grad_history = [], []
    num_perturb = 0
    G_X, G_Xplus = build_G_cache(ObsManager, X, Xplus)
    if train_method == "Monomials":
        Mp_X0 = torch.cat([ObsManager.observables[i].prior_mean(
            Xtrain.to(device=device)) for i in range(nz)], dim=1)
        Mp_X = torch.cat([torch.transpose(ObsManager.observables[i].prior_mean(
            X.to(device=device)), dim0=0, dim1=1) for i in range(nz)], dim=0)
        Mp_Xplus = torch.cat([torch.transpose(ObsManager.observables[i].prior_mean(
            Xplus.to(device=device)), dim0=0, dim1=1) for i in range(nz)], dim=0)
    else:
        Mp_X0, Mp_X, Mp_Xplus = None, None, None
    if sgd_m == 0.0 or sgd_m is None:
        optimizer = torch.optim.SGD(
            [Z], lr=sgd_lr, nesterov=False)
    else:
        optimizer = torch.optim.SGD(
            [Z], lr=sgd_lr, momentum=sgd_m, nesterov=True)
    # ObsManager.print_hyperparameters(indices=None)
    checkpoints = {}
    while iter < max_iter:
        optimizer.zero_grad(set_to_none=True)
        cost = get_cost_simple_fast(Z, X.to(device=device), G_X, G_Xplus,
                               lambda1=lam1, lambda2=lam2, lambda3=lam3,
                               Mp_X0=Mp_X0, Mp_X=Mp_X, Mp_Xplus=Mp_Xplus)
        cost.backward()
        optimizer.step()
        cost_history.append(cost.item())
        grad_history.append(Z.grad.mean().item())
        iter += 1

        if iter > 1000: # STOPPING CONDITION
            rel_change = float((cost_history[-50] - cost_history[-1]) / cost_history[-50])

            if rel_change > 0 and rel_change < stop_tol:
                break
                with torch.no_grad():
                    continue
                    checkpoints[f'{iter}'] = {"Z_val": Z.detach(),
                                              "cost_val": cost.item()}
                    scale = (0.5) * Z.clone().detach().std().clamp(min=1e-6)
                    Z.add_((scale * (torch.randn_like(Z))))
                    state = optimizer.state.get(Z, None)
                    state["momentum_buffer"].zero_()
                    num_perturb += 1
    
    checkpoints[f'{iter}'] = {"Z_val": Z.detach(), "cost_val": cost.item()}

    # === Retrain GPs at optimal Z & (optionally) optimize hp ===
    checkpoint_cost = 100.0
    for iter_val, item in checkpoints.items():
        if item['cost_val'] <= checkpoint_cost:
            checkpoint_cost = item['cost_val']
            optimal_Z = item['Z_val'].detach()
            used_iter = iter_val

    for i in range(nz):
        ObsManager.train_observable(i, Xtrain, optimal_Z[:, i].unsqueeze(dim=1))

    ObsManager.optimize_hyperparameters(num_iter=200, lr=0.1, opt_noise=True)
    # print(f'Number of Perturb-and-Restarts: {num_perturb}')
    # === Koopman A, C ===
    A, C = gpk.getKoopman(ObsManager, X, Xplus, nTrain, stateAug=False)
    t_iGPK = time.perf_counter() - t0
    # ObsManager.print_hyperparameters(indices=None)
    
    # Train split (offset 0), Test split (offset nTrain)
    XhatTrain, XcvTrain, TrainNRMSE = gpk.sim_and_eval(
        ObsManager, A, C, ICsetTrain, SimData, traj_offset=0)
    XhatTest,  XcvTest,  TestNRMSE = gpk.sim_and_eval(
        ObsManager, A, C, ICsetTest,  SimData, traj_offset=nTrain)
    with torch.no_grad():
        G_X, G_Xplus = build_G_cache(ObsManager, X, Xplus)
        post_mle_cost = get_cost_simple_fast(optimal_Z, X.to(device=device), G_X, G_Xplus,
                                lambda1=lam1, lambda2=lam2, lambda3=lam3,
                                Mp_X0=Mp_X0, Mp_X=Mp_X, Mp_Xplus=Mp_Xplus)
    # === Package results ===
    return {
        "ObsManager": ObsManager,   # GPObservablesManager object
        "A": A, "C": C, # tensor
        "ICsetTrain": ICsetTrain.detach().cpu(),    # tensor
        "ICsetTest":  ICsetTest.detach().cpu(), # tensor
        "Train": {
            "Xhat": XhatTrain,      # (nTrain, nx, N)
            "Xcv":  XcvTrain,       # (nTrain, nx, nx, N)
            "NRMSE": TrainNRMSE     # (nTrain, nx)
        },  # tensors
        "Test": {
            "Xhat": XhatTest,       # (nTest, nx, N)
            "Xcv":  XcvTest,        # (nTest, nx, nx, N)
            "NRMSE": TestNRMSE      # (nTest, nx)
        }, # tensors
        "history": {
            "cost": torch.tensor(cost_history).detach().cpu(),
            "iters": iter,                  # int : final iteration number
            "opt_time": t_iGPK,             # float : total optimization time
            "mean_grad": grad_history,      # list[float]
            "checkpoints": checkpoints,     # dict
            "post_mle_cost": post_mle_cost, # torch.tensor
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
    import warnings
    warnings.filterwarnings("ignore")
    system_name = 'Cart_data'
    train_frac, test_frac = 0.6, 0.4
    clip = None
    lifted_order = 35
    noise_type = 'uniform'
    # unused, samples, iterations, inner iterations
    MAX_ITER = int(1e6)
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
                            intensity=0.0, seed=100)

    print(f'==== Starting iGPK Model Identification ====')
    results = get_iGPK(SimData, nTrain, nTest, lifted_order,
                       MAX_ITER, sgd_lr=0.005, sgd_m=0.85,
                       opt_weights=[10.0, 10.0, 0.0], routine=routine,
                       train_method="Zero-Mean", hp_scale=[None, HP_INIT, None])

    if True:    # All Post-Processing and Outputs
        # unpack iGPK
        ObsManager = results["ObsManager"]
        A_igpk, C_igpk = results["A"], results["C"]
        XhatTrain, XcvhatTrain, TrainNRMSE = results["Train"][
            "Xhat"], results["Train"]["Xcv"], results["Train"]["NRMSE"]
        XhatTest,  XcvhatTest,  TestNRMSE = results["Test"][
            "Xhat"],  results["Test"]["Xcv"],  results["Test"]["NRMSE"]
        t_iGPK, total_epochs = results['history']['opt_time'], results[
            "history"]['iters']
        
        print(f'Lifted Model Order:         {lifted_order:d}')
        print(f'Total Epochs executed:      {total_epochs:d}')
        print(f'Learning Time:              {t_iGPK:.2f} seconds')
        
        gpk.plot_eigen(A_igpk)
        gpk.MatViz(C_igpk, 'heat')
        TrainNRMSE = TrainNRMSE.clamp(max=1.5)
        TestNRMSE = TestNRMSE.clamp(max=1.5)
        gpk.plot_NRMSE_metrics([TrainNRMSE*100], [TestNRMSE*100], ['iGPK'])

        # 6) indices + timebase
        idx_trainMIN = torch.argmin(TrainNRMSE.mean(dim=1))
        idx_testMIN = torch.argmin(TestNRMSE.mean(dim=1))
        idx_testMAX = torch.argmax(TestNRMSE.mean(dim=1))
        time_arr = torch.arange(0., ts * (SimData.shape[2] - 1), ts)
        print(f'Median Test NMRSE:          {100*TestNRMSE.mean(dim=1).median():.2f}%')
        print(f'Mean Test NMRSE:            {100*TestNRMSE.mean(dim=1).mean():.2f}%')

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

        print(f'Post-MLE Cost: {results['history']['post_mle_cost']:.3e}')
        # plt.plot(results["history"]['mean_grad'])

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
