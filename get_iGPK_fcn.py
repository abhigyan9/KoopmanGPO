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
    jitter=1e-6,
    Mp_X0=None, Mp_X=None, Mp_Xplus=None,
    num_total_samples: int | None = None,
):
    """
    Fast iGPK projection cost.

    Supports both full-batch and mini-batch evaluation.

    If num_total_samples is not None, the Frobenius norms are scaled by

        sqrt(num_total_samples / batch_samples)

    so that mini-batch costs are approximately on the same scale as the
    full-batch cost.
    """
    nz = Z.shape[1]
    Ns = Z.shape[0]
    dtype = G_X.dtype
    device = G_X.device

    # ------------------------------------------------------------
    # 1. Build lifted mean matrices M and Mplus
    # ------------------------------------------------------------
    if Mp_X0 is not None:
        lifted_residual = Z - Mp_X0
        M = Mp_X + torch.einsum("isr,ri->is", G_X, lifted_residual)
        Mplus = Mp_Xplus + torch.einsum("isr,ri->is", G_Xplus, lifted_residual)
    else:
        M = torch.einsum("isr,ri->is", G_X, Z)
        Mplus = torch.einsum("isr,ri->is", G_Xplus, Z)

    # ------------------------------------------------------------
    # 2. Compute Gram matrix in lifted space
    # ------------------------------------------------------------
    eye_p = torch.eye(nz, dtype=dtype, device=device)
    Gram = M @ M.mT

    try:
        L = torch.linalg.cholesky(Gram + jitter * eye_p)
    except RuntimeError:
        try:
            L = torch.linalg.cholesky(Gram + (10 * jitter) * eye_p)
        except RuntimeError:
            L = torch.linalg.cholesky(Gram + (100 * jitter) * eye_p)

    # ------------------------------------------------------------
    # 3. Compute B P_M without forming P_M
    # ------------------------------------------------------------
    B = torch.cat([Mplus, X], dim=0)  # (nz + nx, S_batch)

    BMt = B @ M.mT
    coeff = torch.cholesky_solve(BMt.mT, L).mT

    residual = B - coeff @ M

    R1 = residual[:nz, :]
    R2 = residual[nz:, :]

    # ------------------------------------------------------------
    # 4. Cost
    # ------------------------------------------------------------
    cost1 = torch.linalg.matrix_norm(R1, ord="fro")
    cost2 = torch.linalg.matrix_norm(R2, ord="fro")

    if num_total_samples is not None:
        batch_samples = X.shape[1]
        scale = (num_total_samples / batch_samples) ** 0.5
        cost1 = scale * cost1
        cost2 = scale * cost2

    return ((lambda1 * cost1) + (lambda2 * cost2)) / (nz * Ns)


def print_obs_fingerprint(ObsManager, tag, max_obs=5):
    print(f"\n===== OBS FINGERPRINT: {tag} =====")
    for i in range(min(max_obs, ObsManager.num_obs)):
        obs = ObsManager.observables[i]
        k = obs.kernel

        print(f"obs {i}")
        print("  obs device:", obs.device)
        print("  obs dtype :", obs.dtype)
        print("  Xtrain norm:", float(obs.Xtrain.detach().norm().cpu()) if obs.Xtrain is not None else None)
        print("  Xtrain sum :", float(obs.Xtrain.detach().sum().cpu()) if obs.Xtrain is not None else None)
        print("  ytrain norm:", float(obs.ytrain.detach().norm().cpu()) if obs.ytrain is not None else None)
        print("  ytrain sum :", float(obs.ytrain.detach().sum().cpu()) if obs.ytrain is not None else None)
        print("  noise     :", float(obs.noise.detach().cpu()))
        print("  raw_noise :", float(obs.raw_noise.detach().cpu()))

        if hasattr(k, "hp1"):
            print("  hp1       :", float(k.hp1.detach().cpu()))
            print("  hp2       :", float(k.hp2.detach().cpu()))
            print("  raw_hp1   :", float(k.raw_hp1.detach().cpu()))
            print("  raw_hp2   :", float(k.raw_hp2.detach().cpu()))
            print("  kernel eps/beta/thresh:", k.eps, k.beta, k.thresh)


def get_iGPK(
    Data: torch.tensor,          # (num_traj, n_x, N+1)
    nTrain: int,
    nTest: int,
    lifting_order: int = 10,
    max_iter: int = 100,
    sgd_lr: float = 0.01,
    sgd_m: float = 0.8,
    stop_tol: float = 1e-6,
    opt_weights: list[float] = [1., 1., 0.01],
    routine: str = "standard",        # OR "multi-perturb" - to be implemented
    train_method: str = "Zero-Mean",  # Zero-Mean | Monomials
    hp_scale: list[float] = [None, 1.0, None],  # [hp1, hp2, _]
    device: str | torch.device = "cuda:0",
    seed_z: int = 1234,
    seed_hp: int = 1234,

    # ------------------------------------------------------------
    # New batch-SGD controls
    # ------------------------------------------------------------
    traj_batch_size: int | None = None,
    full_cost_eval_every: int = 50,
):
    """
    Train iGPK using trajectory-wise mini-batch SGD on Z.

    traj_batch_size:
        Number of full trajectories used in each SGD step.
        If None or >= nTrain, this recovers full-batch SGD.

    full_cost_eval_every:
        Frequency at which the full training cost is evaluated for
        checkpointing and early stopping.
    """
    SimData = Data['SimData']
    X = Data['X']
    Xplus = Data['Xplus']
    ICsetTrain = Data['ICsetTrain']
    ICsetTest = Data['ICsetTest']
    Xtrain = Data['Xtrain']

    # ------------------------------------------------------------
    # Shapes and basic splits
    # ------------------------------------------------------------
    nx, N, Ns_gpo = Data['dims']
    nz = int(lifting_order)

    t0 = time.perf_counter()

    ObsManager = gpk.GPObservablesManager()

    # ------------------------------------------------------------
    # Initialize manager and decision variable Z
    # ------------------------------------------------------------
    if train_method == "Zero-Mean":

        torch.manual_seed(seed=seed_z)

        Z_raw = torch.zeros((Ns_gpo, nz))
        monomial_powers = generate_monomial_powers(nx, total_orders=(1,))
        num_monomial_means = min(nz, len(monomial_powers))

        for i in range(nz):
            if i < num_monomial_means:
                monomial = gpk.MonomialMean(powers=monomial_powers[i])
                Z_raw[:, i] = monomial(Xtrain).squeeze(dim=1)
            else:
                Z_raw[:, i] += torch.rand(Ns_gpo, 1).squeeze(dim=1)

        Z = torch.nn.Parameter(Z_raw.to(device=device))

        for i in range(nz):
            kernel = gpk.GaussianKernel()
            ObsManager.add_observable(
                index=i,
                d=nx,
                Ns=Ns_gpo,
                kernel=kernel,
                prior_mean=None,
                noise=1e-4,
                device=device,
                beta=20.0,
                thresh=20.0,
                eps=1e-8,
            )

        ObsManager.set_random_hyperparameters(seed=seed_hp, scale=hp_scale)

        for i in range(nz):
            ObsManager.train_observable(i, Xtrain, Z[:, i:i+1])

    elif train_method == "Monomials":
        torch.manual_seed(seed=seed_z)

        Z = torch.nn.Parameter(torch.rand(nTrain, nz, device=device))

        monomial_powers = generate_monomial_powers(nx, total_orders=(1, 2, 3))
        num_monomial_means = min(nz, len(monomial_powers))

        for i in range(nz):
            kernel = gpk.GaussianKernel()

            if i < num_monomial_means:
                prior_mean = gpk.MonomialMean(powers=monomial_powers[i])
            else:
                prior_mean = None

            ObsManager.add_observable(
                index=i,
                d=nx,
                Ns=Ns_gpo,
                kernel=kernel,
                prior_mean=prior_mean,
                noise=1e-6,
                device=device,
                beta=20.0,
                thresh=20.0,
                eps=1e-12,
            )

        ObsManager.set_random_hyperparameters(seed=seed_hp, scale=hp_scale)

        for i in range(nz): # train GPOs
            ObsManager.train_observable(i, Xtrain, Z[:, i:i+1])

    else:
        raise ValueError(f"Unrecognized train_method: {train_method}")

    # ------------------------------------------------------------
    # Optimization setup
    # ------------------------------------------------------------
    lam1, lam2, lam3 = opt_weights

    iter = 0
    cost_history = []
    full_cost_history = []
    grad_history = []

    # Build full GP smoother caches once, using the initial GP hyperparameters.
    # Mini-batches slice these cached tensors.
    G_X, G_Xplus = build_G_cache(ObsManager, X, Xplus)

    X_dev = X.to(device=device)
    Xplus_dev = Xplus.to(device=device)

    G_X = G_X.to(device=device)
    G_Xplus = G_Xplus.to(device=device)

    num_total_samples = X_dev.shape[1]  # N * Ns_gpo

    # Prior-mean caches for monomial prior case
    if train_method == "Monomials":
        Mp_X0 = torch.cat(
            [
                ObsManager.observables[i].prior_mean(Xtrain.to(device=device))
                for i in range(nz)
            ],
            dim=1,
        )  # (Ns_gpo, nz)

        Mp_X = torch.cat(
            [
                torch.transpose(
                    ObsManager.observables[i].prior_mean(X_dev),
                    dim0=0,
                    dim1=1,
                )
                for i in range(nz)
            ],
            dim=0,
        )  # (nz, N*Ns_gpo)

        Mp_Xplus = torch.cat(
            [
                torch.transpose(
                    ObsManager.observables[i].prior_mean(Xplus_dev),
                    dim0=0,
                    dim1=1,
                )
                for i in range(nz)
            ],
            dim=0,
        )  # (nz, N*Ns_gpo)

    else:
        Mp_X0, Mp_X, Mp_Xplus = None, None, None

    # Determine trajectory batch size
    if traj_batch_size is None:
        traj_batch_size = nTrain

    traj_batch_size = int(min(traj_batch_size, nTrain))

    optimizer = torch.optim.SGD([Z], lr=sgd_lr, momentum=sgd_m, nesterov=True)

    # ------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------
    def _trajectory_batch_columns(traj_idx: torch.Tensor) -> torch.Tensor:
        """
        Given trajectory indices shaped (traj_batch_size,), return all
        corresponding time-column indices in the concatenated X/Xplus layout.

        If trajectory j is selected, columns j*N ... j*N + N-1 are included.
        """
        time_idx = torch.arange(N, device=device)
        col_idx = traj_idx[:, None] * N + time_idx[None, :]
        return col_idx.reshape(-1)

    def _perturb_Z_():
        """Perturb Z in-place using a scale tied to the current Z magnitude."""
        with torch.no_grad():
            z_std = Z.detach().std()
            z_mean_abs = Z.detach().abs().mean()
            z_scale = torch.maximum(z_std, z_mean_abs)
            perturb_std = max(1e-4, 1e-2 * float(z_scale.item()))
            Z.add_(perturb_std * torch.randn_like(Z))

    def _full_cost_() -> float:
        """Evaluate the full training cost at the current Z."""
        with torch.no_grad():
            full_cost = get_cost_simple_fast(
                Z, X_dev, G_X, G_Xplus,
                lambda1=lam1, lambda2=lam2, lambda3=lam3,
                Mp_X0=Mp_X0, Mp_X=Mp_X, Mp_Xplus=Mp_Xplus,
                num_total_samples=None)

        return float(full_cost.item())
    # ------------------------------------------------------------
    # Initial full cost and checkpoint
    # ------------------------------------------------------------
    checkpoints = {}
    num_perturb = 0
    initial_full_cost = _full_cost_()

    best_full_cost = initial_full_cost
    best_iter = 0
    optimal_Z = Z.detach().clone()

    checkpoints["0"] = {
        "Z_val": optimal_Z.detach().clone(),
        "cost_val": best_full_cost,
    }

    last_full_cost = best_full_cost
    full_cost_history.append(best_full_cost)

    # ------------------------------------------------------------
    # Trajectory-wise batch-SGD loop
    # ------------------------------------------------------------
    while iter < max_iter:
        # Select a batch of complete trajectories
        if traj_batch_size == nTrain:
            traj_idx = torch.arange(nTrain, device=device)
        else:
            traj_idx = torch.randperm(nTrain, device=device)[:traj_batch_size]

        batch_cols = _trajectory_batch_columns(traj_idx)

        X_b = X_dev[:, batch_cols]
        G_X_b = G_X[:, batch_cols, :]
        G_Xplus_b = G_Xplus[:, batch_cols, :]

        if train_method == "Monomials":
            Mp_X_b = Mp_X[:, batch_cols]
            Mp_Xplus_b = Mp_Xplus[:, batch_cols]
        else:
            Mp_X_b, Mp_Xplus_b = None, None

        # Mini-batch SGD update
        optimizer.zero_grad(set_to_none=True)
        cost = get_cost_simple_fast(Z, X_b, G_X_b, G_Xplus_b,
            lambda1=lam1, lambda2=lam2, lambda3=lam3,
            Mp_X0=Mp_X0, Mp_X=Mp_X_b, Mp_Xplus=Mp_Xplus_b,
            num_total_samples=num_total_samples)
        cost.backward()
        optimizer.step()
        cost_history.append(float(cost.item()))

        if Z.grad is not None:
            grad_history.append(float(Z.grad.mean().item()))
        else:
            grad_history.append(float("nan"))

        iter += 1

        # Periodic full-cost evaluation for checkpointing/stopping
        do_full_eval = (
            iter == 1
            or iter % full_cost_eval_every == 0
            or iter == max_iter
        )

        if do_full_eval:
            full_cost_val = _full_cost_()
            full_cost_history.append(full_cost_val)

            if full_cost_val < best_full_cost:
                best_full_cost = full_cost_val
                best_iter = iter
                optimal_Z = Z.detach().clone()

                checkpoints[f"{iter}"] = {
                    "Z_val": optimal_Z.detach().clone(),
                    "cost_val": best_full_cost,
                }

            # Full cost is used for stopping/perturb
            if iter > 1000:
                rel_change = ( full_cost_history[-2] - full_cost_val
                    ) / max(abs(full_cost_history[-2]), 1e-12)
                
                if num_perturb < 20:
                    stagnated = rel_change > 0 and rel_change < (stop_tol)
                else:
                    stagnated = rel_change > 0 and rel_change < (stop_tol)

                if stagnated:
                    if routine == "multi-perturb" and num_perturb < 20:
                        num_perturb += 1

                        _perturb_Z_()
                        optimizer = torch.optim.SGD([Z],
                                        lr=sgd_lr, momentum=sgd_m, nesterov=True)

                        perturbed_full_cost_val = _full_cost_()
                        full_cost_history.append(perturbed_full_cost_val)

                        checkpoints[f"{iter}_perturb{num_perturb}"] = {
                            "Z_val": Z.detach().clone(),
                            "cost_val": perturbed_full_cost_val,
                        }

                        if perturbed_full_cost_val < best_full_cost:
                            best_full_cost = perturbed_full_cost_val
                            best_iter = iter
                            optimal_Z = Z.detach().clone()

                        full_cost_val = perturbed_full_cost_val

                    else:
                        break

            last_full_cost = full_cost_val

    # ------------------------------------------------------------
    # Make sure final iterate is considered
    # ------------------------------------------------------------
    final_full_cost_val = _full_cost_()

    if final_full_cost_val < best_full_cost:
        best_full_cost = final_full_cost_val
        best_iter = iter
        optimal_Z = Z.detach().clone()

    checkpoints[f"{iter}"] = {
        "Z_val": Z.detach().clone(),
        "cost_val": final_full_cost_val,
    }

    # ------------------------------------------------------------
    # Select optimal_Z from the lowest-cost checkpoint
    # ------------------------------------------------------------
    best_key, best_checkpoint = min(
        checkpoints.items(),
        key=lambda item: item[1]["cost_val"],
    )
    optimal_Z = best_checkpoint["Z_val"].detach().clone()
    best_full_cost = float(best_checkpoint["cost_val"])

    try:
        best_iter = int(str(best_key).split("_")[0])
        print(f'Iteration {best_iter} selected with full cost {best_full_cost:.6f}')
    except ValueError:
        print(f'Using last iter as best_iter')
        best_iter = iter

    # ------------------------------------------------------------
    # Retrain GPs at optimal Z
    # ------------------------------------------------------------
    for i in range(nz):
        ObsManager.train_observable(i, Xtrain, optimal_Z[:, i:i+1])

    # ------------------------------------------------------------
    # Optional MLE hyperparameter optimization
    # ------------------------------------------------------------
    ObsManager.optimize_hyperparameters(
        num_iter=100, lr=0.01, opt_noise=True)

    # ------------------------------------------------------------
    # Koopman A, C from full training data
    # ------------------------------------------------------------
    A, C = gpk.getKoopman(ObsManager, X, Xplus, nTrain, stateAug=False)

    t_iGPK = time.perf_counter() - t0

    # ------------------------------------------------------------
    # Train/test rollout evaluation
    # ------------------------------------------------------------
    XhatTrain, XcvTrain, TrainNRMSE = gpk.sim_and_eval(
        ObsManager,
        A,
        C,
        ICsetTrain,
        SimData,
        traj_offset=nTest,
    )

    XhatTest, XcvTest, TestNRMSE = gpk.sim_and_eval(
        ObsManager,
        A,
        C,
        ICsetTest,
        SimData,
        traj_offset=0,
    )

    # ------------------------------------------------------------
    # Post-MLE full cost
    # ------------------------------------------------------------
    with torch.no_grad():
        G_X_post, G_Xplus_post = build_G_cache(ObsManager, X, Xplus)
        G_X_post = G_X_post.to(device=device)
        G_Xplus_post = G_Xplus_post.to(device=device)

        post_mle_cost = get_cost_simple_fast(
            optimal_Z,
            X_dev,
            G_X_post,
            G_Xplus_post,
            lambda1=lam1,
            lambda2=lam2,
            lambda3=lam3,
            Mp_X0=Mp_X0,
            Mp_X=Mp_X,
            Mp_Xplus=Mp_Xplus,
            num_total_samples=None,
        )
    print(f'Num perturbations: {num_perturb}, Final full cost: {post_mle_cost:.6f}')
    # ------------------------------------------------------------
    # Package results
    # ------------------------------------------------------------
    return {
        "ObsManager": ObsManager,
        "A": A,
        "C": C,
        "ICsetTrain": ICsetTrain.detach().cpu(),
        "ICsetTest": ICsetTest.detach().cpu(),

        "Train": {
            "Xhat": XhatTrain,
            "Xcv": XcvTrain,
            "NRMSE": TrainNRMSE,
        },

        "Test": {
            "Xhat": XhatTest,
            "Xcv": XcvTest,
            "NRMSE": TestNRMSE,
        },

        "history": {
            "cost": torch.tensor(cost_history).detach().cpu(),
            "full_cost": torch.tensor(full_cost_history).detach().cpu(),
            "iters": iter,
            "best_iter": best_iter,
            "best_full_cost": best_full_cost,
            "opt_time": t_iGPK,
            "mean_grad": grad_history,
            "checkpoints": checkpoints,
            "post_mle_cost": post_mle_cost.detach().cpu(),
            "traj_batch_size": traj_batch_size,
        },
    }


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    SYSTEM_NAME = 'Lorenz96_8D'
    TRAIN_FRAC, TEST_FRAC = 0.8, 0.2
    CLIP = 50
    LIFTING_ORDER = 50
    NOISE_TYPE = 'gaussian'
    NOISE_INTENSITY = 0.0
    NOISE_SEED = 100
    # unused, samples, iterations, inner iterations
    MAX_ITER = int(500_000)
    OPT_WEIGHTS = [1.0, 1.0, 0.0]
    ROUTINE = "multi-perturb"  # OR "multi-perturb"
    TRAIN_METHOD = "Zero-Mean"
    DEVICE = "cuda:0"
    SEED_Z = 1234
    SEED_HP = 1234
    traj_batch_size = 32
    FULL_COST_EVAL_EVERY = 50
    # 1) Load + normalize
    SimData_raw, ts, num_traj, N, nTrain, nTest = gpk.load_SimData(
        SYSTEM_NAME, TRAIN_FRAC, TEST_FRAC, clip=CLIP)
    print(f'num_train: {nTrain}, num_test: {nTest}, num_steps: {N}')

    # SimData_raw = torch.flip(SimData_raw, dims=[0])
    SimData_clean, mu_vec, std_vec = gpk.normalize_data(
        SimData_raw.to(dtype=torch.float32), nTest, nTrain, N)

    # 2) Find Initial Hyperparameter
    HP_INIT = gpk.find_hp_init(SimData_clean[nTest:nTest+nTrain, :, :-1])
    print(f'Heuristic Kernel-lengthscale param found to be {HP_INIT:.3e}')
    hp_scale = [None, HP_INIT, None]

    # 2) Noise
    SimData = gpk.add_noise(SimData_clean, noise_type=NOISE_TYPE,
                            intensity=NOISE_INTENSITY, seed=NOISE_SEED)

    Dataset = {}
    nx = SimData.shape[1]
    N = SimData.shape[2] - 1
    Ns_gpo = 3 * nTrain
    Dataset['SimData'] = SimData
    Dataset['X'] = torch.cat([SimData[nTest+j, :, 0:N] for j in range(nTrain)],
                            dim=1)  # (nx, N*nTrain)
    Dataset['Xplus'] = torch.cat([SimData[nTest+j, :, 1:] for j in range(nTrain)],
                            dim=1)  # (nx, N*nTrain)
    Dataset['ICsetTrain'] = torch.cat([SimData[nTest+j, :, 0].view(nx, 1) 
        for j in range(nTrain)], dim=1)
    Dataset['ICsetTest'] = torch.cat([SimData[j, :, 0].view(nx, 1)
        for j in range(nTest)], dim=1)
    Dataset['Xtrain'] = gpk.get_kmeans(Dataset['X'], num_centers=Ns_gpo)
    Dataset['dims'] = (nx, N, Ns_gpo)

    print(f'==== Starting iGPK Model Identification ====')

    results = get_iGPK(
        Data=Dataset,
        nTrain=nTrain,
        nTest=nTest,
        lifting_order=LIFTING_ORDER,
        max_iter=MAX_ITER,
        sgd_lr=1e-2,
        sgd_m=0.8,
        stop_tol=1e-4,
        opt_weights=OPT_WEIGHTS,
        routine=ROUTINE,
        train_method=TRAIN_METHOD,
        hp_scale=hp_scale,
        device=DEVICE,
        seed_z=SEED_Z,
        seed_hp=SEED_HP,

        # NEW: trajectory-wise batch-SGD arguments
        traj_batch_size=traj_batch_size,
        full_cost_eval_every=FULL_COST_EVAL_EVERY,
    )

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
        
        print(f'Lifted Model Order:         {LIFTING_ORDER:d}')
        print(f'Total Epochs executed:      {total_epochs:d}')
        print(f'Learning Time:              {t_iGPK:.2f} seconds')
        
        if A_igpk.shape[0] <= 50:
            gpk.plot_eigen(A_igpk)
        gpk.MatViz(C_igpk, 'heat')
        gpk.plot_NRMSE_metrics([TrainNRMSE*100], [TestNRMSE*100], ['iGPK'])

        # 6) indices + timebase
        idx_trainMIN = torch.argmin(TrainNRMSE.mean(dim=1))
        idx_testMIN = torch.argmin(TestNRMSE.mean(dim=1))
        idx_testMAX = torch.argmax(TestNRMSE.mean(dim=1))
        time_arr = torch.arange(0., ts * (SimData.shape[2] - 1), ts)
        print(f'Median Test NMRSE:          {100*TestNRMSE.mean(dim=1).median():.2f}%')
        print(f'Mean Test NMRSE:            {100*float(TestNRMSE.mean()):.2f}%')

        # 7) pack models for overlay plot
        models = [
            {"name": "iGPK", "train": {"Xhat": XhatTrain, "Xcvhat": XcvhatTrain},
                "test": {"Xhat": XhatTest, "Xcvhat": XcvhatTest}}
        ]

        # a) 3 trajectory overlays
        for (which, idx, split, sim_offset, suffix) in [
            ("best-train", idx_trainMIN, "train", nTest,         "Best_Train"),
            ("best-test",  idx_testMIN,  "test",  0,    "Best_Test"),
            ("worst-test", idx_testMAX,  "test",  0,    "Worst_Test"),
        ]:
            gpk.compare_model_predictions(
                time=time_arr, models=models, SimData=SimData, idx=idx, N=(
                    SimData.shape[2]-1),
                system_name=SYSTEM_NAME, title_suffix=suffix, split=split, sim_offset=sim_offset,
                compare_to="SimData_clean", SimData_clean=SimData_clean, sigma=1.0
            )

        cost_history = results["history"].get("full_cost", None)
        # Plot Cost History
        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('log(Full Cost)', color=color)
        ax1.plot(torch.log10(torch.abs(cost_history)), color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, which='both', linestyle='--', alpha=0.7)
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Full Cost', color=color)
        ax2.plot(cost_history, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()

        print(f'Post-MLE Full Cost: {results['history']['post_mle_cost']:.3e}')
        # plt.plot(results["history"]['mean_grad'])

        ### NLPD Calulation
        def _ms(x):
            return float(x.mean()), float(x.std(unbiased=False))

        GT_test = SimData[:nTest, :, :N-1]  # (nTest, n, N)

        nlpd_traj_test_igpk = gpk.nlpd_per_traj(
            XhatTest[:, :, :N-1], XcvhatTest[:, :, :, :N-1], GT_test).detach().cpu()

        # Print summary
        m, s = _ms(nlpd_traj_test_igpk)
        print(f"Test  NLPD iGPK:     mean={m:.4f}, std={s:.4f}")

        plt.show()
