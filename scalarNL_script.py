## --- IMPORTS --- ###
import GPKoopman as gpk
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from get_iGPK_fcn import get_iGPK
import os
from datetime import datetime

## --- HELPER FUNCTIONS --- ##


def _save(fig, outdir, fname_stub: str):
    """
    Helper to persist figures on disk.  Creates the output directory if it
    doesn't yet exist and writes a PNG file with reasonable DPI.  The
    filename is constructed from the provided stub.
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{fname_stub}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    # print(f"saved: {path}")


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


def _diag_std_from_cov(Xcvhat, LB=1e-8):
    # Xcvhat: (nTraj, n, n, N) -> std: (nTraj, n, N)
    nTraj, n, _, N = Xcvhat.shape
    temp = torch.clamp(torch.sqrt(
        torch.abs(torch.diagonal(Xcvhat, dim1=1, dim2=2))), min=LB)
    return torch.reshape(temp, (nTraj, n, N))


def compute_interval_coverage(Xhat, Xcvhat, SimData, sim_offset=0, alpha_levels=[0.90, 0.95]):
    """
    Computes interval coverage metrics for specified alpha levels (e.g., 0.90, 0.95)
    using predictive mean and covariance from the model.

    Args:
        Xhat:     (nTraj, n, N) predictive mean trajectories
        Xcvhat:   (nTraj, n, n, N) predictive covariance trajectories
        SimData:  (nTraj_total, n, N_total) ground truth trajectories
        sim_offset: offset if test data is after training set
        alpha_levels: list of confidence levels (default [0.90, 0.95])
    Returns:
        coverage_dict: {alpha: per-state coverage fraction tensor (n,)}
    """
    nTraj, n, N = Xhat.shape
    coverage_dict = {}
    quantiles = {0.90: 1.645, 0.95: 1.96}  # standard normal quantiles

    # Stack all predictions for simplicity
    y_true = SimData[sim_offset:sim_offset +
                     nTraj, :, :N]          # (nTraj, n, N)
    mu = Xhat
    sigma = _diag_std_from_cov(Xcvhat)

    for alpha in alpha_levels:
        z = quantiles.get(alpha, 1.96)
        lower = mu - z * sigma
        upper = mu + z * sigma
        inside = (y_true >= lower) & (y_true <= upper)
        coverage_statewise = inside.float().mean(
            dim=(0, 2))  # average over trajs & time
        coverage_dict[alpha] = coverage_statewise.cpu()
    return coverage_dict


def coverage_curve(Xhat, Xcvhat, SimData, sim_offset=0, alphas=None, reduce="mean"):
    """
    Compute empirical coverage for 1D normal intervals across a grid of nominal levels.
    Args:
        Xhat:     (nTraj, n, N) predictive mean
        Xcvhat:   (nTraj, n, n, N) predictive covariance
        SimData:  (nTraj_total, n, N_total) ground truth
        sim_offset: offset index in SimData (e.g., nTrain for test set)
        alphas: list/1D-tensor of nominal coverages in [0,1] (e.g., 0.50..0.99)
        reduce: "mean" → average over states; "none" → return per-state coverage
    Returns:
        alphas (tensor), empirical (tensor of shape (len(alphas),) or (len(alphas), n))
    """
    if alphas is None:
        alphas = torch.linspace(0.50, 0.99, steps=50, device=Xhat.device)
    alphas = torch.as_tensor(alphas, dtype=Xhat.dtype, device=Xhat.device)

    # Standard normal quantile via erfinv: z = sqrt(2)*erfinv(alpha)
    # (Strictly, two-sided interval half-width uses z such that P(|Z|<=z)=alpha, i.e., z = Φ^{-1}((1+alpha)/2))
    from torch import special
    z = torch.special.ndtri((1 + alphas) / 2)  # (A,)

    y_true = SimData[sim_offset:sim_offset +
                     Xhat.shape[0], :, :Xhat.shape[2]]  # (nTraj, n, N)
    mu = Xhat
    sigma = _diag_std_from_cov(Xcvhat)
    nT, n, N = mu.shape
    # Broadcast: for each alpha/z, build interval and test coverage
    # Shapes:
    #   mu, sigma, y_true : (T, n, N)
    #   z[:,None,None,None] → (A,1,1,1)
    lower, upper = torch.zeros((len(alphas), nT, n, N)), torch.zeros(
        (len(alphas), nT, n, N))
    for i in range(alphas.shape[0]):
        lower[i, :, :, :] = mu - z[i] * sigma
        upper[i, :, :, :] = mu + z[i] * sigma
    inside = (y_true >= lower) & (y_true <= upper)
    emp_coverage_state = inside.float().mean(dim=(1, 3))
    # lower = mu. - z.view(-1, 1, 1, 1) * sigma.unsqueeze(0)  # (A,T,n,N)
    # upper = mu.unsqueeze(0) + z.view(-1, 1, 1, 1) * \
    #     sigma.unsqueeze(0)  # (A,T,n,N)
    # inside = (y_true.unsqueeze(0) >= lower) & (
    #     y_true.unsqueeze(0) <= upper)  # (A,T,n,N)

    # Per-state empirical coverage
    # emp_state = inside.float().mean(dim=(1, 3))  # (A, n) avg over traj & time

    if reduce == "mean":
        # mean over states, keep one value per alpha: (A,)
        return alphas, emp_coverage_state.mean(dim=1)        # <-- FIXED
    elif reduce == "none":
        return alphas, emp_coverage_state
    else:
        raise ValueError("reduce must be 'mean' or 'none'")


def miscalibration_area(alphas, empirical):
    """
    L1 area between empirical and nominal coverage curves.
    Args:
        alphas:    (A,)
        empirical: (A,) mean curve (use reduce='mean') or (A,n) per-state
    Returns:
        scalar if 1D, or (n,) if per-state
    """
    # Trapezoidal rule on |emp - alpha|
    diff = (empirical - alphas.unsqueeze(-1)
            ) if empirical.ndim == 2 else (empirical - alphas)
    area = torch.trapz(diff.abs(), alphas, dim=0)
    return area


def plot_calibration_curve(alphas, empirical, title="Calibration Curve", label=None):
    """
    One figure per curve set. Draws y=x reference and a single empirical curve.
    """
    # ensure 1D cpu numpy arrays
    a = alphas.detach().cpu().view(-1).numpy()
    e = empirical.detach().cpu().view(-1).numpy()

    fig, ax = plt.subplots()
    ax.plot(a, a, linestyle='--', label='Ideal')
    ax.plot(a, e, marker='o', linestyle='-',
            label=label if label else 'Empirical')
    ax.set_xlabel("Nominal Coverage")
    ax.set_ylabel("Empirical Coverage")
    if title is not None:
        ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig, ax   # <-- return handles upstream unpacking & saving


def compare_coverage_curves(
    alphas1, emp_state1,
    alphas2, emp_state2,
    system_name="System",
    split="test",
    save_path=None
):
    """
    Compare two empirical coverage curves (SSID-GPK vs iGPK).

    Args:
        alphas1, emp_state1: tensors for first model (SSID-GPK)
        alphas2, emp_state2: tensors for second model (iGPK)
        system_name (str): name of the system for plot title
        split (str): 'train' or 'test'
        save_path (str, optional): if given, saves plot instead of showing
    """

    # Ensure torch tensors → numpy arrays
    alphas1 = alphas1.detach().cpu().numpy() if torch.is_tensor(alphas1) else alphas1
    emp_state1 = emp_state1.detach().cpu().numpy(
    ) if torch.is_tensor(emp_state1) else emp_state1
    alphas2 = alphas2.detach().cpu().numpy() if torch.is_tensor(alphas2) else alphas2
    emp_state2 = emp_state2.detach().cpu().numpy(
    ) if torch.is_tensor(emp_state2) else emp_state2

    # Plot
    plt.figure(figsize=(6, 3))
    plt.plot(alphas1 * 100, emp_state1
             * 100, 'o-', label="SSID-GPK", lw=2)
    plt.plot(alphas2 * 100, emp_state2
             * 100, 's--', label="iGPK", lw=2)
    plt.plot([50, 100], [50, 100], 'k--', alpha=0.7, label="Ideal")

    plt.xlabel("Nominal Coverage (%)", fontsize=11)
    plt.ylabel("Empirical Coverage (%)", fontsize=11)
    # plt.title(f"{system_name} ({split.capitalize()} Set)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.8)
    plt.legend(frameon=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_calibration_per_state(alphas, empirical_state, state_labels=None, title="Per-State Calibration"):
    a = alphas.detach().cpu().view(-1).numpy()
    es = empirical_state.detach().cpu().numpy()  # (A, n)
    fig, ax = plt.subplots()
    ax.plot(a, a, linestyle='--', label='Ideal')
    n = es.shape[1]
    for i in range(n):
        lab = state_labels[i] if state_labels else f"x{i+1}"
        ax.plot(a, es[:, i], marker='o', linestyle='-', label=lab)
    ax.set_xlabel("Nominal Coverage")
    ax.set_ylabel("Empirical Coverage")
    if title is not None:
        ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def _save_latex_table(results_dict, outdir, fname_stub: str):
    """
    Save LaTeX table with summary stats (min, median, max, mean, std) for each entry in results_dict.
    Assumes each value is a 1D-ish torch.Tensor / np.ndarray / list of NRMSE values in *decimal*,
    and converts to percent in the output (x100). Numbers formatted with 2 decimals.

    Parameters
    ----------
    results_dict : dict[str, array-like]
        e.g. {"Poly-eDMD": NRMSE_poly, "RBF-eDMD": NRMSE_rbf, "SSID-GPK": NRMSE_ssid, "iGPK": NRMSE}
    outdir : str
        Output directory
    fname_stub : str
        Filename stub, e.g. f"{tag}_latex_table" (will write .txt)
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{fname_stub}.txt")

    def _to_1d_percent_array(x):
        # torch -> numpy
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x)

        x = np.ravel(x).astype(np.float64)
        x = x[~np.isnan(x)]  # drop NaNs
        return 100.0 * x     # decimal -> percent

    def _stats_row(arr):
        if arr.size == 0:
            return (np.nan, np.nan, np.nan, np.nan, np.nan)
        return (np.min(arr), np.median(arr), np.max(arr), np.mean(arr), np.std(arr, ddof=0))

    def _fmt(v):
        return f"{v:.2f}"

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{|l|c|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Model} & \textbf{Min (\%)} & \textbf{Median (\%)} & \textbf{Max (\%)} & \textbf{Mean (\%)} & \textbf{Std (\%)} \\")
    lines.append(r"\hline\hline")

    for name, vals in results_dict.items():
        arr = _to_1d_percent_array(vals)
        mn, med, mx, mean, sd = _stats_row(arr)
        lines.append(
            rf"{name} & {_fmt(mn)} & {_fmt(med)} & {_fmt(mx)} & {_fmt(mean)} & {_fmt(sd)} \\"
        )
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\caption{NRMSE summary statistics (in percent).}")
    lines.append(r"\end{table}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return path


def run_models_for_noise(
    system_name: str, *,
    train_frac: float,
    test_frac: float,
    clip: int | None,
    noise_type: str,
    intensity: float,
    seed: int,
    outdir: str = "Figures",
    normalizeData: bool = True,
    # modeling knobs
    lifted_order: int = 10, poly_deg: int = 3,
    max_iter: int = 1000, epoch_iters: int = 50,
    learn_rate: float = 0.01, momentum: float = 0.8,
    batch_size: int = 15, stop_tol: float = 1e-4,
    kernel_hp_scale: list[float | None] = [None, 1.0, None], gp_noise:float=1e-6,
    opt_weights: list[float] = [1.0, 1.0, 1.0],
    routine: str = "standard",
    train_method: str = "Zero-Mean",
    device: str = "cuda:0",
):
    """
    Train a suite of Koopman models on noisy simulation data and produce
    diagnostics.  For the Scalar NL system a specialised phase-portrait
    visualisation is produced instead of the usual time-series overlays.  In
    addition to plots, the function returns summary metrics including
    computation times and aggregate NRMSE values for each model.  These
    statistics can be consumed by higher level scripts (e.g. sweep_noise.py)
    when building tables or reports.
    """
    # 1) Load + normalize
    SimData_raw, ts, num_traj, N, nTrain, nTest = gpk.load_SimData(
        system_name, train_frac, test_frac, clip=clip)

    if normalizeData:
        SimData_clean, _, _ = gpk.normalize_data(
            SimData_raw.to(dtype=torch.float32), nTest, nTrain, N)
    else:
        SimData_clean = SimData_raw.to(dtype=torch.float32)

    # 2) Noise
    SimData = gpk.add_noise(SimData_clean, noise_type=noise_type,
                            intensity=intensity, seed=seed)
    
    Dataset = {}
    nx = SimData.shape[1]
    N = SimData.shape[2] - 1
    Ns_gpo = 1 * nTrain
    
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

    print(f'========================================================')
    print(f'========================================================')
    print(
        f'Dataset: [{nTrain} Training + {nTest} Test Trajectories with {N} time-steps]')
    print(f'========================================================')

    # 3) iGPK
    
    results = get_iGPK(
        Data=Dataset,
        nTrain=nTrain, nTest=nTest,
        lifting_order=lifted_order,
        max_iter=max_iter,
        sgd_lr=learn_rate,sgd_m=momentum, stop_tol=stop_tol,
        opt_weights=opt_weights,
        routine=routine, train_method=train_method,
        hp_scale=kernel_hp_scale, device=device, gp_noise=gp_noise,
        traj_batch_size=batch_size, full_cost_eval_every=epoch_iters,
    )
    t_iGPK = results['history']['opt_time']

    # unpack iGPK
    A_igpk, C_igpk = results["A"], results["C"]
    ObsManager_iGPK = results["ObsManager"]
    XhatTrain, XcvhatTrain, TrainNRMSE = results["Train"][
        "Xhat"], results["Train"]["Xcv"], results["Train"]["NRMSE"]
    XhatTest,  XcvhatTest,  TestNRMSE = results["Test"][
        "Xhat"],  results["Test"]["Xcv"],  results["Test"]["NRMSE"]

    # 4) eDMDs
    _, _, XhatTrain_poly, XhatTest_poly, TrainNRMSE_poly, TestNRMSE_poly, t_poly = gpk.eDMD_poly(
        SimData, nTrain, nTest, poly_deg=poly_deg)

    _, _, XhatTrain_rbf, XhatTest_rbf, TrainNRMSE_rbf, TestNRMSE_rbf, t_rbf = gpk.eDMD_RBF_kmeans(
        SimData, nTrain, nTest, num_centers=lifted_order, width=kernel_hp_scale[1],
        rbf_type='gaussian', state_aug=True)

    # 5) SSID-GPK
    # delay = math.ceil(lifted_order / nx)
    delay=8
    results_ssid = gpk.get_ssidgpk(
        SimData=SimData,
        nTrain=nTrain, nTest=nTest,
        lifting_order=lifted_order,
        delay=delay)
    t_ssid = results_ssid["time"]

    # unpack SSID-GPK results
    # A_ssid, C_ssid = results_ssid["A"], results_ssid["C"]
    # ObsManager_ssid = results_ssid["ObsManager"]
    XhatTrain_ssid, XcvhatTrain_ssid, TrainNRMSE_ssid = results_ssid["Train"][
        "Xhat"], results_ssid["Train"]["Xcv"], results_ssid["Train"]["NRMSE"]
    XhatTest_ssid,  XcvhatTest_ssid,  TestNRMSE_ssid = results_ssid["Test"][
        "Xhat"],  results_ssid["Test"]["Xcv"],  results_ssid["Test"]["NRMSE"]

    # 6) indices + timebase (not used directly for Scalar NL plotting)
    idx_trainMIN = torch.argmin(TrainNRMSE.mean(dim=1))
    idx_testMIN = torch.argmin(TestNRMSE.mean(dim=1))
    idx_testMAX = torch.argmax(TestNRMSE.mean(dim=1))
    time_arr = torch.arange(0., ts * (SimData.shape[2] - 1), ts)

    # 7) pack models for overlay plot (not used here but kept for completeness)
    models = [
        {"name": "Poly-eDMD", "train": {"Xhat": XhatTrain_poly},
            "test": {"Xhat": XhatTest_poly},
            "ErrTrain": TrainNRMSE_poly, "ErrTest": TestNRMSE_poly},
        {"name": "RBF-eDMD",  "train": {"Xhat": XhatTrain_rbf},
            "test": {"Xhat": XhatTest_rbf},
            "ErrTrain": TrainNRMSE_rbf, "ErrTest": TestNRMSE_rbf},
        {"name": "SSID-GPK", "train": {"Xhat": XhatTrain_ssid, "Xcvhat": XcvhatTrain_ssid},
            "test": {"Xhat": XhatTest_ssid, "Xcvhat": XcvhatTest_ssid},
            "ErrTrain": TrainNRMSE_ssid, "ErrTest": TestNRMSE_ssid},
        {"name": "iGPK", "train": {"Xhat": XhatTrain, "Xcvhat": XcvhatTrain},
            "test": {"Xhat": XhatTest, "Xcvhat": XcvhatTest},
            "ErrTrain": TrainNRMSE, "ErrTest": TestNRMSE},
    ]

    # 8) produce Scalar NL phase-portrait instead of time-series overlays
    stamp = datetime.now().strftime("%Y%m%d")
    tag = f"{system_name.replace(' ', '_')}_noise-{noise_type}_int-{intensity:.3f}_seed-{seed}_{stamp}"
    torch.save(models, f'{outdir}/AllData_{tag}.pt')

    if system_name.lower().startswith("scalar"):
        # Generate 20 evenly spaced initial conditions in [-8, 8]
        x0_vals = torch.linspace(-8.0, 8.0, 25, dtype=C_igpk.dtype)
        # Compute true one-step evolution using the provided discrete-time simulator.
        # The sim_discrete function returns a tensor of shape (n, num_steps).  We
        # request two steps (initial and next) to extract the one-step map.
        x1_true = []
        for x0 in x0_vals:
            x0_tensor = x0.view(1)  # shape (1,)
            states = gpk.sim_discrete(
                gpk.df_scalarNL, x0_tensor, ts, num_steps=2)
            x1_true.append(states[0, 1])
        x1_true = torch.stack(x1_true)

        Zmean = torch.zeros((x0_vals.shape[0], C_igpk.shape[1], 3))
        Zcv = torch.zeros(
            (x0_vals.shape[0], C_igpk.shape[1], C_igpk.shape[1], 3))

        preds = torch.zeros((x0_vals.shape[0], C_igpk.shape[0], 3))
        preds_cv = torch.zeros(
            (x0_vals.shape[0], C_igpk.shape[0], C_igpk.shape[0], 3))

        for j in range(x0_vals.shape[0]):
            # lift all states
            for i in range(C_igpk.shape[1]):
                Zmean[j, i, 0] = ObsManager_iGPK.predict_mean(
                    i, x0_vals[j].view(1, 1))
                Zcv[j, i, i, 0] = torch.abs(ObsManager_iGPK.predict_covariance(
                    i, x0_vals[j].view(1, 1)))

            _, _, preds[j, :, :], preds_cv[j, :, :, :] = gpk.sim_LTI(
                Zmean[j, :, 0].view(C_igpk.shape[1], 1).cpu(), A_igpk.cpu(), C_igpk.cpu(), num_steps=3, ts=None, x0cv=Zcv[j, :, :, 0].cpu())

        preds = preds.squeeze(1).detach().cpu()
        preds_cv = torch.abs(preds_cv.squeeze().detach().cpu())
        print(f'All the sigma is {torch.sqrt(preds_cv)}')
        # First figure: compare all models against the true mapping
        fig1, ax1 = plt.subplots(figsize=(7, 6))
        ax1.plot(x0_vals.numpy(), x1_true.numpy(),
                 'k-o', label='Original', markersize=4)
        ax1.errorbar(x0_vals.numpy(),
                     preds[:, 1], yerr=torch.sqrt(preds_cv[:, 1]), fmt='o', capsize=5, label='iGPK')
        # ax1.set_title("Scalar NL: x_1 vs x_0 (All Models)")
        ax1.set_xlabel("$x_0$")
        ax1.set_ylabel("$x_1$")
        ax1.legend()
        ax1.grid(True)
        _save(fig1, outdir, f"{tag}_1step-errorbar_iGPK-only")
        plt.close(fig1)

        # Eigen (iGPK)
        fig_eig = gpk.plot_eigen(A_igpk)
        _save(fig_eig, outdir, f"{tag}_eig_igpk")

        # ==== NEW: NRMSE vs time-step (test set) for all models ====
        def _nrmse_vs_time(Xhat_model, SimData, nTrain, N):
            """
            Returns a 1D tensor of length N: % NRMSE_k averaged over test trajectories
            (and states, if n>1) at each time step k.
            """
            # Ground-truth slice for test trajectories (shape: nTest, n, N)
            # nTrain:nTrain + Xhat_model.shape[0]
            GT = SimData[nTrain:nTrain + Xhat_model.shape[0], :, :N]

            # RMSE over test trajectories at each time/state index
            err = Xhat_model[:, :, :N] - GT
            # (n, N) mean over test trajs
            mse_t = err.pow(2).mean(dim=0)
            rmse_t = torch.sqrt(mse_t)               # (n, N)

            # Normalization range at each time/state (over test trajs)
            max_t = GT.max(dim=0).values             # (n, N)
            min_t = GT.min(dim=0).values             # (n, N)
            rng_t = (max_t - min_t)
            rng_t = torch.where(rng_t == 0, torch.ones_like(rng_t), rng_t)

            nrmse_t = rmse_t / rng_t                 # (n, N)

            # Average across states (for Scalar NL: n=1 so this is a no-op)
            return 100 * nrmse_t.mean(dim=0).detach().cpu()  # (N,)

        # Compute NRMSE vs time for each model
        nrmse_vs_t = {
            "iGPK":      _nrmse_vs_time(XhatTest,       SimData, nTrain, N),
            "Poly-eDMD": _nrmse_vs_time(XhatTest_poly,  SimData, nTrain, N),
            "RBF-eDMD":  _nrmse_vs_time(XhatTest_rbf,   SimData, nTrain, N),
            "SSID-GPK":  _nrmse_vs_time(XhatTest_ssid,  SimData, nTrain, N),
        }

        # Plot: NRMSE vs time-step (optionally vs time in seconds if you prefer)
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        # time (seconds); use np.arange(N) for pure steps
        t_axis = np.arange(N)
        for name, curve in nrmse_vs_t.items():
            ax3.plot(t_axis, curve.numpy(), label=name)
        # change to "Time step k" if using np.arange(N)
        ax3.set_xlabel("Time Step ($k$)")
        ax3.set_ylabel("Percent NRMSE")
        # ax3.set_title("Scalar NL: Test NRMSE vs Time-step (All Models)")
        ax3.grid(True)
        ax3.legend()
        _save(fig3, outdir, f"{tag}_NRMSE_vs_time_test")
        plt.close(fig3)
        GT_test = SimData[nTrain:nTrain+nTest, :, :N-1]  # (nTest, n, N)

        # Per-trajectory NLPD statistics (mean ± std across trajectories)
        # nlpd_traj_train_igpk = _nlpd_per_traj(XhatTrain[:,:,:N-1],      XcvhatTrain[:,:,:,:N-1],      GT_train).detach().cpu()
        nlpd_traj_test_igpk = _nlpd_per_traj(
            XhatTest[:, :, :N-1],       XcvhatTest[:, :, :, :N-1],       GT_test).detach().cpu()
        # nlpd_traj_train_ssid = _nlpd_per_traj(XhatTrain_ssid[:,:,:N-1], XcvhatTrain_ssid[:,:,:,:N-1], GT_train).detach().cpu()
        nlpd_traj_test_ssid = _nlpd_per_traj(
            XhatTest_ssid[:, :, :N-1],  XcvhatTest_ssid[:, :, :, :N-1],  GT_test).detach().cpu()

        # Print summary
        def _ms(x):
            return float(x.mean()), float(x.std(unbiased=False))
        # m, s = _ms(nlpd_traj_train_igpk);  print(f"Train NLPD iGPK:     mean={m:.4f}, std={s:.4f}")
        # m, s = _ms(nlpd_traj_train_ssid);  print(f"Train NLPD SSID-GPK: mean={m:.4f}, std={s:.4f}")
        m, s = _ms(nlpd_traj_test_igpk)
        print(f"Test  NLPD iGPK:     mean={m:.4f}, std={s:.4f}")
        m, s = _ms(nlpd_traj_test_ssid)
        print(f"Test  NLPD SSID-GPK: mean={m:.4f}, std={s:.4f}")

    else:
        # For all other systems we defer to the time-series visualisations in
        # GPKoopman.  We reuse the same plotting logic as the ACC26 script.
        # Note: these calls may be unused in the Scalar NL workflow but are
        # included for completeness if this helper is reused on other systems.
        y_labels = None  # ['P(t)', 'Q(t)']
        for (which, idx, split, sim_offset, suffix) in [
            ("best-train", idx_trainMIN, "train", nTest,         "Best_Train"),
            ("best-test",  idx_testMIN,  "test",  0,    "Best_Test"),
            ("worst-test", idx_testMAX,  "test",  0,    "Worst_Test"),
        ]:
            fig, _ = gpk.compare_model_predictions(
                time=time_arr, models=models, SimData=SimData, idx=idx, N=(
                    SimData.shape[2]-1),
                system_name=system_name, title_suffix=suffix, split=split, sim_offset=sim_offset,
                compare_to="SimData_clean", SimData_clean=SimData_clean, sigma=1.0, skip_title=True,
                y_labels=y_labels)
            _save(fig, outdir, f"{tag}_timeseries_{which}")

            models_nocv = [
                {"name": "iGPK", "train": {"Xhat": XhatTrain},
                    "test": {"Xhat": XhatTest}},
                {"name": "Poly-eDMD", "train": {"Xhat": XhatTrain_poly},
                    "test": {"Xhat": XhatTest_poly}},
                {"name": "RBF-eDMD",  "train": {"Xhat": XhatTrain_rbf},
                    "test": {"Xhat": XhatTest_rbf}},
                {"name": "SSID-GPK", "train": {"Xhat": XhatTrain_ssid},
                    "test": {"Xhat": XhatTest_ssid}},
            ]
            fig, _ = gpk.compare_model_predictions(
                time=time_arr, models=models_nocv, SimData=SimData, idx=idx, N=(
                    SimData.shape[2]-1),
                system_name=system_name, title_suffix=suffix, split=split, sim_offset=sim_offset,
                compare_to="SimData_clean", SimData_clean=SimData_clean, sigma=1.0, skip_title=True,
                y_labels=y_labels)
            _save(fig, outdir, f"{tag}_timeseries_NoCV_{which}")


        train_int_coverage = compute_interval_coverage(
            XhatTrain, XcvhatTrain, SimData, sim_offset=nTest)
        test_int_coverage = compute_interval_coverage(
            XhatTest,  XcvhatTest,  SimData, sim_offset=0)

        print("Train coverage (per state):")
        for alpha, vals in train_int_coverage.items():
            print(f"  {int(alpha*100)}% interval: {vals.numpy()*100}")
        print("\nTest coverage (per state):")
        for alpha, vals in test_int_coverage.items():
            print(f"  {int(alpha*100)}% interval: {vals.numpy()*100}")
        # Choose a coverage grid (50%..99%)
        alphas = torch.linspace(0.50, 0.99, steps=50)

        # iGPK
        a_tr_i, emp_tr_i = coverage_curve(
            XhatTrain, XcvhatTrain, SimData, sim_offset=nTest,      alphas=alphas, reduce="mean")
        a_te_i, emp_te_i = coverage_curve(
            XhatTest,  XcvhatTest,  SimData, sim_offset=0, alphas=alphas, reduce="mean")
        print(
            f"iGPK miscalibration area : TRAIN: {miscalibration_area(a_tr_i, emp_tr_i).item():.2e} || TEST: {miscalibration_area(a_te_i, emp_te_i).item():.2e}")

        # SSID-GPK
        a_tr_s, emp_tr_s = coverage_curve(
            XhatTrain_ssid, XcvhatTrain_ssid, SimData, sim_offset=nTest,      alphas=alphas, reduce="mean")
        a_te_s, emp_te_s = coverage_curve(
            XhatTest_ssid,  XcvhatTest_ssid,  SimData, sim_offset=0, alphas=alphas, reduce="mean")
        print(
            f"SSID-GPK miscalibration area : TRAIN: {miscalibration_area(a_tr_s, emp_tr_s).item():.2e} || TEST: {miscalibration_area(a_te_s, emp_te_s).item():.2e}")

        save_path = os.path.join(outdir, f"{tag}_CalibCurve_Compare.png")
        compare_coverage_curves(a_te_s, emp_te_s, a_te_i, emp_te_i,
                                system_name="Lorenz", split="test", save_path=save_path)

        # c) NRMSE comparison plot
        fig_nrmse = gpk.plot_NRMSE_metrics(
            [TrainNRMSE, TrainNRMSE_poly, TrainNRMSE_rbf,
                TrainNRMSE_ssid],
            [TestNRMSE,  TestNRMSE_poly,  TestNRMSE_rbf,
                TestNRMSE_ssid],
            ["iGPK", "Poly-eDMD", "RBF-eDMD", "SSID-GPK"]
        )
        _save(fig_nrmse, outdir, f"{tag}_NRMSE_compare")

        # d) Eigen (iGPK)
        fig_eig = gpk.plot_eigen(A_igpk)
        _save(fig_eig, outdir, f"{tag}_eig_igpk")

        # ==== NEW: NLPD for iGPK & SSID-GPK (Train/Test) + plots & summary ====
        # Ground-truth slices
        # GT_train = SimData[0:nTrain, :, :N-1]         # (nTrain, n, N)
        GT_test = SimData[:nTest, :, :N-1]  # (nTest, n, N)

        # Per-trajectory NLPD statistics (mean ± std across trajectories)
        # nlpd_traj_train_igpk = _nlpd_per_traj(XhatTrain[:,:,:N-1],      XcvhatTrain[:,:,:,:N-1],      GT_train).detach().cpu()
        nlpd_traj_test_igpk = gpk.nlpd_per_traj(
            XhatTest[:, :, :N-1],       XcvhatTest[:, :, :, :N-1],       GT_test).detach().cpu()
        # nlpd_traj_train_ssid = _nlpd_per_traj(XhatTrain_ssid[:,:,:N-1], XcvhatTrain_ssid[:,:,:,:N-1], GT_train).detach().cpu()
        nlpd_traj_test_ssid = gpk.nlpd_per_traj(
            XhatTest_ssid[:, :, :N-1],  XcvhatTest_ssid[:, :, :, :N-1],  GT_test).detach().cpu()

        # Print summary
        def _ms(x):
            return float(x.mean()), float(x.std(unbiased=False))
        # m, s = _ms(nlpd_traj_train_igpk);  print(f"Train NLPD iGPK:     mean={m:.4f}, std={s:.4f}")
        # m, s = _ms(nlpd_traj_train_ssid);  print(f"Train NLPD SSID-GPK: mean={m:.4f}, std={s:.4f}")
        m, s = _ms(nlpd_traj_test_igpk)
        print(f"Test  NLPD iGPK:     mean={m:.4f}, std={s:.4f}")
        m, s = _ms(nlpd_traj_test_ssid)
        print(f"Test  NLPD SSID-GPK: mean={m:.4f}, std={s:.4f}")

    # Compute aggregate NRMSE metrics for reporting.  Use the mean over all
    # dimensions and trajectories for each model.  These values are stored in
    # the return dictionary under the "NRMSE" key.
    train_nrmse = {
        "Poly-eDMD": TrainNRMSE_poly,
        "RBF-eDMD": TrainNRMSE_rbf,
        "SSID-GPK": TrainNRMSE_ssid,
        "iGPK": TrainNRMSE,
    }
    test_nrmse = {
        "Poly-eDMD": TestNRMSE_poly,
        "RBF-eDMD": TestNRMSE_rbf,
        "SSID-GPK": TestNRMSE_ssid,
        "iGPK": TestNRMSE,
    }

    _save_latex_table(train_nrmse, outdir, f"{tag}_latex_table_train")
    _save_latex_table(test_nrmse, outdir, f"{tag}_latex_table_test")

    print(f'========================================================')
    print(
        f'Train NRMSE Metrics for {noise_type} Noise with Intensity = {intensity*100}%')
    print(f'========================================================')
    print(
        f'Train NRMSE iGPK      = {TrainNRMSE.mean()*100:.2f} \u00B1 {(TrainNRMSE*100).std():.2f} %')
    print(
        f'Train NRMSE Poly-eDMD = {TrainNRMSE_poly.mean()*100:.2f} \u00B1 {(TrainNRMSE_poly*100).std():.2f} %')
    print(
        f'Train NRMSE RBF-eDMD  = {TrainNRMSE_rbf.mean()*100:.2f} \u00B1 {(TrainNRMSE_rbf*100).std():.2f} %')
    print(
        f'Train NRMSE SSID-GPK  = {TrainNRMSE_ssid.mean()*100:.2f} \u00B1 {(TrainNRMSE_ssid*100).std():.2f} %')
    print(f'========================================================')
    print(
        f'Test NRMSE Metrics for {noise_type} Noise with Intensity = {intensity*100}%')
    print(f'========================================================')
    print(
        f'Test NRMSE iGPK      = {TestNRMSE.mean()*100:.2f} \u00B1 {(TestNRMSE*100).std():.2f} %')
    print(
        f'Test NRMSE Poly-eDMD = {TestNRMSE_poly.mean()*100:.2f} \u00B1 {(TestNRMSE_poly*100).std():.2f} %')
    print(
        f'Test NRMSE RBF-eDMD  = {TestNRMSE_rbf.mean()*100:.2f} \u00B1 {(TestNRMSE_rbf*100).std():.2f} %')
    print(
        f'Test NRMSE SSID-GPK  = {TestNRMSE_ssid.mean()*100:.2f} \u00B1 {(TestNRMSE_ssid*100).std():.2f} %')
    print(f'========================================================')
    print(f'Computation Time iGPK       = {t_iGPK:.2f} seconds')
    print(f'Computation Time Poly-eDMD  = {t_poly:.2f} seconds')
    print(f'Computation Time RBF-eDMD   = {t_rbf:.2f} seconds')
    print(f'Computation Time SSID-GPK   = {t_ssid:.2f} seconds')
    # print(f'Computation Time R3Koopman  = {t_r3k:.2f} seconds')
    print(f'========================================================')
    print(f'========================================================')

    plt.close('all')
    print(f'All figures saved.')

    # Return bundle
    return {
        "timings": {"iGPK": t_iGPK, "Poly-eDMD": t_poly, "RBF-eDMD": t_rbf, "SSID-GPK": t_ssid},
        "tag": tag,
        "outdir": outdir,
        "Train-NRMSE": train_nrmse,
        "Test-NRMSE": test_nrmse,
    }


if __name__ == "__main__":
    # Example usage: run the sweep for the Scalar NL system with no noise.  To
    # create additional sweeps simply call run_models_for_noise with different
    # noise specifications.
    run_models_for_noise(
        system_name="Scalar NL",
        train_frac=0.3, test_frac=0.2, clip=150,
        noise_type="gaussian", intensity=0.0, seed=100,
        outdir="Figures_ScalarNL"
    )


# ===== END ===== #
