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


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _save(fig, outdir, fname_stub: str):
    _ensure_dir(outdir)
    path = os.path.join(outdir, f"{fname_stub}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"saved: {path}")


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

    if noise_type == None:
        noise = 0
    elif noise_type == "gaussian":
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
    elif noise_type == "linear_uniform":
        # Uniform noise with linearly varying intensity
        var_intensity = intensity * SimData_norm
        noise = (torch.rand_like(SimData_norm) * 2 - 1) * var_intensity
    else:
        raise ValueError(
            f"Unsupported noise_type {noise_type}. Choose 'gaussian', 'uniform', 'linear_gaussian', 'quadratic_gaussian', or 'linear_uniform'.")

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
    return fig


def plot_NRMSE_metrics(TrainNRMSE_list, TestNRMSE_list, model_names):
    """
    Compare Train/Test NRMSE across multiple models.

    Args:
        TrainNRMSE_list (list of torch.Tensor): Each tensor has shape (nTraj, nStates).
        TestNRMSE_list  (list of torch.Tensor): Each tensor has shape (nTraj, nStates).
        model_names     (list of str): Names of the models, used as plot labels.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    # --- Training set (per trajectory curves) ---
    for train_rmse, name in zip(TrainNRMSE_list, model_names):
        mean_nrmse = train_rmse.mean(dim=1)   # average across states
        axes[0].plot(
            range(train_rmse.shape[0]),
            mean_nrmse.numpy(),
            marker='o', linestyle='-', label=name
        )
    axes[0].set_title('Training Metrics (Per Trajectory)')
    axes[0].set_xlabel("Trajectory Index")
    axes[0].set_ylabel("Mean NRMSE")
    axes[0].legend()
    axes[0].grid()

    # --- Test set (per trajectory curves) ---
    for test_rmse, name in zip(TestNRMSE_list, model_names):
        mean_nrmse = test_rmse.mean(dim=1)   # average across states
        axes[1].plot(
            range(test_rmse.shape[0]),
            mean_nrmse.numpy(),
            marker='o', linestyle='-', label=name
        )
    axes[1].set_title('Test Metrics (Per Trajectory)')
    axes[1].set_xlabel("Trajectory Index")
    axes[1].set_ylabel("Mean NRMSE")
    axes[1].legend()
    axes[1].grid()

    # --- Training set (bar chart: overall average) ---
    overall_train = [rmse.mean().item() for rmse in TrainNRMSE_list]
    axes[2].bar(np.arange(len(model_names)),
                overall_train, tick_label=model_names)
    axes[2].set_title("Training Metrics (Overall Mean)")
    axes[2].set_ylabel("Mean NRMSE")
    axes[2].grid(axis="y")

    # --- Test set (bar chart: overall average) ---
    overall_test = [rmse.mean().item() for rmse in TestNRMSE_list]
    axes[3].bar(np.arange(len(model_names)),
                overall_test, tick_label=model_names)
    axes[3].set_title("Test Metrics (Overall Mean)")
    axes[3].set_ylabel("Mean NRMSE")
    axes[3].grid(axis="y")

    plt.tight_layout()
    return fig


def compare_model_predictions(
    time,
    models,
    SimData,
    idx,
    N,
    system_name,
    title_suffix="",
    *,
    split="train",                    # "train" or "test"
    sim_offset=0,                     # e.g., nTrain for test set
    compare_to="SimData",             # "SimData" or "SimData_clean"
    SimData_clean=None,               # required if compare_to="SimData_clean"
    sigma=1.0,                        # number of std-devs for bands (iGPK)
    colors=None                       # optional color map per model
):
    """
    Compare time-series predictions from multiple models to ground truth, with optional uncertainty bands.

    Parameters
    ----------
    time : 1D tensor/array of length N
    models : list of dicts, one per model. Each dict should contain:
        {
          "name": "iGPK",
          # For the chosen 'split' (train/test), provide tensors shaped as below.
          # Xhat_* : (nTraj, nStates, N)
          # Xcvhat_* : (nTraj, nStates, nStates, N)  (optional; if absent → deterministic line)
          "train": {"Xhat": XhatTrain, "Xcvhat": XcvhatTrain (optional)},
          "test" : {"Xhat": XhatTest,  "Xcvhat": XcvhatTest  (optional)}
        }
    SimData : tensor of shape (numTraj, nStates, N_total)
        Ground truth (noisy or clean, depending on your dataset). For test plots,
        use sim_offset=nTrain to index into test trajectories.
    idx : int
        Which trajectory index to plot (within the chosen split).
    N : int
        Number of time steps to plot.
    system_name : str
    title_suffix : str
    split : {"train","test"}
    sim_offset : int
        Added to SimData's first index to align with chosen split.
    compare_to : {"SimData","SimData_clean"}
        Choose which ground truth to overlay. If "SimData_clean", pass SimData_clean.
    SimData_clean : tensor like SimData
    sigma : float
        Width of the uncertainty band in standard deviations for stochastic models (e.g., iGPK).
    colors : dict or list
        Optional mapping from model name → color string, or a list of colors matching `models` order.

    Notes
    -----
    - Any model entry without 'Xcvhat' is treated as deterministic (no bands).
    - Works for any state dimension; creates one subplot per state.
    - If `nStates == 1`, returns a single-axis figure for convenience.
    """

    assert split in ("train", "test"), "split must be 'train' or 'test'"

    # Pick ground truth
    if compare_to == "SimData":
        GT = SimData
    elif compare_to == "SimData_clean":
        if SimData_clean is None:
            raise ValueError(
                "compare_to='SimData_clean' requires SimData_clean")
        GT = SimData_clean
    else:
        raise ValueError("compare_to must be 'SimData' or 'SimData_clean'")

    # Infer state dimension from the first model
    if len(models) == 0:
        raise ValueError("`models` cannot be empty.")
    Xhat0 = models[0][split]["Xhat"]
    n_states = Xhat0.shape[1]

    # Colors
    default_cycle = plt.rcParams['axes.prop_cycle'].by_key().get(
        'color', ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])
    if isinstance(colors, dict):
        def color_for(name, k): return colors.get(
            name, default_cycle[k % len(default_cycle)])
    elif isinstance(colors, (list, tuple)):
        def color_for(name, k): return colors[k % len(colors)]
    else:
        def color_for(name, k): return default_cycle[k % len(default_cycle)]

    # Setup figure/axes
    fig_height = max(5, 1.8 * n_states)
    fig, axes = plt.subplots(
        n_states, 1, figsize=(6, fig_height), sharex=True)
    if n_states == 1:
        axes = [axes]

    # Title
    gt_label = "NL (truth: noisy)" if compare_to == "SimData" else "NL (truth: clean)"
    fig.suptitle(f"{system_name}: {title_suffix} [{split.capitalize()}]")

    # Plot per state
    for s in range(n_states):
        ax = axes[s]

        # Ground truth
        gt = GT[sim_offset + idx, s, :N].cpu().numpy()
        # choose about 20 evenly spaced marker points
        n_markers = 20
        marker_idx = np.linspace(0, N - 1, n_markers, dtype=int)

        ax.plot(time, gt, linestyle="--", linewidth=1.3, color="black", alpha=0.75,
            label=gt_label)

        # overlay sparse markers for clarity
        ax.plot(time[marker_idx], gt[marker_idx], marker='o', linestyle='None',
            color="black", markersize=4, alpha=0.8, label=None)

        # Overlay all models
        for k, model in enumerate(models):
            name = model.get("name", f"Model {k+1}")
            pack = model[split]
            Xhat = pack["Xhat"][idx, s, :N]

            col = color_for(name, k)
            ax.plot(time, Xhat.cpu().numpy(),
                    label=name, linewidth=1.6, color=col)

            # Optional uncertainty band if available
            Xcvhat = pack.get("Xcvhat", None)
            if Xcvhat is not None:
                # take diag element (state s) variance over time
                var_s = Xcvhat[idx, s, s, :N]
                # clamp small negatives due to numerical issues
                std_s = torch.sqrt(torch.clamp(var_s, min=0.0))
                lower = (Xhat - sigma * std_s).cpu().numpy()
                upper = (Xhat + sigma * std_s).cpu().numpy()
                ax.fill_between(time, lower, upper, alpha=0.16, color=col)

        ax.set_ylabel(f"X{s+1}")
        ax.grid(True, linestyle=":", linewidth=0.7)

    axes[-1].set_xlabel("Time [s]")

    # One shared legend
    # Build a clean legend across axes: collect handles/labels from the last axis
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),   # just above the plots
        ncol=len(labels),             # all entries in one horizontal row
        frameon=False
    )
    # Deduplicate while preserving order
    # seen = set()
    # uniq = [(h, l) for h, l in zip(handles, labels)
    #         if not (l in seen or seen.add(l))]
    # if uniq:
    #     fig.legend(*zip(*uniq), loc="upper right", bbox_to_anchor=(0.98, 0.98))

    fig.tight_layout(rect=[0, 0, 1., 0.95])
    # plt.show()
    return fig, axes


def run_models_for_noise(
    system_name: str,
    train_frac: float,
    test_frac: float,
    clip: int | None,
    noise_type: str,
    intensity: float,
    seed: int,
    # modeling knobs
    lifted_order: int = 10,
    iters_list=(250, 50, 50, 100),
    learn_rate: float = 0.04,
    opt_weights=(10.0, 1.0, 10.0),
    routine: str = "Z_only",
    train_method: str = "Horizon",
    device: str = "cuda:0",
    # saving
    outdir: str = "Figures"
):
    # 1) Load + normalize
    SimData_raw, ts, num_traj, N, nTrain, nTest = load_SimData(
        # :contentReference[oaicite:9]{index=9}
        system_name, train_frac, test_frac, clip=clip)
    # SimData_clean, mu_vec, std_vec = normalize_data(
    #     SimData_raw, nTrain, N)  # :contentReference[oaicite:10]{index=10}
    SimData_clean = SimData_raw
    # 2) Noise
    SimData = add_noise(SimData_clean, noise_type=noise_type,
                        intensity=intensity, seed=seed)

    # 3) iGPK
    t0 = time.perf_counter()
    results = get_iGPK(
        SimData=SimData,
        nTrain=nTrain, nTest=nTest,
        lifting_order=lifted_order,
        iters_list=list(iters_list),
        learn_rate=learn_rate,
        opt_weights=list(opt_weights),
        routine=routine,
        train_method=train_method,
        device=device
    )
    t_iGPK = time.perf_counter() - t0

    # unpack iGPK
    A_igpk, C_igpk = results["A"], results["C"]
    # ICsetTrain, ICsetTest = results["ICsetTrain"], results["ICsetTest"]
    XhatTrain, XcvhatTrain, TrainNRMSE = results["Train"][
        "Xhat"], results["Train"]["Xcv"], results["Train"]["NRMSE"]
    XhatTest,  XcvhatTest,  TestNRMSE = results["Test"][
        "Xhat"],  results["Test"]["Xcv"],  results["Test"]["NRMSE"]

    # 4) eDMDs
    t0 = time.perf_counter()
    A_poly, C_poly, XhatTrain_poly, XhatTest_poly, TrainNRMSE_poly, TestNRMSE_poly = gpk.eDMD_poly(
        SimData, nTrain, nTest, poly_deg=3)
    t_poly = time.perf_counter() - t0

    t0 = time.perf_counter()
    A_rbf, C_rbf, XhatTrain_rbf, XhatTest_rbf, TrainNRMSE_rbf, TestNRMSE_rbf = gpk.eDMD_RBF_kmeans(
        SimData, nTrain, nTest, num_centers=lifted_order, width=0.2, rbf_type='thin_plate', state_aug=True)
    t_rbf = time.perf_counter() - t0

    # 5) SSID-GPK
    t0 = time.perf_counter()
    results_ssid = gpk.get_ssidgpk(
        SimData=SimData,
        nTrain=nTrain, nTest=nTest,
        lifting_order=lifted_order,
        delay=N - 1)
    t_ssid = time.perf_counter() - t0

    # unpack SSID-GPK results
    A_ssid, C_ssid = results_ssid["A"], results_ssid["C"]
    # ICsetTrain_ssid, ICsetTest_ssid = results["ICsetTrain"], results["ICsetTest"]
    XhatTrain_ssid, XcvhatTrain_ssid, TrainNRMSE_ssid = results_ssid["Train"][
        "Xhat"], results_ssid["Train"]["Xcv"], results_ssid["Train"]["NRMSE"]
    XhatTest_ssid,  XcvhatTest_ssid,  TestNRMSE_ssid = results_ssid["Test"][
        "Xhat"],  results_ssid["Test"]["Xcv"],  results_ssid["Test"]["NRMSE"]

    # 6) indices + timebase
    idx_trainMIN = torch.argmin(TrainNRMSE.mean(dim=1))
    idx_testMIN = torch.argmin(TestNRMSE.mean(dim=1))
    idx_testMAX = torch.argmax(TestNRMSE.mean(dim=1))
    # same shape you used (see your callsite) :contentReference[oaicite:11]{index=11}
    time_arr = torch.arange(0., ts * (SimData.shape[2] - 1), ts)

    # 7) pack models for overlay plot
    models = [
        {"name": "iGPK", "train": {"Xhat": XhatTrain, "Xcvhat": XcvhatTrain},
            "test": {"Xhat": XhatTest, "Xcvhat": XcvhatTest}},
        {"name": "Poly-eDMD", "train": {"Xhat": XhatTrain_poly},
            "test": {"Xhat": XhatTest_poly}},
        {"name": "RBF-eDMD",  "train": {"Xhat": XhatTrain_rbf},
            "test": {"Xhat": XhatTest_rbf}},
        {"name": "SSID-GPK", "train": {"Xhat": XhatTrain_ssid, "Xcvhat": XcvhatTrain_ssid},
            "test": {"Xhat": XhatTest_ssid, "Xcvhat": XcvhatTest_ssid}}
    ]

    models_nocv = [
        {"name": "iGPK", "train": {"Xhat": XhatTrain},
            "test": {"Xhat": XhatTest}},
        {"name": "Poly-eDMD", "train": {"Xhat": XhatTrain_poly},
            "test": {"Xhat": XhatTest_poly}},
        {"name": "RBF-eDMD",  "train": {"Xhat": XhatTrain_rbf},
            "test": {"Xhat": XhatTest_rbf}},
        {"name": "SSID-GPK", "train": {"Xhat": XhatTrain_ssid},
            "test": {"Xhat": XhatTest_ssid}}
    ]

    # 8) make & save all figures
    stamp = datetime.now().strftime("%Y%m%d")
    tag = f"{system_name.replace(' ', '_')}_noise-{noise_type}_int-{intensity:.3f}_seed-{seed}_{stamp}"

    # a) 3 trajectory overlays
    for (which, idx, split, sim_offset, suffix) in [
        ("best-train", idx_trainMIN, "train", 0,         "Best_Train"),
        ("best-test",  idx_testMIN,  "test",  nTrain,    "Best_Test"),
        ("worst-test", idx_testMAX,  "test",  nTrain,    "Worst_Test"),
    ]:
        fig, _ = compare_model_predictions(
            time=time_arr, models=models, SimData=SimData, idx=idx, N=(
                SimData.shape[2]-1),
            system_name=system_name, title_suffix=suffix, split=split, sim_offset=sim_offset,
            compare_to="SimData_clean", SimData_clean=SimData_clean, sigma=1.0
        )
        _save(fig, outdir, f"{tag}_timeseries_{which}")

        fig, _ = compare_model_predictions(
            time=time_arr, models=models_nocv, SimData=SimData, idx=idx, N=(
                SimData.shape[2]-1),
            system_name=system_name, title_suffix=suffix, split=split, sim_offset=sim_offset,
            compare_to="SimData_clean", SimData_clean=SimData_clean, sigma=1.0
        )
        _save(fig, outdir, f"{tag}_timeseries_NoCV_{which}")

    # b) Eigen (iGPK)
    fig_eig = plot_eigen(A_igpk)
    _save(fig_eig, outdir, f"{tag}_eig_igpk")

    # c) NRMSE comparison
    fig_nrmse = plot_NRMSE_metrics(
        [TrainNRMSE, TrainNRMSE_poly, TrainNRMSE_rbf, TrainNRMSE_ssid],
        [TestNRMSE,  TestNRMSE_poly,  TestNRMSE_rbf, TestNRMSE_ssid],
        ["iGPK", "Poly-eDMD", "RBF-eDMD", "SSID-GPK"]
    )
    _save(fig_nrmse, outdir, f"{tag}_NRMSE_compare")

    # 9) small return bundle (optional)
    return {
        "timings": {"iGPK": t_iGPK, "Poly-eDMD": t_poly, "RBF-eDMD": t_rbf, "SSID-GPK": t_ssid},
        "orders":  {"iGPK": C_igpk.shape[1], "Poly-eDMD": C_poly.shape[1], "RBF-eDMD": C_rbf.shape[1], "SSID-GPK": C_ssid.shape[1]},
        "splits":  {"nTrain": nTrain, "nTest": nTest},
        "tag": tag,
        "outdir": outdir
    }


if __name__ == "__main__":
    # Optional: a single manual call for quick test
    run_models_for_noise(
        system_name="Simple Pendulum",
        train_frac=0.3, test_frac=0.2, clip=150,
        noise_type="linear_gaussian", intensity=0.0, seed=100,
        outdir="Figures_Trial"
    )


### ---- ARCHIVE ---- ###
"""
system_name = 'Simple Pendulum'
SimData_raw, ts, num_trajectories, N, nTrain, nTest = load_SimData(
    system_name, 0.4, 0.2, clip=100)
# Normalization
SimData_clean, mu_vec, std_vec = normalize_data(SimData_raw, nTrain, N)
# Add Noise
SimData = add_noise(SimData_clean, noise_type="linear_gaussian",
                    intensity=0.1, seed=100)

iters_list = [250, 50, 50, 100]
opt_weights = [10., 1., 10.]
lifted_order = 10

# COMPUTE iGPK MODEL
t_start = time.perf_counter()
results = get_iGPK(
    SimData=SimData,
    nTrain=nTrain, nTest=nTest,
    lifting_order=lifted_order,
    iters_list=iters_list,
    learn_rate=0.04,
    opt_weights=opt_weights,
    routine="Z_only",          # or "SpacedOpt"
    train_method="Horizon",    # or "K-Means"
    device="cuda:0"
)
t_igpk = time.perf_counter() - t_start

# COMPUTE EDMD-POLY MODEL
t_start = time.perf_counter()
A_edmd, C_edmd, XhatTrain_edPoly, XhatTest_edPoly, TrainNRMSE_edPoly, TestNRMSE_edPoly = gpk.eDMD_poly(
    SimData, nTrain, nTest, poly_deg=3)
t_eDMD_poly = time.perf_counter() - t_start
lifted_order_edPoly = C_edmd.shape[1]

# COMPUTE EDMD-RBF MODEL
t_start = time.perf_counter()
A_edmdrbf, C_edmdrbf, XhatTrain_edRBF, XhatTest_edRBF, TrainNRMSE_edRBF, TestNRMSE_edRBF = gpk.eDMD_RBF_kmeans(
    SimData, nTrain, nTest, num_centers=10, width=0.2, rbf_type='thin_plate', state_aug=True)
t_eDMD_rbf = time.perf_counter() - t_start
lifted_order_edrbf = C_edmdrbf.shape[1]

# COMPUTE SSID-GP-KOOPMAN (to be implemented)

## --- UNPACK AND PLOT --- ##
# Unpack what you need
A_igpk, C_igpk = results["A"], results["C"]
ICsetTrain, ICsetTest = results["ICsetTrain"], results["ICsetTest"]
XhatTrain, XcvhatTrain, TrainNRMSE = results["Train"][
    "Xhat"], results["Train"]["Xcv"], results["Train"]["NRMSE"]
XhatTest,  XcvhatTest,  TestNRMSE = results["Test"]["Xhat"],  results["Test"]["Xcv"],  results["Test"]["NRMSE"]

# Example: quick indices & plots (optional)
idx_trainMIN = torch.argmin(TrainNRMSE.mean(dim=1))
idx_testMIN = torch.argmin(TestNRMSE.mean(dim=1))
idx_testMAX = torch.argmax(TestNRMSE.mean(dim=1))
time_array = torch.arange(0., ts * (SimData.shape[2] - 1), ts)

# gpk.plot_time_series_with_bounds(time_array, XhatTest,  XcvhatTest,  SimData, idx_testMIN,
#                                  SimData.shape[2]-1, system_name, title_suffix='Best Test Trajectory', sim_offset=nTrain)

models = [
    {"name": "iGPK",
     "train": {"Xhat": XhatTrain, "Xcvhat": XcvhatTrain},
     "test":  {"Xhat": XhatTest,  "Xcvhat": XcvhatTest}},
    {"name": "Poly-eDMD",
     "train": {"Xhat": XhatTrain_edPoly},   # deterministic
     "test":  {"Xhat": XhatTest_edPoly}},
    {"name": "RBF-eDMD",
     "train": {"Xhat": XhatTrain_edRBF},   # deterministic
     "test":  {"Xhat": XhatTest_edRBF}}
]

# Predicted Trajectory Comparison
# Best Training Trajectory
compare_model_predictions(
    time=time_array,
    models=models,
    SimData=SimData,
    idx=idx_trainMIN,
    N=N,
    system_name=system_name,
    title_suffix="Best Train Trajectory",
    split="train",
    sim_offset=0,                 # for train
    compare_to="SimData_clean",         # or "SimData_clean"
    SimData_clean=SimData_clean,  # if using clean
    sigma=1.0                     # 1σ band for iGPK
)
# Best Test Trajectory
compare_model_predictions(
    time=time_array,
    models=models,
    SimData=SimData,
    idx=idx_testMIN,
    N=N,
    system_name=system_name,
    title_suffix="Best Test Trajectory",
    split="test",
    sim_offset=nTrain,                 # for train
    compare_to="SimData_clean",         # or "SimData_clean"
    SimData_clean=SimData_clean,  # if using clean
    sigma=1.0                     # 1σ band for iGPK
)
# Worst Test Trajectory
compare_model_predictions(
    time=time_array,
    models=models,
    SimData=SimData,
    idx=idx_testMAX,
    N=N,
    system_name=system_name,
    title_suffix="Worst Test Trajectory",
    split="test",
    sim_offset=nTrain,                 # for train
    compare_to="SimData_clean",         # or "SimData_clean"
    SimData_clean=SimData_clean,  # if using clean
    sigma=1.0                     # 1σ band for iGPK
)

# Other Plots
plot_eigen(A_igpk)
plot_NRMSE_metrics([TrainNRMSE, TrainNRMSE_edPoly, TrainNRMSE_edRBF], [
                   TestNRMSE, TestNRMSE_edPoly, TestNRMSE_edRBF], ['iGPK', 'Poly-eDMD', 'RBF-eDMD'])

# Computation Time
print(f'{lifted_order:d}-D GP-K Model identified in {t_igpk:.2f} seconds.')
print(f'{lifted_order_edPoly:d}-D Poly-eDMD Model identified in {t_eDMD_poly:.2f} seconds.')
print(f'{lifted_order_edrbf:d}-D RBF-eDMD Model (with K-Means centers) identified in {t_eDMD_rbf:.2f} seconds.')

plt.show()
"""

### --- MODEL SAVING --- ###


# ===== END ===== #
