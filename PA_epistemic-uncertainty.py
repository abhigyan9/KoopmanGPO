"""
Perturbation Analysis to study Epistemic Uncertainty
Vary the injected noise and training dataset size
    -> Track the learned noise and performance
"""
from get_iGPK_new import get_iGPK
import matplotlib.pyplot as plt
import os
import csv
import itertools
from typing import Dict, Any, List, Tuple, Optional
import time

import torch
import GPKoopman as gpk

# HELPERS


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


def _extract_gp_noise_stats(obs_manager) -> Tuple[List[float], float, float, float, float, float]:
    """Extract learned GP noise hyperparameters from an ObsManager (GPObservablesManager).

    Returns:
        noise_vals: list of learned noise scalars (flattened across observables)
        mean, median, std, min, max: aggregate stats (NaN if empty)
    """
    noise_vals: List[float] = []
    if obs_manager is None:
        return noise_vals, float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    # Prefer the manager API
    params_all = obs_manager.get_all_params()
    for _, pd in params_all.items():
        nv = pd.get("noise")
        if isinstance(nv, torch.Tensor):
            nv = nv.detach().cpu().reshape(-1)
            noise_vals.extend([float(x.item()) for x in nv])
        else:
            noise_vals.append(float(nv))

    if len(noise_vals) == 0:
        return noise_vals, float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    t = torch.tensor(noise_vals, dtype=torch.float32)
    mean = float(t.mean().item())
    median = float(t.median().item())
    std = float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0
    minv = float(t.min().item())
    maxv = float(t.max().item())
    return noise_vals, mean, median, std, minv, maxv


def save_results_csv(rows: List[dict], outdir: str, fname: str = "noise_sweep_results.csv"):
    path = os.path.join(outdir, fname)
    if not rows:
        print("No rows to save.")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results CSV to: {path}")


def plot_gpo_noise_vs_injected_noise(rows: List[dict], outdir: str, system_name: str, noise_type: str):
    """
    Plot gpo_noise_mean vs injected_noise for each training dataset size.
    """
    if not rows:
        print("No rows available for plotting.")
        return

    train_sizes = sorted(set(r["train_size"] for r in rows))

    fig, ax = plt.subplots(figsize=(7, 5))

    for train_size in train_sizes:
        subset = [r for r in rows if r["train_size"] == train_size]
        subset = sorted(subset, key=lambda r: r["noise_intensity"])

        x = [100.0 * r["noise_intensity"] for r in subset]
        y = [r["gpo_noise_mean"] for r in subset]

        ax.plot(x, y, marker='o', linewidth=1.8, label=f"{train_size} traj")

    ax.set_xlabel("Injected noise intensity (%)")
    ax.set_ylabel("Mean learned GP noise")
    ax.set_title(f"{system_name}: learned GP noise vs injected noise")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Training size")

    plt.tight_layout()
    save_path = os.path.join(
        outdir, f"{system_name}_{noise_type}_gpo_noise_vs_injected_noise.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {save_path}")


def plot_train_nrmse_vs_injected_noise(rows: List[dict], outdir: str, system_name: str, noise_type: str):
    """
    Plot train_nrmse_mean vs injected_noise for each training dataset size.
    """
    if not rows:
        print("No rows available for plotting.")
        return

    train_sizes = sorted(set(r["train_size"] for r in rows))

    fig, ax = plt.subplots(figsize=(7, 5))

    for train_size in train_sizes:
        subset = [r for r in rows if r["train_size"] == train_size]
        subset = sorted(subset, key=lambda r: r["noise_intensity"])

        x = [100.0 * r["noise_intensity"] for r in subset]
        y = [100.0 * r["train_nrmse_mean"] for r in subset]  # percent

        ax.plot(x, y, marker='o', linewidth=1.8, label=f"{train_size} traj")

    ax.set_xlabel("Injected noise intensity (%)")
    ax.set_ylabel("Mean training NRMSE (%)")
    ax.set_title(f"{system_name}: training NRMSE vs injected noise")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Training size")

    plt.tight_layout()
    save_path = os.path.join(
        outdir, f"{system_name}_{noise_type}_train_nrmse_vs_injected_noise.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {save_path}")


def plot_mean_runtime_vs_train_size(rows: List[dict], outdir: str, system_name: str, noise_type: str):
    """
    Plot mean computation time (averaged across noise intensities)
    vs training dataset size.
    """
    if not rows:
        print("No rows available for plotting.")
        return

    train_sizes = sorted(set(r["train_size"] for r in rows))
    mean_runtimes = []

    for train_size in train_sizes:
        subset = [r for r in rows if r["train_size"] == train_size]
        runtimes = [r["runtime_sec"] for r in subset]
        mean_runtimes.append(sum(runtimes) / len(runtimes))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_sizes, mean_runtimes, marker='o', linewidth=1.8)

    ax.set_xlabel("Training dataset size (number of trajectories)")
    ax.set_ylabel("Mean computation time (s)")
    ax.set_title(f"{system_name}: mean runtime vs training dataset size")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(
        outdir, f"{system_name}_{noise_type}_mean_runtime_vs_train_size.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {save_path}")

# ============================================================
# USER SETTINGS
# ============================================================


SYSTEM_NAME = "IPP-Large"

TRAIN_FRAC_LIST = [0.025, 0.050, 0.100, 0.200, 0.400]
NOISE_TYPE = "gaussian"          # can also include "uniform"
NOISE_INTENSITIES = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
NORMALIZE_DATA = True
CLIP = 100
TEST_FRAC = 0.01
LIFTED_ORDER = 10
MAX_ITER = 2000
LEARN_RATE = 0.001

OUTDIR = f"Figures/Journal/Epistemic_Uncertainty_{SYSTEM_NAME}"
os.makedirs(OUTDIR, exist_ok=True)

# -------------------------------------------------
# Run experiments
# -------------------------------------------------
all_rows = []
for train_frac, intensity in itertools.product(TRAIN_FRAC_LIST, NOISE_INTENSITIES):

    print(
        f"\n=== {NOISE_TYPE} | INTENSITY={100*intensity:.1f}% | TRAIN-SIZE={train_frac} ===")

    # Find Scale of Hyperparameter Initialization
    SimData_raw, _, _, N, nTrain, nTest = gpk.load_SimData(
        SYSTEM_NAME, train_frac, TEST_FRAC, clip=CLIP)
    SimData_raw = SimData_raw[:nTrain+10, :, :]
    if NORMALIZE_DATA:
        SimData_clean, _, _ = gpk.normalize_data(
            SimData_raw, nTrain, N)
    else:
        SimData_clean = SimData_raw
    SimData = gpk.add_noise(SimData_clean, noise_type=NOISE_TYPE,
                            intensity=intensity)
    hp_scale = find_hp_init(SimData, nTrain)

    t0 = time.perf_counter()
    results = get_iGPK(
        SimData=SimData,
        nTrain=nTrain, nTest=nTest,
        lifting_order=LIFTED_ORDER,
        max_iter=MAX_ITER,
        learn_rate=LEARN_RATE,
        hp_scale=[None, hp_scale, None],
    )
    t_iGPK = time.perf_counter() - t0
    # unpack iGPK
    TrainNRMSE = results["Train"]["NRMSE"]
    _, gpo_noise_mean, gpo_noise_median, gpo_noise_std, gpo_noise_min, gpo_noise_max = _extract_gp_noise_stats(
        results["ObsManager"])
    train_perf_mean = TrainNRMSE.mean()

    row = {
        "system_name": SYSTEM_NAME,
        "noise_type": NOISE_TYPE,
        "noise_intensity": float(intensity),
        "n_train": int(nTrain),
        "n_test": int(nTest),
        "gpo_noise_mean": float(gpo_noise_mean),
        "gpo_noise_median": float(gpo_noise_median),
        "gpo_noise_std": float(gpo_noise_std),
        "gpo_noise_min": float(gpo_noise_min),
        "gpo_noise_max": float(gpo_noise_max),
        "train_nrmse_mean": float(train_perf_mean),
        "runtime_sec": float(t_iGPK),
    }
    all_rows.append(row)

    print(f"  mean learned GP noise = {gpo_noise_mean:.6g}")
    print(f"  mean train NRMSE      = {train_perf_mean:.6g}")
    print(f"  runtime               = {t_iGPK:.2f} s")


# ============================================================
# SAVE RESULTS + PLOTS
# ============================================================

save_results_csv(all_rows, OUTDIR,
                 fname=f"{SYSTEM_NAME}_{NOISE_TYPE}_noise_sweep_results.csv")

plot_gpo_noise_vs_injected_noise(
    all_rows,
    outdir=OUTDIR,
    system_name=SYSTEM_NAME,
    noise_type=NOISE_TYPE,
)

plot_train_nrmse_vs_injected_noise(
    all_rows,
    outdir=OUTDIR,
    system_name=SYSTEM_NAME,
    noise_type=NOISE_TYPE,
)

plot_mean_runtime_vs_train_size(
    all_rows,
    outdir=OUTDIR,
    system_name=SYSTEM_NAME,
    noise_type=NOISE_TYPE,
)
