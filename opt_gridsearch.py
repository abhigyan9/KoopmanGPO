# sweep_igpk_optim_concise.py
import os
import gc
import itertools
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import GPKoopman as gpk
from get_iGPK_fcn import get_iGPK

import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# User configuration
# -----------------------------
SYSTEM_NAME = "Inhibited Predator-Prey"
TRAIN_FRAC = 0.60
TEST_FRAC = 1.0 - TRAIN_FRAC
CLIP = None

NORMALIZE_DATA = True
NOISE_TYPE = "gaussian"
NOISE_INTENSITY = 0.0
NOISE_SEED = 100

LIFTING_ORDER = 10
MAX_ITER = 50000
DEVICE = "cuda:0"

LEARN_RATES = [0.02, 0.01, 0.001]
MOMENTUMS = [0.7, 0.75, 0.8]
STOP_TOLS = [1e-3, 1e-4]
# LEARN_RATES = [0.001,]
# MOMENTUMS = [0.75,]
# STOP_TOLS = [0.001,]

# NEW: trajectory-wise mini-batch sizes.
# Use None to include one full-batch run.
# These values are clipped later so batch_size <= nTrain.
TRAJ_BATCH_SIZES = [20, 30]
# TRAJ_BATCH_SIZES = [15, ]

# For the modified get_iGPK with trajectory-wise batches.
FULL_COST_EVAL_EVERY = 50

OPT_WEIGHTS = [1.0, 1.0, 0.0]
ROUTINE = "standard"  # OR "multi-perturb"
TRAIN_METHOD = "Zero-Mean"

SEED_Z = 1234
SEED_HP = 1234

STAMP = datetime.now().strftime("%Y%m%d")
OUTDIR = f"Figures/GridSearch/Trial123_delta2_{SYSTEM_NAME.replace(' ', '_')}_{LIFTING_ORDER}D_batchSGD_{STAMP}_Z-{SEED_Z}_HP-{SEED_HP}"
os.makedirs(OUTDIR, exist_ok=True)


# -----------------------------
# Small helpers
# -----------------------------
def to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().mean())
    return float(x)


def batch_label(batch_size, nTrain):
    if batch_size is None or batch_size >= nTrain:
        return "full"
    return str(int(batch_size))


def actual_batch_size(batch_size, nTrain):
    if batch_size is None or batch_size >= nTrain:
        return nTrain
    return int(batch_size)


def plot_surface(df, metric, zlabel):
    """
    Creates one LR-vs-momentum surface per stop_tol and trajectory batch size.
    """
    metric_dir = os.path.join(OUTDIR, metric)
    os.makedirs(metric_dir, exist_ok=True)

    for stop_tol in sorted(df["stop_tol"].unique()):
        for b in sorted(df["traj_batch_size"].unique()):
            sub = df[
                (df["stop_tol"] == stop_tol)
                & (df["traj_batch_size"] == b)
            ]

            if sub.empty:
                continue

            Z = sub.pivot_table(
                index="momentum",
                columns="learn_rate",
                values=metric,
                aggfunc="mean",
            )

            if Z.empty:
                continue

            X_grid, Y_grid = np.meshgrid(Z.columns.values, Z.index.values)

            fig = plt.figure(figsize=(7, 5.5))
            ax = fig.add_subplot(111, projection="3d")

            ax.plot_surface(
                X_grid,
                Y_grid,
                Z.values,
                edgecolor="k",
                linewidth=0.3,
                alpha=0.9,
            )

            ax.set_xscale("log")
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("Momentum")
            ax.set_zlabel(zlabel)
            ax.set_title(f"{zlabel} | stop_tol={stop_tol:.1e} | traj_batch_size={b}")

            fig.tight_layout()

            fname = f"{metric}_stop_tol_{stop_tol:.1e}_batch_{b}.png"
            fig.savefig(
                os.path.join(metric_dir, fname),
                dpi=250,
                bbox_inches="tight",
            )
            plt.close(fig)


def save_best_rows(df):
    """
    Saves sorted summaries for the main metrics.
    """
    summary_dir = os.path.join(OUTDIR, "best_summaries")
    os.makedirs(summary_dir, exist_ok=True)

    for metric in [
        "pre_mle_cost",
        "post_mle_cost",
        "train_nrmse",
        "test_nrmse",
        "train_nlpd",
        "test_nlpd",
    ]:
        df_sorted = df.sort_values(metric, ascending=True)
        df_sorted.head(20).to_csv(
            os.path.join(summary_dir, f"best_by_{metric}.csv"),
            index=False,
        )


# -----------------------------
# Load data
# -----------------------------
SimData_raw, ts, num_traj, N, nTrain, nTest = gpk.load_SimData(
    SYSTEM_NAME,
    TRAIN_FRAC,
    TEST_FRAC,
    clip=CLIP,
)

if NORMALIZE_DATA:
    SimData_clean, mu_vec, std_vec = gpk.normalize_data(
        SimData_raw.to(dtype=torch.float32), nTest, nTrain, N)
else:
    SimData_clean = SimData_raw

hp2_scale = gpk.find_hp_init(SimData_clean[nTest:nTest+nTrain, :, :-1])
hp_scale = [None, hp2_scale, None]

SimData = gpk.add_noise(
    SimData_clean,
    noise_type=NOISE_TYPE,
    intensity=NOISE_INTENSITY,
    seed=NOISE_SEED,
)

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

# Clean and validate batch sizes after nTrain is known.
TRAJ_BATCH_SIZES_CLEAN = []
for b in TRAJ_BATCH_SIZES:
    if b is None:
        TRAJ_BATCH_SIZES_CLEAN.append(None)
    elif int(b) <= nTrain:
        TRAJ_BATCH_SIZES_CLEAN.append(int(b))

# Remove duplicates while preserving order.
seen = set()
tmp = []
for b in TRAJ_BATCH_SIZES_CLEAN:
    key = "full" if b is None else int(b)
    if key not in seen:
        tmp.append(b)
        seen.add(key)

TRAJ_BATCH_SIZES_CLEAN = tmp

print("---------------------------------------")
print(f"System        : {SYSTEM_NAME}")
print(f"Train/Test    : {nTrain}/{nTest}")
print(f"N             : {N}")
print(f"hp_scale      : {hp2_scale:.3e}")
print(f"Batch sizes   : {[batch_label(b, nTrain) for b in TRAJ_BATCH_SIZES_CLEAN]}")
print(f"Output folder : {OUTDIR}")


# -----------------------------
# Grid search
# -----------------------------
rows = []

grid = itertools.product(
    LEARN_RATES,
    MOMENTUMS,
    TRAJ_BATCH_SIZES_CLEAN,
    STOP_TOLS,
)

num_runs = (
    len(LEARN_RATES)
    * len(MOMENTUMS)
    * len(TRAJ_BATCH_SIZES_CLEAN)
    * len(STOP_TOLS)
)

for run_idx, (lr, momentum, traj_batch_size, stop_tol) in enumerate(grid, start=1):
    b_actual = actual_batch_size(traj_batch_size, nTrain)
    b_label = batch_label(traj_batch_size, nTrain)

    print("--------------------------------------------")
    print(
        f"Run {run_idx:04d}/{num_runs:04d} | "
        f"lr={lr:.2e}, momentum={momentum:.2f}, "
        f"traj_batch_size={b_label}, stop_tol={stop_tol:.1e}"
    )

    results = get_iGPK(
        Data=Dataset,
        nTrain=nTrain,
        nTest=nTest,
        lifting_order=LIFTING_ORDER,
        max_iter=MAX_ITER,
        sgd_lr=lr,
        sgd_m=momentum,
        stop_tol=stop_tol,
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

    # Prefer full_cost if available because mini-batch cost is noisy.
    if "full_cost" in results["history"]:
        pre_mle_cost = results["history"]["full_cost"][-1]
    else:
        pre_mle_cost = results["history"]["cost"][-1]

    train_nlpd = gpk.nlpd_per_traj(
        results["Train"]["Xhat"][:,:,:N-1],
        results["Train"]["Xcv"][:,:,:,:N-1],
        SimData_clean[:nTrain, :, :N-1],
    )

    test_nlpd = gpk.nlpd_per_traj(
        results["Test"]["Xhat"][:,:,:N-1],
        results["Test"]["Xcv"][:,:,:,:N-1],
        SimData_clean[nTrain:nTrain + nTest, :, :N-1],
    )

    row = {
        "run_idx": run_idx,

        "learn_rate": lr,
        "momentum": momentum,
        "traj_batch_size": b_actual,
        "traj_batch_label": b_label,
        "stop_tol": stop_tol,

        "pre_mle_cost": to_float(pre_mle_cost),
        "post_mle_cost": to_float(results["history"]["post_mle_cost"]),

        "train_nrmse": 100.0 * to_float(results["Train"]["NRMSE"].mean()),
        "test_nrmse": 100.0 * to_float(results["Test"]["NRMSE"].mean()),

        "train_nlpd": to_float(train_nlpd.mean()),
        "test_nlpd": to_float(test_nlpd.mean()),

        "iters": int(results["history"]["iters"]),
        "opt_time": to_float(results["history"]["opt_time"]),
    }

    if "best_iter" in results["history"]:
        row["best_iter"] = int(results["history"]["best_iter"])
    else:
        row["best_iter"] = row["iters"]

    if "best_full_cost" in results["history"]:
        row["best_full_cost"] = to_float(results["history"]["best_full_cost"])
    else:
        row["best_full_cost"] = row["pre_mle_cost"]

    rows.append(row)

    # Save after every run.
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTDIR, "grid_results.csv"),
        index=False,
    )

    print(
        f"Finished run {run_idx:04d}/{num_runs:04d} | "
        f"iters={row['iters']} | "
        f"time={row['opt_time']:.2f}s | "
        f"pre={row['pre_mle_cost']:.3e} | "
        f"post={row['post_mle_cost']:.3e} | "
        f"train NRMSE={row['train_nrmse']:.2f}% | "
        f"test NRMSE={row['test_nrmse']:.2f}% | "
        f"train NLPD={row['train_nlpd']:.3f} | "
        f"test NLPD={row['test_nlpd']:.3f}"
    )

    del results
    del train_nlpd
    del test_nlpd

    gc.collect()
    torch.cuda.empty_cache()


# -----------------------------
# Save final table
# -----------------------------
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUTDIR, "grid_results.csv"), index=False)

save_best_rows(df)


# -----------------------------
# Plot surfaces
# -----------------------------
plot_surface(df, "pre_mle_cost", "Pre-MLE Cost")
plot_surface(df, "post_mle_cost", "Post-MLE Cost")
plot_surface(df, "train_nrmse", "Train %-NRMSE")
plot_surface(df, "test_nrmse", "Test %-NRMSE")
plot_surface(df, "train_nlpd", "Train NLPD")
plot_surface(df, "test_nlpd", "Test NLPD")


# -----------------------------
# Print best runs
# -----------------------------
print("---------------------------------------")
print("Best runs by key metrics")
print("---------------------------------------")

for metric in ["post_mle_cost", "test_nrmse", "test_nlpd"]:
    best = df.sort_values(metric, ascending=True).iloc[0]

    print(
        f"\nBest by {metric}:"
        f"\n  value           = {best[metric]:.6e}"
        f"\n  learn_rate      = {best['learn_rate']:.3e}"
        f"\n  momentum        = {best['momentum']:.2f}"
        f"\n  traj_batch_size = {best['traj_batch_label']}"
        f"\n  stop_tol        = {best['stop_tol']:.1e}"
        f"\n  iters           = {int(best['iters'])}"
        f"\n  opt_time        = {best['opt_time']:.2f}s"
    )

print("---------------------------------------")
print(f"All results saved to: {OUTDIR}")
print("---------------------------------------")


metrics = {
    "pre_mle_cost": "Pre-MLE Cost",
    "post_mle_cost": "Post-MLE Cost",
    "train_nrmse": "Train %-NRMSE",
    "test_nrmse": "Test %-NRMSE",
    "train_nlpd": "Train NLPD",
    "test_nlpd": "Test NLPD",
}

for metric, label in metrics.items():
    plot_surface(df, metric, label)

best = df.sort_values("test_nrmse").iloc[0]
print("\nBest by test NRMSE:")
print(best)

print(f"\nSaved results and plots in:\n{OUTDIR}")