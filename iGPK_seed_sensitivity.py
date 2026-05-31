import os
import math
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import GPKoopman as gpk
from get_iGPK_fcn import FULL_COST_EVAL_EVERY
from get_iGPK_new import get_iGPK
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# ----------------------------
# Utilities
# ----------------------------


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


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism flags (optional; can slow things down a bit)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def _extract_costs(results: dict):
    """
    Adapter: pull 'cost_history' and 'final_train_cost' from your results dict.
    Edit here if your keys differ.
    """
    # Common patterns:
    cost_hist = results["history"].get("cost", None)

    final_cost = results.get("final_train_cost", None)

    cost_hist = cost_hist.detach().cpu().double().reshape(-1).numpy()
    if torch.is_tensor(final_cost):
        final_cost = float(final_cost.detach().cpu().item())
    elif isinstance(final_cost, np.ndarray):
        final_cost = float(final_cost.reshape(-1)[0])
    elif final_cost is not None:
        final_cost = float(final_cost)

    return cost_hist, final_cost


def plot_final_cost_surface(
    runs,
    z_key="z_seed",
    hp_key="hp_seed",
    cost_key="final_train_cost",
    labels: list = ['X', 'Y', 'Z', 'Title'],
    outdir=".",
    fname_stub="final_cost_surface"
):
    os.makedirs(outdir, exist_ok=True)
    save_path = os.path.join(outdir, f"{fname_stub}.png")

    # 1) Unique sorted seed values
    z_vals = sorted({r[z_key] for r in runs})
    hp_vals = sorted({r[hp_key] for r in runs})

    # 2) Build cost grid
    Zgrid = np.full((len(z_vals), len(hp_vals)), np.nan, dtype=float)

    z_to_i = {z: i for i, z in enumerate(z_vals)}
    hp_to_j = {h: j for j, h in enumerate(hp_vals)}

    for r in runs:
        i = z_to_i[r[z_key]]
        j = hp_to_j[r[hp_key]]
        Zgrid[i, j] = float(r.get(cost_key, np.nan))

    # 3) Meshgrid
    HP, Z = np.meshgrid(hp_vals, z_vals)

    # 4) Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(HP, Z, Zgrid, linewidth=0, antialiased=True)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(labels[3])

    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=labels[2])
    plt.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return (z_vals, hp_vals, Zgrid)


def plot_final_cost_heatmap(
    runs,
    z_key="z_seed",
    hp_key="hp_seed",
    cost_key="final_train_cost",
    labels: list = ['X', 'Y', 'Z', 'Title'],
    outdir=".",
    fname_stub="final_cost_heatmap"
):
    os.makedirs(outdir, exist_ok=True)
    save_path = os.path.join(outdir, f"{fname_stub}.png")

    # 1) Unique sorted seed values
    z_vals = sorted({r[z_key] for r in runs})
    hp_vals = sorted({r[hp_key] for r in runs})

    # 2) Build cost grid
    Zgrid = np.full((len(z_vals), len(hp_vals)), np.nan, dtype=float)

    z_to_i = {z: i for i, z in enumerate(z_vals)}
    hp_to_j = {h: j for j, h in enumerate(hp_vals)}

    for r in runs:
        i = z_to_i[r[z_key]]
        j = hp_to_j[r[hp_key]]
        Zgrid[i, j] = float(r.get(cost_key, np.nan))

    # 3) Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        Zgrid,
        origin="lower",
        aspect="auto",
        interpolation="nearest"
    )

    ax.set_xticks(np.arange(len(hp_vals)))
    ax.set_yticks(np.arange(len(z_vals)))

    ax.set_xticklabels(hp_vals)
    ax.set_yticklabels(z_vals)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(labels[3])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(labels[2])

    plt.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return (z_vals, hp_vals, Zgrid)


# ----------------------------
# Run Experiments
# ----------------------------
if __name__ == "__main__":

    # 1) EXPERIMENT CONFIGURATION
    system_name = 'Cart_data'
    train_frac, test_frac = 0.6, 0.4
    clip = None
    lifted_order = 35
    noise_type, noise_intensity = 'gaussian', 0.
    MAX_ITER = int(1e5)
    SGD_LR, SGD_MOM = 0.01, 0.7
    OPT_WEIGHTS = [10., 10., 0.]
    ROUTINE = "Z-only"
    TRAIN_METHOD = "Zero-Mean"
    device = "cuda:0"

    OUTDIR = f"Figures/iGPK-Seed_Sensitivity/{system_name}/LR-{SGD_LR:.2e}_MOM-{SGD_MOM:.2f}"
    hp_seeds, z_seeds = [1234], [1, 5, 10, 19, 50, 77, 100, 111, 123, 241, 511, 777, 1234, 1419, 2000]
    os.makedirs(OUTDIR, exist_ok=True)

    # 1.1) Load and Normalize Data
    SimData_raw, ts, num_traj, N, nTrain, nTest = gpk.load_SimData(
        system_name, train_frac, test_frac, clip=clip)
    SimData_clean, mu_vec, std_vec = gpk.normalize_data(SimData_raw, nTrain, N)

    # 2) Find Initial Hyperparameter
    HP_INIT = gpk.find_hp_init(SimData_clean[nTest:nTest+nTrain, :, :-1])
    print(f'Heuristic Kernel-lengthscale param found to be {HP_INIT:.3e}')

    # 1.2) Add Noise - Optional
    SimData = gpk.add_noise(SimData_clean, noise_type=noise_type,
                            intensity=noise_intensity, seed=100)

    # 2) RUN EXPERIMENTS
    runs = []
    run_id = 0

    BATCH_SIZE = 15
    FULL_COST_EVAL_EVERY = 50

    for z_seed in z_seeds:

        for hp_seed in hp_seeds:
            run_id += 1
            tag = f'run-{run_id}_zs-{z_seed}_hps-{hp_seed}'
            print(
                f'Tag: {tag} || Run: {run_id} | Z Seed: {z_seed} | HP Seed: {hp_seed}\n')

            results = get_iGPK(SimData, nTrain, nTest, lifted_order,
                        MAX_ITER, sgd_lr=SGD_LR, sgd_m=SGD_MOM,
                        opt_weights=OPT_WEIGHTS, routine=ROUTINE,
                        train_method=TRAIN_METHOD, hp_scale=[None, HP_INIT, None],
                        seed_z=z_seed, seed_hp=hp_seed,
                        traj_batch_size=BATCH_SIZE,
                        full_cost_eval_every=FULL_COST_EVAL_EVERY,
                        )

            # Pull metrics
            final_cost_mle = results['history']['post_mle_cost']

            cost_history = results['history']['cost']           # tensor
            post_mle_cost = results['history']['post_mle_cost'] # tensor

            train_nlpd = gpk.nlpd_per_traj(results['Train']['Xhat'],
                                          results['Train']['Xcv'],
                                          SimData[:nTrain, :, :N])
            test_nlpd = gpk.nlpd_per_traj(results['Test']['Xhat'],
                                          results['Test']['Xcv'],
                                          SimData[nTrain:, :, :N])

            runs.append({
                "run_id": run_id,
                "tag": tag,
                "z_seed": z_seed,
                "hp_seed": hp_seed,
                "cost_history": cost_history,
                # "final_train_cost": cost_history[-1],
                "post_mle_cost": results['history']['post_mle_cost'],
                "Train": {
                    "NRMSE": results['Train'].get('NRMSE', None).mean(),
                    "NLPD":  train_nlpd.mean(),
                    },
                "Test": {
                    "NRMSE": results['Test'].get('NRMSE', None).mean(),
                    "NLPD":  test_nlpd.mean(),
                    },
                })
            
            print(f'Finished Run ID {run_id}')

    # Save everything
    torch.save(runs, os.path.join(OUTDIR, "iGPK-SEED_sweep-runs.pt"))

    # ---------------------------
    # 3) PLOT AND SAVE RESULTS  #
    # ---------------------------
    # Gather major Results
    cost_pre, cost_post = [], []
    train_nrmse, test_nrmse = [], []
    train_nlpd, test_nlpd = [], []
    for run in runs:
        cost_pre.append(float(run['cost_history'][-1]))
        cost_post.append(float(run['post_mle_cost']))
        train_nrmse.append(float(100*run['Train']['NRMSE']))
        train_nlpd.append(float(run['Train']['NLPD']))
        test_nrmse.append(float(100*run['Test']['NRMSE']))
        test_nlpd.append(float(run['Test']['NLPD']))

    if True:    # PLOT 1 : COST HISTORY
        plt.figure(figsize=(10, 6))
        for r in runs:
            ch = r["cost_history"]
            if ch is None or len(ch) == 0:
                continue
            ch_plot = np.clip(ch, 1e-16, None)
            plt.plot(np.arange(len(ch_plot)), ch_plot,
                    linewidth=1.0, alpha=0.85, label=r["tag"])

        plt.yscale("log")
        plt.xlabel("GD Iteration")
        plt.ylabel("Training Cost (log scale)")
        plt.title("iGPK Cost Histories Across Initializations")
        if len(runs) <= 16:
            plt.legend(fontsize=8, ncol=2)
        else:
            plt.text(0.01, 0.01, f"{len(runs)} runs (legend suppressed)",
                    transform=plt.gca().transAxes, fontsize=9, va="bottom")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "cost_histories_log.png"), dpi=200)
        plt.grid()
        plt.close()

    # Plot 2: final_train_cost distribution
    if True:    # PLOT 2 : PRE AND POST MLE COST
        plt.figure(figsize=(5, 5))
        plt.scatter(cost_pre, cost_post, alpha=0.75)
        # plt.xscale('log'), plt.yscale('log')
        plt.xlabel('Pre-MLE Cost'), plt.ylabel('Post-MLE Cost')
        plt.grid()
        plt.title('Post-MLE v/s Pre-MLE Training Cost')
        plt.tight_layout()
        plt.savefig(os.path.join(
            OUTDIR, "iGPK-pr_vs_post-train_cost.png"), dpi=200)
        plt.close()

    if True:    # PLOT 3 : PRE and POST MLE COST V/S MEAN TRAIN %-NRMSE
        fig, ax = plt.subplots(1, 2, sharey=True, figsize=(8, 5))
        ax[0].scatter(cost_pre, train_nrmse, alpha=0.75)
        ax[1].scatter(cost_post, train_nrmse, alpha=0.75)
        # ax[0].set_xscale('log'), ax[1].set_xscale('log')
        ax[0].set_xlabel('Pre-MLE Cost'), ax[1].set_xlabel('Post-MLE Cost')
        ax[0].set_ylabel('Train NRMSE [%]')
        # Cleaner ticks
        for a in ax:
            a.grid(True)
            a.xaxis.set_major_locator(MaxNLocator(nbins=5))
            # Consistent decimal formatting
            a.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        fig.suptitle('Train NRMSE [%] v/s Pre and Post-MLE Training Cost')
        plt.tight_layout()
        plt.savefig(os.path.join(
            OUTDIR, "iGPK-train_nrmse.png"), dpi=200)
        plt.close()

    if True:    # PLOT 4 : POST-MLE COST V/S MEAN TEST %-NRMSE
        fig, ax = plt.subplots(1, 2, sharey=True, figsize=(8, 5))
        ax[0].scatter(cost_pre, test_nrmse, alpha=0.75)
        ax[1].scatter(cost_post, test_nrmse, alpha=0.75)
        # ax[0].set_xscale('log'), ax[1].set_xscale('log')
        ax[0].set_xlabel('Pre-MLE Cost'), ax[1].set_xlabel('Post-MLE Cost')
        ax[0].set_ylabel('Test NRMSE [%]')
        # Cleaner ticks
        for a in ax:
            a.grid(True)
            a.xaxis.set_major_locator(MaxNLocator(nbins=5))
            # Consistent decimal formatting
            a.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        fig.suptitle('Test NRMSE [%] v/s Pre and Post-MLE Training Cost')
        plt.tight_layout()
        plt.savefig(os.path.join(
            OUTDIR, "iGPK-test_nrmse.png"), dpi=200)
        plt.close()

    if True:    # PLOT 5 : POST-MLE COST V/S MEAN TEST NLPD
        fig, ax = plt.subplots(1, 2, sharey=True, figsize=(8, 5))
        ax[0].scatter(cost_pre, train_nlpd, alpha=0.75)
        ax[1].scatter(cost_post, train_nlpd, alpha=0.75)
        # ax[0].set_xscale('log'), ax[1].set_xscale('log')
        ax[0].set_xlabel('Pre-MLE Cost'), ax[1].set_xlabel('Post-MLE Cost')
        ax[0].set_ylabel('Train NLPD')
        # Cleaner ticks
        for a in ax:
            a.grid(True)
            a.xaxis.set_major_locator(MaxNLocator(nbins=5))
            # Consistent decimal formatting
            a.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        fig.suptitle('Train NLPD v/s Pre and Post-MLE Training Cost')
        plt.tight_layout()
        plt.savefig(os.path.join(
            OUTDIR, "iGPK-train_nlpd.png"), dpi=200)
        plt.close()

    if True:    # PLOT 6 : POST-MLE COST V/S MEAN TEST NLPD
        fig, ax = plt.subplots(1, 2, sharey=True, figsize=(8, 5))
        ax[0].scatter(cost_pre, test_nlpd, alpha=0.75)
        ax[1].scatter(cost_post, test_nlpd, alpha=0.75)
        # ax[0].set_xscale('log'), ax[1].set_xscale('log')
        ax[0].set_xlabel('Pre-MLE Cost'), ax[1].set_xlabel('Post-MLE Cost')
        ax[0].set_ylabel('Test NLPD')
        # Cleaner ticks
        for a in ax:
            a.grid(True)
            a.xaxis.set_major_locator(MaxNLocator(nbins=5))
            # Consistent decimal formatting
            a.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        fig.suptitle('Test NLPD v/s Pre and Post-MLE Training Cost')
        plt.tight_layout()
        plt.savefig(os.path.join(
            OUTDIR, "iGPK-test_nlpd.png"), dpi=200)
        plt.close()

    print(f'Finishied Plotting')


