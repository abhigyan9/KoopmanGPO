import os
import math
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import GPKoopman as gpk
from get_iGPK_new import get_iGPK

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
    system_name = 'Inhibited Predator-Prey'
    train_frac, test_frac = 0.4, 0.6
    clip = None
    lifted_order = 10
    noise_type, noise_intensity = 'gaussian', 0.
    iters_list = [0, 0, 0, 500]
    learn_rate = 0.001
    opt_weights = [1., 1., 1.]
    routine = "Z-only"
    train_method = "Horizon"
    device = "cuda:0"

    OUTDIR = f"Figures/iGPK_Seed_Sensitivity/{routine}"
    hp_seeds, z_seeds = [1, 3, 7, 8, 40, 59, 61], [11, 13, 14, 21, 33, 43, 53]
    os.makedirs(OUTDIR, exist_ok=True)

    # 1.1) Load and Normalize Data
    SimData_raw, ts, num_traj, N, nTrain, nTest = gpk.load_SimData(
        system_name, train_frac, test_frac, clip=clip)
    SimData_clean, mu_vec, std_vec = gpk.normalize_data(SimData_raw, nTrain, N)

    # 2) Find Initial Hyperparameter
    HP_INIT = find_hp_init(SimData_clean, nTrain)
    print(f'Heuristic Kernel-lengthscale param found to be {HP_INIT:.3e}')

    # 1.2) Add Noise - Optional
    SimData = gpk.add_noise(SimData_clean, noise_type=noise_type,
                            intensity=noise_intensity, seed=100)

    # 2) RUN EXPERIMENTS
    runs = []
    run_id = 0

    for z_seed in z_seeds:

        for hp_seed in hp_seeds:
            run_id += 1
            tag = f'run-{run_id}_zs-{z_seed}_hps-{hp_seed}'
            print(
                f'Tag: {tag} || Run: {run_id} | Z Seed: {z_seed} | HP Seed: {hp_seed}\n')

            results = get_iGPK(
                SimData, nTrain, nTest, lifted_order,
                iters_list, learn_rate,
                opt_weights, routine,
                train_method, hp_scale=[None, HP_INIT, None],
                device=device, seed_z=z_seed, seed_hp=hp_seed
            )

            # Pull metrics
            cost_hist, final_cost = _extract_costs(results)

            A = results["A"]
            A = A.detach().cpu().numpy()
            eigvals = np.linalg.eigvals(A)

            runs.append({
                "run_id": run_id,
                "tag": tag,
                "z_seed": z_seed,
                "hp_seed": hp_seed,
                "cost_history": cost_hist,
                "final_train_cost": final_cost,
                "TrainNRMSE": results['Train'].get('NRMSE', None),
                "TestNRMSE": results['Test'].get('NRMSE', None),
                "TrainErrMean": torch.mean(results['Train'].get('NRMSE', None)),
                "TestErrMean": torch.mean(results['Test'].get('NRMSE', None)),
            })

    # Save everything
    torch.save(runs, os.path.join(OUTDIR, "iGPK_init_sweep_runs.pt"))

    # ---------------------------
    # 3) PLOT AND SAVE RESULTS  #
    # ---------------------------

    # Plot 1: cost histories
    plt.figure(figsize=(10, 6))
    for r in runs:
        ch = r["cost_history"]
        if ch is None or len(ch) == 0:
            continue
        ch_plot = np.clip(ch, 1e-16, None)
        plt.plot(np.arange(len(ch_plot)), ch_plot,
                 linewidth=1.25, alpha=0.85, label=r["tag"])

    plt.yscale("log")
    plt.xlabel("GD Iteration")
    plt.ylabel("Training Cost (log scale)")
    plt.title("iGPK Cost Histories Across Initializations")
    if len(runs) <= 12:
        plt.legend(fontsize=8, ncol=1)
    else:
        plt.text(0.01, 0.01, f"{len(runs)} runs (legend suppressed)",
                 transform=plt.gca().transAxes, fontsize=9, va="bottom")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "cost_histories_log.png"), dpi=200)
    plt.close()

    # Plot 2: final_train_cost distribution
    finals = np.array([r["final_train_cost"]
                      for r in runs if r["final_train_cost"] is not None], dtype=np.float64)
    plt.figure(figsize=(9, 5))
    if len(finals) > 0:
        plt.hist(finals, bins=min(30, max(5, int(math.sqrt(len(finals))))),
                 edgecolor="k", linewidth=0.5)
        plt.xlabel("final_train_cost")
        plt.ylabel("Count")
        plt.title("Distribution of final_train_cost Across Initializations")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    else:
        plt.text(0.5, 0.5, "No final_train_cost values found.",
                 ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(
        OUTDIR, "iGPK_final_train_cost_hist.png"), dpi=200)
    plt.close()

    # Plot 3 (6 plots): Surface Plots and Heatmaps of Final Train Cost and Mean Train and Test NRMSE
    if True:
        # PLOT 3 A
        plot_final_cost_surface(
            runs,
            labels=["HP Seed", "Z Seed", "Final Train Cost",
                    "iGPK Final Cost Surface"],
            outdir=OUTDIR,
            fname_stub=f"{system_name}_finalcost_surface"
        )
        plot_final_cost_heatmap(
            runs,
            labels=["HP Seed", "Z Seed", "Final Train Cost",
                    "iGPK Final Cost Heatmap"],
            outdir=OUTDIR,
            fname_stub=f"{system_name}_finalcost_heatmap"
        )
        # PLOT 3 B
        plot_final_cost_surface(
            runs,
            cost_key='TrainErrMean',
            labels=["HP Seed", "Z Seed", "Mean Train NRMSE",
                    "iGPK Mean Train NRMSE Surface"],
            outdir=OUTDIR,
            fname_stub=f"{system_name}_meanNRMSE_Train_surface"
        )
        plot_final_cost_heatmap(
            runs,
            cost_key='TrainErrMean',
            labels=["HP Seed", "Z Seed", "Mean Train NRMSE",
                    "iGPK Mean Train NRMSE HeatMap"],
            outdir=OUTDIR,
            fname_stub=f"{system_name}_meanNRMSE_Train_heatmap"
        )
        # PLOT 3 C
        plot_final_cost_surface(
            runs,
            cost_key='TestErrMean',
            labels=["HP Seed", "Z Seed", "Mean Test NRMSE",
                    "iGPK Mean Test NRMSE Surface"],
            outdir=OUTDIR,
            fname_stub=f"{system_name}_meanNRMSE_Test_surface"
        )
        plot_final_cost_heatmap(
            runs,
            cost_key='TestErrMean',
            labels=["HP Seed", "Z Seed", "Mean Test NRMSE",
                    "iGPK Mean Test NRMSE HeatMap"],
            outdir=OUTDIR,
            fname_stub=f"{system_name}_meanNRMSE_Test_heatmap"
        )

    # --- NEW Plot 3: Eigenvalues of A in complex plane across seeds ---
    plt.figure(figsize=(7, 7))
    have_any = False
    for r in runs:
        ev = r.get("eigvals_A", None)
        if ev is None:
            continue
        have_any = True
        plt.scatter(ev.real, ev.imag, s=22, alpha=0.75,
                    label=f"run {r['run_id']}")

    # unit circle reference
    th = np.linspace(0, 2*np.pi, 400)
    plt.plot(np.cos(th), np.sin(th), linewidth=1.0, alpha=0.7)

    plt.axhline(0.0, linewidth=0.8, alpha=0.5)
    plt.axvline(0.0, linewidth=0.8, alpha=0.5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("Re(λ)")
    plt.ylabel("Im(λ)")
    plt.title("Eigenvalues of A across iGPK seeds (complex plane)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    if len(runs) <= 10:
        plt.legend(fontsize=8, ncol=1)
    plt.tight_layout()
    if have_any:
        plt.savefig(os.path.join(
            OUTDIR, "A_eigvals_complex_plane.png"), dpi=220)
    plt.close()

    # --- NEW Plot 4: |eigs| vs mode index, per seed (sorted by magnitude) ---
    plt.figure(figsize=(10, 6))
    have_any = False
    for r in runs:
        ev = r.get("eigvals_A", None)
        if ev is None:
            continue
        have_any = True
        mags = np.abs(ev)
        mags_sorted = np.sort(mags)[::-1]
        plt.plot(np.arange(1, len(mags_sorted) + 1), mags_sorted, linewidth=1.25, alpha=0.85,
                 label=f"run {r['run_id']}")

    plt.xlabel("Mode index (sorted)")
    plt.ylabel("|λ|")
    plt.title("Eigenvalue magnitudes of A across seeds")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    if len(runs) <= 12:
        plt.legend(fontsize=8, ncol=1)
    plt.tight_layout()
    if have_any:
        plt.savefig(os.path.join(OUTDIR, "A_eigvals_magnitudes.png"), dpi=220)
    plt.close()

    # Ranked report
    ranked = sorted(
        [(r["tag"], r["final_train_cost"])
         for r in runs if r["final_train_cost"] is not None],
        key=lambda x: x[1]
    )
    report_path = os.path.join(OUTDIR, "iGPK_final_train_cost_ranked.txt")
    with open(report_path, "w") as f:
        f.write("tag\tfinal_train_cost\tA_path\n")
        for r in runs:
            if r["final_train_cost"] is None:
                continue
            f.write(
                f"{r['tag']}\t{r['final_train_cost']:.6e}\t{r.get('A_path', None)}\n")

    # NRMSE Metrics
    print(f'\n==================================================\n')
    for r in runs:
        print(f'==== Z-SEED: {r['z_seed']} || HP-SEED: {r['hp_seed']} ====')
        Error = 100*r["TrainNRMSE"]
        print(f'\n==== %-NRMSE Metrics | TRAIN ====\n')
        print(
            f'MIN: {Error.min():.3e} || MEDIAN: {Error.median():.3e} || MAX: {Error.max():.3e}\n')
        print(f'MEAN: {Error.mean():.3e} || STD: {Error.std():.3e}\n')

        Error = 100*r["TestNRMSE"]
        print(f'====== %-NRMSE Metrics | TEST ======\n')
        print(
            f'MIN: {Error.min():.3e} || MEDIAN: {Error.median():.3e} || MAX: {Error.max():.3e}')
        print(f'MEAN: {Error.mean():.3e} || STD: {Error.std():.3e}')
        print('==================================================')

    print(f"\nSaved:")
    print(f"  - {os.path.join(OUTDIR, 'iGPK_init_sweep_runs.pt')}")
    print(f"  - {os.path.join(OUTDIR, 'cost_histories_log.png')}")
    print(f"  - {os.path.join(OUTDIR, 'iGPK_final_train_cost_hist.png')}")
    print(f"  - {os.path.join(OUTDIR, 'A_eigvals_complex_plane.png')}")
    print(f"  - {os.path.join(OUTDIR, 'A_eigvals_magnitudes.png')}")
    print(f"  - {report_path}")
