import os
import math
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import GPKoopman as gpk
from get_iGPK_fcn import get_iGPK

# ----------------------------
# Utilities
# ----------------------------


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


# ----------------------------
# Run Experiments
# ----------------------------
if __name__ == "__main__":

    # 1) EXPERIMENT CONFIGURATION
    system_name = 'Inhibited Predator-Prey'
    train_frac, test_frac = 0.4, 0.4
    clip = None
    lifted_order = 10
    noise_type, noise_intensity = 'gaussian', 0.
    # unused, samples, iterations, inner iterations
    iters_list = [0, 0, 0, 500]
    learn_rate = 0.0025
    opt_weights = [1., 1., 1.]
    routine = "Z_only"
    train_method = "Horizon"
    device = "cuda:0"

    OUTDIR = "Figures/iGPK_Testing"
    # Choose sweep size (start small; explode later)
    seeds_list = [1, 3, 5, 11, 17, 20, 21, 24, 50, 101, 140, 142]

    os.makedirs(OUTDIR, exist_ok=True)

    # 1.1) Load and Normalize Data
    SimData_raw, ts, num_traj, N, nTrain, nTest = gpk.load_SimData(
        system_name, train_frac, test_frac, clip=clip)
    SimData_clean, mu_vec, std_vec = gpk.normalize_data(
        SimData_raw, nTrain, N)
    # 1.2) Add Noise - Optional
    SimData = gpk.add_noise(SimData_clean, noise_type=noise_type,
                            intensity=noise_intensity, seed=100)

    # 2) RUN EXPERIMENTS
    runs = []
    run_id = 0

    # Grid sweep over (hp_seed, z_seed)
    for igpk_seed in seeds_list:
        run_id += 1
        tag = f"run-{run_id:3d}_seed-{igpk_seed:3d}"

        print(f"[iGPK sweep] {tag} | Iters={iters_list[3]} lr={learn_rate}")

        results = get_iGPK(SimData, nTrain, nTest, lifted_order,
                           iters_list, learn_rate,
                           opt_weights, routine,
                           train_method, device, seed=igpk_seed)

        # Pull metrics
        cost_hist, final_cost = _extract_costs(results)

        runs.append({
            "tag": tag,
            "igpk_seed": igpk_seed,
            "cost_history": cost_hist,
            "final_train_cost": final_cost,
            "raw_results": results,  # keep everything for later inspection
        })

    # Save everything
    torch.save(runs, os.path.join(OUTDIR, "iGPK_init_sweep_runs.pt"))

    # ---------------------------
    # 3) PLOT AND SAVE RESULTS  #
    # ---------------------------
    plt.figure(figsize=(10, 6))
    for r in runs:
        ch = r["cost_history"]
        if ch is None or len(ch) == 0:
            continue
        # Avoid log(0) if you ever hit exactly 0
        ch_plot = np.clip(ch, 1e-16, None)
        plt.plot(np.arange(len(ch_plot)), ch_plot,
                 linewidth=1.25, alpha=0.85, label=r["tag"])

    plt.yscale("log")
    plt.xlabel("GD Iteration")
    plt.ylabel("log10(Training Cost)")
    plt.title("iGPK Cost Histories Across Initializations")
    # If there are many runs, a full legend becomes unusable.
    # Keep legend only if small sweep.
    if len(runs) <= 12:
        plt.legend(fontsize=8, ncol=1)
    else:
        plt.text(
            0.01, 0.01,
            f"{len(runs)} runs (legend suppressed)",
            transform=plt.gca().transAxes,
            fontsize=9,
            va="bottom"
        )
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "cost_histories_log.png"), dpi=200)
    plt.close()

    # Plot 2: final_train_cost across runs
    finals = np.array([r["final_train_cost"]
                      for r in runs if r["final_train_cost"] is not None], dtype=np.float64)
    plt.figure(figsize=(9, 5))
    if len(finals) > 0:
        plt.hist(finals, bins=min(
            30, max(5, int(math.sqrt(len(finals))))), edgecolor="k", linewidth=0.5)
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

    # Also save a ranked text report
    ranked = sorted(
        [(r["tag"], r["final_train_cost"])
         for r in runs if r["final_train_cost"] is not None],
        key=lambda x: x[1]
    )
    report_path = os.path.join(OUTDIR, "iGPK_final_train_cost_ranked.txt")
    with open(report_path, "w") as f:
        f.write("tag\tfinal_train_cost\n")
        for tag, fc in ranked:
            f.write(f"{tag}\t{fc:.6e}\n")

    print(f"\nSaved:")
    print(f"  - {os.path.join(OUTDIR, 'iGPK_init_sweep_runs.pt')}")
    print(f"  - {os.path.join(OUTDIR, 'iGPK_cost_histories_log.png')}")
    print(f"  - {os.path.join(OUTDIR, 'iGPK_final_train_cost_hist.png')}")
    print(f"  - {report_path}")
