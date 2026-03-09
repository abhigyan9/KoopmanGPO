"""perturbation_analysis.py

Extends iGPK perturbation analysis across noise levels.

Outputs (all under PERTURBATION_ANALYSIS_OUT/):
  - <SYSTEM>_metrics_summary.csv
  - <SYSTEM>_A_eigenvalues_long.csv
  - <SYSTEM>_C_singular_values_long.csv
  - <SYSTEM>_mode_tracking_<noise>_seed<seed>.csv
  - <SYSTEM>_<noise>_<metric>_vs_intensity.png

Mode tracking:
  For each (noise_type, seed), eigenvalues of A are matched across intensities
  by solving a minimum-cost assignment (Hungarian) between consecutive
  intensity levels, using distance in the complex plane.

Notes:
  - Filenames start with a sanitized system name.
  - Intensity is included in *rows* for long CSVs; the file naming convention
    is applied to per-(noise,seed) tracking outputs and plots.
"""

from get_iGPK_new import get_iGPK
import matplotlib.pyplot as plt
import os
import csv
import itertools
from typing import Dict, Any, List, Tuple, Optional

import torch
import GPKoopman as gpk

# Optional plotting (safe for headless)
import matplotlib
matplotlib.use("Agg")


# --- optional Hungarian assignment
try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    _HAS_SCIPY = True
except Exception:
    linear_sum_assignment = None
    _HAS_SCIPY = False


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _slug(s: str) -> str:
    # keep letters, digits, dash, underscore; convert spaces to underscores
    out = []
    for ch in s.strip().replace(" ", "_"):
        if ch.isalnum() or ch in ["_", "-"]:
            out.append(ch)
    return "".join(out) if out else "SYSTEM"


def _to_cpu(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    return x.detach().to("cpu")


def _sorted_eigs(A: torch.Tensor) -> torch.Tensor:
    """Eigenvalues sorted by descending magnitude."""
    eigs = torch.linalg.eigvals(A)
    idx = torch.argsort(torch.abs(eigs), descending=True)
    return eigs[idx]


def _sorted_svdvals(M: torch.Tensor) -> torch.Tensor:
    sv = torch.linalg.svdvals(M)
    return torch.sort(sv, descending=True).values


def _safe_cond_from_svdvals(sv: torch.Tensor, eps: float = 1e-12) -> float:
    sv = _to_cpu(sv).double()
    if sv.numel() == 0:
        return float("nan")
    smax = float(sv.max().item())
    smin = float(sv.min().item())
    if smin < eps:
        return float("inf")
    return smax / smin


def _write_csv(path: str, header: List[str], rows: List[List[Any]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _plot_vs_intensity(out_path: str, title: str, x: List[float], y: List[float], ylabel: str) -> None:
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("Noise intensity")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _boxplot_by_intensity(
    out_path: str,
    title: str,
    intensities: List[float],
    data_by_intensity: Dict[float, List[float]],
    ylabel: str,
) -> None:
    """Single-axis box plot where each box corresponds to an intensity."""
    data = [data_by_intensity.get(float(i), []) for i in intensities]

    plt.figure(figsize=(10, 4.5))
    plt.boxplot(
        data,
        tick_labels=[f"{i:g}" for i in intensities],
        showfliers=False,
        whis=(5, 95),
    )
    plt.xlabel("Noise intensity")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


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
            nv = _to_cpu(nv).reshape(-1)
            noise_vals.extend([float(x.item()) for x in nv])
        else:
            noise_vals.append(float(nv))

    if len(noise_vals) == 0:
        return noise_vals, float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    t = torch.tensor(noise_vals, dtype=torch.float64)
    mean = float(t.mean().item())
    median = float(t.median().item())
    std = float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0
    minv = float(t.min().item())
    maxv = float(t.max().item())
    return noise_vals, mean, median, std, minv, maxv


def _match_eigs(prev: List[complex], curr: List[complex]) -> List[int]:
    """Return a permutation p such that curr[p[i]] matches prev[i]."""
    n = len(prev)
    if len(curr) != n:
        raise ValueError(
            "Eigenvalue count changed across intensities; cannot track modes reliably.")

    # Cost matrix: |prev_i - curr_j|
    # Build as float list-of-lists to avoid torch<->numpy ceremony.
    cost = [[abs(prev[i] - curr[j]) for j in range(n)] for i in range(n)]

    if _HAS_SCIPY and linear_sum_assignment is not None:
        import numpy as np
        C = np.asarray(cost, dtype=float)
        r, c = linear_sum_assignment(C)
        # r is [0..n-1] in some order; ensure we return mapping for each i
        perm = [0] * n
        for ri, ci in zip(r.tolist(), c.tolist()):
            perm[ri] = ci
        return perm

    # Fallback: greedy (not optimal, but better than nothing)
    remaining = set(range(n))
    perm = [-1] * n
    for i in range(n):
        j_best = min(remaining, key=lambda j: cost[i][j])
        perm[i] = j_best
        remaining.remove(j_best)
    return perm


def _track_modes(eigs_by_intensity: Dict[float, List[complex]], intensities_sorted: List[float]) -> Tuple[List[int], Dict[int, List[complex]]]:
    """Track eigenvalues across intensities.

    Returns:
      mode_ids: [0..n-1]
      tracked: dict mode_id -> list of eigenvalues aligned with intensities_sorted
    """
    if not intensities_sorted:
        return [], {}

    base = eigs_by_intensity[intensities_sorted[0]]
    n = len(base)
    tracked: Dict[int, List[complex]] = {k: [base[k]] for k in range(n)}

    prev_ordered = base
    for inten in intensities_sorted[1:]:
        curr = eigs_by_intensity[inten]
        perm = _match_eigs(prev_ordered, curr)
        curr_ordered = [curr[perm[i]] for i in range(n)]
        for k in range(n):
            tracked[k].append(curr_ordered[k])
        prev_ordered = curr_ordered

    return list(range(n)), tracked


if __name__ == "__main__":
    # ---------------------------
    # User config
    # ---------------------------
    SYSTEM_NAME = "Cart_data"  # change as needed
    TRAIN_FRAC = 0.6
    TEST_FRAC = 1 - TRAIN_FRAC
    CLIP = None
    LIFTED_ORDER = 20

    NOISE_TYPES = ["gaussian"]
    INTENSITIES = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
    SEEDS = [100]

    DEVICE = "cuda:0"

    # ---------------------------
    # Output config
    # ---------------------------
    OUTDIR = f"Figures/Perturbation_Analysis/{SYSTEM_NAME}"
    _ensure_dir(OUTDIR)

    SYS = _slug(SYSTEM_NAME)

    # ---------------------------
    # Load and normalize data
    # ---------------------------
    SimData_raw, ts, num_traj, N, nTrain, nTest = gpk.load_SimData(
        SYSTEM_NAME, TRAIN_FRAC, TEST_FRAC, clip=CLIP
    )
    SimData_clean, mu_vec, std_vec = gpk.normalize_data(SimData_raw, nTrain, N)

    # ---------------------------
    # Accumulators
    # ---------------------------
    metrics_rows: List[List[Any]] = []
    eig_long_rows: List[List[Any]] = []
    svC_long_rows: List[List[Any]] = []

    # For mode tracking
    # eig_db[(noise_type, seed)][intensity] = list(complex eigs sorted by |.| desc
    eig_db: Dict[Tuple[str, int], Dict[float, List[complex]]] = {}

    # For plots: scalar metrics aggregated across seeds
    # plot_db[noise_type][intensity][metric] -> list over seeds
    plot_db: Dict[str, Dict[float, Dict[str, List[float]]]] = {}

    # NRMSE distributions for boxplots (flattened over trajectories & states)
    # nrmse_db[noise_type][intensity]["train"|"test"] -> list of floats
    nrmse_db: Dict[str, Dict[float, Dict[str, List[float]]]] = {}

    # Cost histories
    # cost_db[noise_type][intensity] -> list of cost history lists (one per seed)
    cost_db: Dict[str, Dict[float, List[List[float]]]] = {}

    # Learned GP noise values (flattened across observables)
    # gpnoise_db[noise_type][intensity] -> list of learned noise scalars (across seeds)
    gpnoise_db: Dict[str, Dict[float, List[float]]] = {}

    for noise_type, intensity, seed in itertools.product(NOISE_TYPES, INTENSITIES, SEEDS):
        # Avoid duplicate zero-noise runs across noise types
        if intensity == 0.0 and noise_type != "gaussian":
            continue

        print(
            f"\n=== {SYSTEM_NAME} | {noise_type} | intensity={intensity:.3f} | seed={seed} ===")

        SimData = gpk.add_noise(
            SimData_clean, noise_type=noise_type, intensity=float(intensity), seed=int(seed)
        )

        results = get_iGPK(
            SimData=SimData,
            nTrain=nTrain,
            nTest=nTest,
            lifting_order=LIFTED_ORDER,
            iters_list=list((0, 32, 16, 1000)),
            learn_rate=0.01,
            opt_weights=list((1.0, 1.0, 1.0)),
            routine="Z_only",
            train_method="Horizon",
            device=DEVICE,
        )

        A = results["A"]
        C = results["C"]

        # Cost history: 1D tensor (n_iters,)
        cost_hist_t = results.get("history", {}).get("cost", None)
        final_cost = float("nan")
        cost_hist = None
        if isinstance(cost_hist_t, torch.Tensor) and cost_hist_t.numel() > 0:
            cost_hist = _to_cpu(cost_hist_t).reshape(-1).double()
            final_cost = float(cost_hist[-1].item())
            cost_db.setdefault(noise_type, {}).setdefault(
                float(intensity), []).append([float(x) for x in cost_hist.tolist()])

        # Learned GP noise hyperparameters (per observable)
        obs_manager = results.get("ObsManager", None)
        gp_noise_vals, gp_noise_mean, gp_noise_median, gp_noise_std, gp_noise_min, gp_noise_max = _extract_gp_noise_stats(
            obs_manager)
        if len(gp_noise_vals) > 0:
            gpnoise_db.setdefault(noise_type, {}).setdefault(
                float(intensity), []).extend(gp_noise_vals)

        # NRMSE: (nTraj, nState). We'll flatten to a single distribution.
        TrainNRMSE = results.get("Train", {}).get("NRMSE", None)
        TestNRMSE = results.get("Test", {}).get("NRMSE", None)
        if isinstance(TrainNRMSE, torch.Tensor):
            train_vals = _to_cpu(TrainNRMSE).reshape(-1).double()
            nrmse_db.setdefault(noise_type, {}).setdefault(float(intensity), {}).setdefault(
                "train", []
            ).extend([float(v.item()) for v in train_vals])
        if isinstance(TestNRMSE, torch.Tensor):
            test_vals = _to_cpu(TestNRMSE).reshape(-1).double()
            nrmse_db.setdefault(noise_type, {}).setdefault(float(intensity), {}).setdefault(
                "test", []
            ).extend([float(v.item()) for v in test_vals])

        A_cpu = _to_cpu(A)
        C_cpu = _to_cpu(C)

        eigA_t = _sorted_eigs(A_cpu)
        svA_t = _sorted_svdvals(A_cpu)
        svC_t = _sorted_svdvals(C_cpu)

        froA = float(torch.linalg.norm(A_cpu, ord="fro").item())
        froC = float(torch.linalg.norm(C_cpu, ord="fro").item())
        condA = _safe_cond_from_svdvals(svA_t)
        condC = _safe_cond_from_svdvals(svC_t)

        spectral_radius = float(
            torch.max(torch.abs(eigA_t)).item()) if eigA_t.numel() else float("nan")
        stable_frac = float((torch.abs(eigA_t) < 1.0).double(
        ).mean().item()) if eigA_t.numel() else float("nan")

        # Save leading values for quick scanning
        top_k = 10
        top_eig_abs = [float(v.item()) for v in torch.abs(eigA_t[:top_k])]
        top_svC = [float(v.item()) for v in svC_t[:top_k]]

        metrics_rows.append([
            SYSTEM_NAME,
            noise_type,
            float(intensity),
            int(seed),
            final_cost,
            gp_noise_mean,
            gp_noise_median,
            gp_noise_std,
            gp_noise_min,
            gp_noise_max,
            len(gp_noise_vals),
            froA,
            float("nan"),  # rel_change_froA (filled after baseline computed)
            condA,
            spectral_radius,
            stable_frac,
            froC,
            float("nan"),  # rel_change_froC (filled after baseline computed)
            condC,
            int(eigA_t.numel()),
            int(svC_t.numel()),
            " ".join([f"{x:.6e}" for x in top_eig_abs]),
            " ".join([f"{x:.6e}" for x in top_svC]),
        ])

        # Long-form eigenvalues
        eigA_list = [complex(z.real, z.imag) for z in eigA_t.tolist()]
        for i, lam in enumerate(eigA_list):
            eig_long_rows.append([
                SYSTEM_NAME,
                noise_type,
                float(intensity),
                int(seed),
                int(i),
                float(lam.real),
                float(lam.imag),
                float(abs(lam)),
            ])

        # Long-form singular values of C
        for i, s in enumerate(svC_t.tolist()):
            svC_long_rows.append([
                SYSTEM_NAME,
                noise_type,
                float(intensity),
                int(seed),
                int(i),
                float(s),
            ])

        # Store for mode tracking
        key = (noise_type, int(seed))
        eig_db.setdefault(key, {})[float(intensity)] = eigA_list

        # Store for plots
        plot_db.setdefault(noise_type, {}).setdefault(
            float(intensity), {}).setdefault("froA", []).append(froA)
        plot_db[noise_type][float(intensity)].setdefault(
            "condA", []).append(condA)
        plot_db[noise_type][float(intensity)].setdefault(
            "rhoA", []).append(spectral_radius)
        plot_db[noise_type][float(intensity)].setdefault(
            "stable_frac", []).append(stable_frac)
        plot_db[noise_type][float(intensity)].setdefault(
            "froC", []).append(froC)
        plot_db[noise_type][float(intensity)].setdefault(
            "condC", []).append(condC)

    # ---------------------------
    # Write core CSV outputs
    # ---------------------------

    # Compute baselines (mean over seeds) at zero noise, per noise type.
    froA0: Dict[str, float] = {}
    froC0: Dict[str, float] = {}
    for noise_type, by_intensity in plot_db.items():
        if 0.0 in by_intensity:
            a0 = by_intensity[0.0].get("froA", [])
            c0 = by_intensity[0.0].get("froC", [])
            if len(a0):
                froA0[noise_type] = float(sum(a0) / len(a0))
            if len(c0):
                froC0[noise_type] = float(sum(c0) / len(c0))

    # Fill relative-change columns in metrics_rows.
    for r in metrics_rows:
        # r layout: system, noise_type, intensity, seed, froA, relA, condA, rhoA, stable, froC, relC, condC, ...
        ntype = str(r[1])
        froA = float(r[4])
        froC = float(r[9])
        if ntype in froA0 and froA0[ntype] not in [0.0, float("nan")]:
            r[5] = (froA - froA0[ntype]) / froA0[ntype]
        if ntype in froC0 and froC0[ntype] not in [0.0, float("nan")]:
            r[10] = (froC - froC0[ntype]) / froC0[ntype]

    _write_csv(
        os.path.join(OUTDIR, f"{SYS}_metrics_summary.csv"),
        [
            "system",
            "noise_type",
            "intensity",
            "seed",
            "final_cost",
            "gp_noise_mean",
            "gp_noise_median",
            "gp_noise_std",
            "gp_noise_min",
            "gp_noise_max",
            "gp_noise_n",
            "froA",
            "rel_change_froA",
            "condA",
            "spectral_radius_A",
            "stable_fraction_A",
            "froC",
            "rel_change_froC",
            "condC",
            "n_eigs_A",
            "n_svals_C",
            "top10_abs_eigs_A",
            "top10_svals_C",
        ],
        metrics_rows,
    )

    _write_csv(
        os.path.join(OUTDIR, f"{SYS}_A_eigenvalues_long.csv"),
        ["system", "noise_type", "intensity", "seed",
            "eig_index_sorted", "real", "imag", "abs"],
        eig_long_rows,
    )

    _write_csv(
        os.path.join(OUTDIR, f"{SYS}_C_singular_values_long.csv"),
        ["system", "noise_type", "intensity", "seed", "sv_index_sorted", "sv"],
        svC_long_rows,
    )

    # ---------------------------
    # Mode tracking outputs + optional plots
    # ---------------------------
    for (noise_type, seed), by_intensity in eig_db.items():
        intens_sorted = sorted(by_intensity.keys())
        mode_ids, tracked = _track_modes(by_intensity, intens_sorted)

        # Write mode tracking CSV (long form)
        rows: List[List[Any]] = []
        for mode_id in mode_ids:
            traj = tracked[mode_id]
            for inten, lam in zip(intens_sorted, traj):
                rows.append([
                    SYSTEM_NAME,
                    noise_type,
                    float(inten),
                    int(seed),
                    int(mode_id),
                    float(lam.real),
                    float(lam.imag),
                    float(abs(lam)),
                ])

        track_csv = os.path.join(
            OUTDIR, f"{SYS}_mode_tracking_{noise_type}_seed{seed}.csv")
        _write_csv(
            track_csv,
            ["system", "noise_type", "intensity", "seed",
                "mode_id", "real", "imag", "abs"],
            rows,
        )

        # A couple of lightweight plots: top few modes by initial |lambda|
        # (Plotting all modes can get unreadable fast.)
        try:
            # rank modes by |lambda| at first intensity
            init_abs = [(mode_id, abs(tracked[mode_id][0]))
                        for mode_id in mode_ids]
            init_abs.sort(key=lambda t: t[1], reverse=True)
            plot_modes = [m for m, _ in init_abs[: min(8, len(init_abs))]]

            # abs trajectories
            plt.figure()
            for m in plot_modes:
                y = [abs(z) for z in tracked[m]]
                plt.plot(intens_sorted, y, marker="o", label=f"mode {m}")
            plt.xlabel("Noise intensity")
            plt.ylabel("|lambda|")
            plt.title(
                f"{SYSTEM_NAME} | A eigenvalue magnitudes (tracked)\n{noise_type}, seed={seed}")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(
                OUTDIR, f"{SYS}_{noise_type}_seed{seed}_tracked_eig_abs.png"), dpi=200)
            plt.close()

            # complex plane trajectories
            plt.figure()
            for m in plot_modes:
                xs = [z.real for z in tracked[m]]
                ys = [z.imag for z in tracked[m]]
                plt.plot(xs, ys, marker="o", label=f"mode {m}")
            plt.xlabel("Re(lambda)")
            plt.ylabel("Im(lambda)")
            plt.title(
                f"{SYSTEM_NAME} | A eigenvalue trajectories (tracked)\n{noise_type}, seed={seed}")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(
                OUTDIR, f"{SYS}_{noise_type}_seed{seed}_tracked_eig_complex.png"), dpi=200)
            plt.close()
        except Exception as e:
            print(f"[WARN] Plotting mode-tracking figures failed: {e}")

    # ---------------------------
    # Summary plots vs intensity (mean across seeds)
    # ---------------------------
    for noise_type, by_intensity in plot_db.items():
        intens_sorted = sorted(by_intensity.keys())

        # Baselines for relative change
        a0 = by_intensity.get(0.0, {}).get("froA", [])
        c0 = by_intensity.get(0.0, {}).get("froC", [])
        froA0_nt = float(sum(a0) / len(a0)) if len(a0) else float("nan")
        froC0_nt = float(sum(c0) / len(c0)) if len(c0) else float("nan")

        def mean(metric: str) -> List[float]:
            out = []
            for inten in intens_sorted:
                vals = by_intensity[inten].get(metric, [])
                out.append(float(sum(vals) / len(vals))
                           if len(vals) else float("nan"))
            return out

        plots = [
            ("rel_froA", "Relative change in A Frobenius norm",
             r"(||A||_F-||A||_{F,0})/||A||_{F,0}"),
            ("condA", "A condition number", "cond(A)"),
            ("rhoA", "A spectral radius", "rho(A)"),
            ("stable_frac", "A stable fraction", "fraction(|lambda|<1)"),
            ("rel_froC", "Relative change in C Frobenius norm",
             r"(||C||_F-||C||_{F,0})/||C||_{F,0}"),
            ("condC", "C condition number", "cond(C)"),
        ]

        for metric, title_mid, ylabel in plots:
            if metric == "rel_froA":
                y = [
                    (m - froA0_nt) / froA0_nt if (froA0_nt ==
                                                  froA0_nt and froA0_nt != 0.0) else float("nan")
                    for m in mean("froA")
                ]
            elif metric == "rel_froC":
                y = [
                    (m - froC0_nt) / froC0_nt if (froC0_nt ==
                                                  froC0_nt and froC0_nt != 0.0) else float("nan")
                    for m in mean("froC")
                ]
            else:
                y = mean(metric)
            out_png = os.path.join(
                OUTDIR, f"{SYS}_{noise_type}_{metric}_vs_intensity.png")
            _plot_vs_intensity(
                out_png,
                f"{SYSTEM_NAME} | {noise_type} | {title_mid}",
                intens_sorted,
                y,
                ylabel,
            )

        # NRMSE boxplots (train/test)
        if noise_type in nrmse_db:
            train_by = {i: nrmse_db[noise_type].get(
                i, {}).get("train", []) for i in intens_sorted}
            test_by = {i: nrmse_db[noise_type].get(
                i, {}).get("test", []) for i in intens_sorted}

            _boxplot_by_intensity(
                os.path.join(
                    OUTDIR, f"{SYS}_{noise_type}_Train_NRMSE_box.png"),
                f"{SYSTEM_NAME} | {noise_type} | Train NRMSE vs intensity",
                intens_sorted,
                train_by,
                "NRMSE",
            )
            _boxplot_by_intensity(
                os.path.join(OUTDIR, f"{SYS}_{noise_type}_Test_NRMSE_box.png"),
                f"{SYSTEM_NAME} | {noise_type} | Test NRMSE vs intensity",
                intens_sorted,
                test_by,
                "NRMSE",
            )

    print("\nAll outputs written to:", os.path.abspath(OUTDIR))
    if not _HAS_SCIPY:
        print("[NOTE] scipy not found; mode tracking used greedy assignment fallback.")

# --- Learned GP noise hyperparameter plots ---
try:
    if noise_type in gpnoise_db and len(gpnoise_db[noise_type]) > 0:
        gp_by = {i: gpnoise_db[noise_type].get(i, []) for i in intens_sorted}
        _boxplot_by_intensity(
            os.path.join(OUTDIR, f"{SYS}_{noise_type}_GPNoise_box.png"),
            f"{SYSTEM_NAME} | {noise_type} | Learned GP noise vs intensity",
            intens_sorted,
            gp_by,
            "Learned GP noise",
        )

        means = []
        for i in intens_sorted:
            vals = gp_by.get(i, [])
            means.append(float(torch.tensor(vals, dtype=torch.float64).mean().item()) if len(
                vals) else float("nan"))

        plt.figure()
        plt.plot(intens_sorted, means, marker="o",
                 label="Mean learned GP noise")
        plt.plot(intens_sorted, intens_sorted,
                 linestyle="--", label="Injected noise (y=x)")
        plt.xlabel("Noise intensity")
        plt.ylabel("Noise")
        plt.title(
            f"{SYSTEM_NAME} | {noise_type} | Mean learned GP noise vs injected noise")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                OUTDIR, f"{SYS}_{noise_type}_GPNoise_mean_vs_injected.png"),
            dpi=200,
        )
        plt.close()
except Exception as e:
    print(
        f"[WARN] Plotting learned GP noise figures failed for {noise_type}: {e}")

# --- Cost history plot (log y) across noise intensities ---
try:
    if noise_type in cost_db and len(cost_db[noise_type]) > 0:
        plt.figure()
        for inten in intens_sorted:
            hist_list = cost_db[noise_type].get(float(inten), [])
            if not hist_list:
                continue
            maxlen = max(len(h) for h in hist_list)
            H = torch.full((len(hist_list), maxlen),
                           float("nan"), dtype=torch.float64)
            for r, h in enumerate(hist_list):
                H[r, :len(h)] = torch.tensor(h, dtype=torch.float64)
            mean_hist = torch.nanmean(H, dim=0).numpy()
            plt.plot(range(len(mean_hist)), mean_hist,
                     label=f"intensity={inten:g}")

        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title(
            f"{SYSTEM_NAME} | {noise_type} | Cost history (mean over seeds)")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(fontsize=8, ncols=2)
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTDIR, f"{SYS}_{noise_type}_cost_history_log.png"),
            dpi=200,
        )
        plt.close()
except Exception as e:
    print(f"[WARN] Plotting cost history figure failed for {noise_type}: {e}")
