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

from get_iGPK_fcn import get_iGPK
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
    SYSTEM_NAME = "Inhibited Predator-Prey"  # change as needed
    TRAIN_FRAC = 0.6
    TEST_FRAC = 1 - TRAIN_FRAC
    CLIP = None
    LIFTED_ORDER = 10

    NOISE_TYPES = ["gaussian"]
    INTENSITIES = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]
    SEEDS = [100]

    DEVICE = "cuda:0"

    # ---------------------------
    # Output config
    # ---------------------------
    OUTDIR = "PERTURBATION_ANALYSIS_OUT"
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
            iters_list=list((0, 32, 16, 500)),
            learn_rate=0.04,
            opt_weights=list((1.0, 1.0, 1.0)),
            routine="Z_only",
            train_method="Horizon",
            device=DEVICE,
        )

        A = results["A"]
        C = results["C"]

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
            froA,
            condA,
            spectral_radius,
            stable_frac,
            froC,
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
    _write_csv(
        os.path.join(OUTDIR, f"{SYS}_metrics_summary.csv"),
        [
            "system",
            "noise_type",
            "intensity",
            "seed",
            "froA",
            "condA",
            "spectral_radius_A",
            "stable_fraction_A",
            "froC",
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

        def mean(metric: str) -> List[float]:
            out = []
            for inten in intens_sorted:
                vals = by_intensity[inten].get(metric, [])
                out.append(float(sum(vals) / len(vals))
                           if len(vals) else float("nan"))
            return out

        plots = [
            ("froA", "A Frobenius norm", "||A||_F"),
            ("condA", "A condition number", "cond(A)"),
            ("rhoA", "A spectral radius", "rho(A)"),
            ("stable_frac", "A stable fraction", "fraction(|lambda|<1)"),
            ("froC", "C Frobenius norm", "||C||_F"),
            ("condC", "C condition number", "cond(C)"),
        ]

        for metric, title_mid, ylabel in plots:
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

    print("\nAll outputs written to:", os.path.abspath(OUTDIR))
    if not _HAS_SCIPY:
        print("[NOTE] scipy not found; mode tracking used greedy assignment fallback.")
