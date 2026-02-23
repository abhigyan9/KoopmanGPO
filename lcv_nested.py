"""
Nested-loop Lipschitz constant / gradient-norm verification for iGPK get_cost_simple.

Outer loop (hp_trial): random hyperparameters (fixed across inner loop)
Inner loop (z_trial): random virtual targets Z + retrain + compute metrics

Seeding:
- Hyperparameters: seed_hp = 1 + 2*hp_trial
- Virtual targets: seed_Z  = 1 + hp_trial*(1 + z_trial)

End-of-run:
- Saves CSV + .pt
- Prints summary stats (min, max, mean, median, std) for each metric except 'device'
"""

import os
import math
import csv
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
import GPKoopman as gpk


# ----------------------------
# Helpers
# ----------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sv_stats(A: torch.Tensor, eps: float = 1e-12):
    """
    Returns (sigma_max, sigma_min, cond) for matrix A using singular values.
    Uses eps-clamp on sigma_min to avoid inf when A is rank-deficient.
    """
    s = torch.linalg.svdvals(A)  # sorted descending
    sigma_max = s[0]
    sigma_min = s[-1]
    sigma_min_clamped = torch.clamp(sigma_min, min=eps)
    cond = sigma_max / sigma_min_clamped
    return float(sigma_max.detach().cpu()), float(sigma_min.detach().cpu()), float(cond.detach().cpu())


def summarize_results(results: List[Dict[str, Any]], exclude_keys=("device",)):
    """
    Print min, max, mean, median, std for each numeric key (excluding exclude_keys).
    """
    if not results:
        print("No results to summarize.")
        return

    keys = [k for k in results[0].keys() if k not in exclude_keys]

    # Collect numeric arrays per key
    stats = {}
    for k in keys:
        vals = []
        for r in results:
            v = r.get(k, None)
            if isinstance(v, (int, float)) and math.isfinite(v):
                vals.append(float(v))
        if len(vals) == 0:
            continue
        t = torch.tensor(vals, dtype=torch.float64)
        stats[k] = {
            "min": float(torch.min(t).item()),
            "max": float(torch.max(t).item()),
            "mean": float(torch.mean(t).item()),
            "median": float(torch.median(t).item()),
            "std": float(torch.std(t, unbiased=False).item()),
            "count": int(t.numel()),
        }

    # Pretty print
    print("\nSummary statistics across all trials:")
    for k in sorted(stats.keys()):
        s = stats[k]
        print(
            f"- {k:24s} | n={s['count']:4d} | "
            f"min={s['min']:.4e}  max={s['max']:.4e}  "
            f"mean={s['mean']:.4e}  median={s['median']:.4e}  std={s['std']:.4e}"
        )


def _fmt2e(x: float) -> str:
    return f"{x:.2e}"


def compute_summary_stats(results: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    vals = [float(r[key]) for r in results if key in r and isinstance(
        r[key], (int, float)) and math.isfinite(r[key])]
    if len(vals) == 0:
        return {"min": float("nan"), "median": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    t = torch.tensor(vals, dtype=torch.float64)
    return {
        "min": float(torch.min(t).item()),
        "median": float(torch.median(t).item()),
        "max": float(torch.max(t).item()),
        "mean": float(torch.mean(t).item()),
        "std": float(torch.std(t, unbiased=False).item()),
    }


def write_latex_summary_table_txt(results: List[Dict[str, Any]], out_txt_path: str):
    """
    Writes a LaTeX tabular to a .txt file with columns:
    Quantity | Expression | Min | Median | Max | Mean | S.D.
    All numeric entries are formatted in scientific notation with 2 decimals.
    """
    rows = [
        # (Quantity, Expression, result_key_for_stats)
        ("Cost", r"$\mathcal{L}$", "cost"),
        (r"Frobenius norm of the calculated gradient",
         r"$\|\nabla_{Z}\mathcal{L}\|_{\mathrm{F}}$", "gradZ_frob"),
        (r"Calculated analytical bound (Lipschitz constant)",
         r"$L_{\mathcal{Z}}$", "bound"),

        (r"Condition number of $\bar{\Phi}\bar{\Phi}^{\top}$",
         r"$\kappa(\bar{\Phi}\bar{\Phi}^{\top})=\kappa(MM^{\top})$", "MMt_cond"),

        (r"Condition number of $\bar{\Phi}(X)$",
         r"$\kappa(\bar{\Phi}(X))=\kappa(M)$", "phi_cond"),

        (r"Condition number of $\bar{\Phi}(X^+)$",
         r"$\kappa(\bar{\Phi}(X^{+}))=\kappa(M^{+})$", "phi_plus_cond"),

        (r"$\gamma_1=\|\bar{\Phi}(X)\|_{\mathrm{F}}$",
         r"$\gamma_{1}$", "gamma1"),

        (r"$\gamma_2=\|\bar{\Phi}(X^+)\|_{\mathrm{F}}$",
         r"$\gamma_{2}$", "gamma2"),
    ]

    # Precompute stats
    stats_map = {k: compute_summary_stats(results, k) for _, _, k in rows}

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{|p{4.0cm}|p{4.0cm}|c|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Quantity} & \textbf{Expression} & \textbf{Min} & \textbf{Median} & \textbf{Max} & \textbf{Mean} & \textbf{S.D.} \\")
    lines.append(r"\hline\hline")

    for qty, expr, key in rows:
        s = stats_map[key]
        lines.append(
            f"{qty} & {expr} & "
            f"{_fmt2e(s['min'])} & {_fmt2e(s['median'])} & {_fmt2e(s['max'])} & {_fmt2e(s['mean'])} & {_fmt2e(s['std'])} \\\\"
        )
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(out_txt_path, "w") as f:
        f.write("\n".join(lines))


# ----------------------------
# Cost function
# ----------------------------
def get_cost_simple(Z, X, Xplus, ObsManager, lambda1=1.0, lambda2=1.0):
    """
    Z:     (nTrain, p) virtual targets on Xtrain
    X:     (n, nTrain*N) query stack
    Xplus: (n, nTrain*N) shifted query stack
    """
    p = Z.shape[1]
    ns_query = X.shape[1]

    M = torch.empty((p, ns_query), device=X.device, dtype=X.dtype)
    Mplus = torch.empty((p, ns_query), device=X.device, dtype=X.dtype)

    for i in range(p):
        mean_i, _ = ObsManager.observables[i].forward(X, Z[:, i])
        M[i, :] = mean_i.reshape(-1)

        mean_pi, _ = ObsManager.observables[i].forward(Xplus, Z[:, i])
        Mplus[i, :] = mean_pi.reshape(-1)

    # Pseudoinverse of M
    try:
        L = torch.linalg.cholesky(M @ M.mT)
        M_pinv = torch.cholesky_solve(M.mT, L)  # (ns_query x p)
    except RuntimeError:
        M_pinv = torch.linalg.pinv(M)

    M_pinvM = M_pinv @ M  # (ns_query x ns_query)

    cost1 = torch.linalg.matrix_norm(Mplus - (Mplus @ M_pinvM))
    cost2 = torch.linalg.matrix_norm(X - (X @ M_pinvM))

    return (lambda1 * cost1) + (lambda2 * cost2)


# ----------------------------
# Analytical bound
# ----------------------------
def grad_norm_upper_bound(
    n_z: int,
    n_TN: int,
    gamma1: float,
    gamma2: float,
    eps: float,
    G_frob: float,
    Gplus_frob: float,
    X_frob: float,
) -> float:
    sqrt_nTN = math.sqrt(n_TN)
    sqrt_nz = math.sqrt(n_z)

    A = sqrt_nTN + (gamma1 ** 2) * (sqrt_nz / eps)
    pref = (2.0 * A) / (n_z * n_TN)

    term1 = (gamma2 ** 2) * A * Gplus_frob
    term2 = (2.0 * sqrt_nz * gamma1 / eps) * (1.0 + (sqrt_nz * (gamma1 ** 2) / eps)) \
        * ((gamma2 ** 2) + (X_frob ** 2)) * G_frob

    return pref * (term1 + term2)


# ----------------------------
# Config
# ----------------------------
@dataclass
class ExperimentConfig:
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    system_name: str = "OT_16steps"
    trainFrac: float = 1.0
    testFrac: float = 0.0
    clip: Optional[int] = None
    noise_type: str = "gaussian"
    noise_level: float = 0    # 0.1 = 10% noise

    p: int = 25
    gpo_noise: float = 1e-4
    m: int = 500

    num_trials_hp: int = 50
    num_trials_Z: int = 7

    # bound stabilizers
    eps_floor: float = 1e-8
    tikhonov_eps: float = 0.0

    lambda1: float = 1.0
    lambda2: float = 1.0

    out_dir: str = "Figures/igpk_grad_sweep_out"
    out_name: str = f"grad_norm_nested_{system_name}_HP{num_trials_hp}_Z{num_trials_Z}_{noise_type}{noise_level}"


# ----------------------------
# Main
# ----------------------------
def main():
    cfg = ExperimentConfig()
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Load and normalize data
    SimData_raw, ts, num_traj, N_total, nTrain, nTest = gpk.utilities.load_SimData(
        cfg.system_name, cfg.trainFrac, cfg.testFrac, clip=cfg.clip
    )
    SimData_clean, mu_vec, std_vec = gpk.utilities.normalize_data(
        SimData_raw, nTrain, N_total)

    SimData = gpk.add_noise(SimData_clean, noise_type=cfg.noise_type,
                            intensity=cfg.noise_level)

    # Build X and Xplus (n x (nTrain*N))
    n = SimData.shape[1]
    # number of transitions per traj used in X (requires N+1 points in SimData)
    N = N_total
    X = torch.cat([SimData[j, :, 0:N] for j in range(nTrain)], dim=1)
    Xplus = torch.cat([SimData[j, :, 1:N+1] for j in range(nTrain)], dim=1)

    X = X.to(cfg.device, dtype=cfg.dtype)
    Xplus = Xplus.to(cfg.device, dtype=cfg.dtype)

    # Xtrain: one point per trajectory, as in your example
    Xtrain = torch.cat([X[:, j*N: j*N + 1]
                       for j in range(nTrain)], dim=1)  # n x nTrain

    ns_query = X.shape[1]   # nTrain*N
    n_TN = ns_query
    n_z = cfg.p

    # Precompute ||X||_F for bound
    X_frob = float(torch.linalg.norm(X, ord="fro").detach().cpu())

    # Build ObsManager + observables once (hyperparams will be randomized each outer loop)
    ObsManager = gpk.GPObservablesManager()
    for i in range(cfg.p):
        ObsManager.add_observable(
            index=i,
            d=n,
            ns=nTrain,  # training set size = nTrain (columns of Xtrain)
            kernel_types=["Gaussian"],
            combination="sum",
            noise=cfg.gpo_noise,
            m=cfg.m,
            device=cfg.device,
        )

    results: List[Dict[str, Any]] = []
    t0 = time.time()

    # ----------------------------
    # Nested loops
    # ----------------------------
    for hp_trial in range(cfg.num_trials_hp):
        seed_hp = 1 + 2 * hp_trial
        set_seed(seed_hp)

        # Randomize hyperparameters ONCE per outer loop
        ObsManager.set_random_hyperparameters(scale=[1.0, 1.0, None])

        for z_trial in range(cfg.num_trials_Z):
            seed_Z = (11 + 5*hp_trial) * (7 + 3*z_trial)
            set_seed(seed_Z)

            # Random init Z for this inner loop
            Z = torch.nn.Parameter(torch.rand(
                (nTrain, cfg.p), device=cfg.device, dtype=cfg.dtype))

            # Train each observable on (Xtrain, Z[:,i]) with current hyperparams
            for i in range(cfg.p):
                ObsManager.train_observable(i, Xtrain, Z[:, i])

            # Cost + gradient wrt Z
            if Z.grad is not None:
                Z.grad.zero_()

            cost = get_cost_simple(
                Z, X, Xplus, ObsManager, lambda1=cfg.lambda1, lambda2=cfg.lambda2)
            cost.backward()

            gradZ_frob = float(torch.linalg.norm(
                Z.grad, ord="fro").detach().cpu())
            cost_val = float(cost.detach().cpu())

            # Bound + singular value stats
            with torch.no_grad():
                # Lifted means
                M = torch.empty((cfg.p, ns_query),
                                device=cfg.device, dtype=cfg.dtype)
                Mplus = torch.empty((cfg.p, ns_query),
                                    device=cfg.device, dtype=cfg.dtype)
                for i in range(cfg.p):
                    mean_i, _ = ObsManager.observables[i].forward(
                        X, Z[:, i].detach())
                    M[i, :] = mean_i.reshape(-1)
                    mean_pi, _ = ObsManager.observables[i].forward(
                        Xplus, Z[:, i].detach())
                    Mplus[i, :] = mean_pi.reshape(-1)

                gamma1 = float(torch.linalg.norm(M, ord="fro").cpu())
                gamma2 = float(torch.linalg.norm(Mplus, ord="fro").cpu())

                MMt = M @ M.mT
                if cfg.tikhonov_eps > 0.0:
                    MMt = MMt + cfg.tikhonov_eps * \
                        torch.eye(MMt.shape[0],
                                  device=MMt.device, dtype=MMt.dtype)
                eigs = torch.linalg.eigvalsh(MMt)
                eps = float(torch.clamp(eigs.min(), min=cfg.eps_floor).cpu())

                # G and Gplus (stacked across observables)
                G_blocks = [ObsManager.observables[i].forward_G(
                    X) for i in range(cfg.p)]
                Gp_blocks = [ObsManager.observables[i].forward_G(
                    Xplus) for i in range(cfg.p)]
                G_big = torch.vstack(G_blocks)    # (p*ns_query, nTrain)
                Gp_big = torch.vstack(Gp_blocks)  # (p*ns_query, nTrain)

                G_frob = float(torch.linalg.norm(G_big, ord="fro").cpu())
                Gp_frob = float(torch.linalg.norm(Gp_big, ord="fro").cpu())

                bound = grad_norm_upper_bound(
                    n_z=n_z,
                    n_TN=n_TN,
                    gamma1=gamma1,
                    gamma2=gamma2,
                    eps=eps,
                    G_frob=G_frob,
                    Gplus_frob=Gp_frob,
                    X_frob=X_frob,
                )

                ratio = gradZ_frob / bound if bound > 0 else float("inf")

                # Singular-value stats
                phi_sigma_max, phi_sigma_min, phi_cond = sv_stats(
                    M, eps=cfg.eps_floor)
                phiP_sigma_max, phiP_sigma_min, phiP_cond = sv_stats(
                    Mplus, eps=cfg.eps_floor)
                G_sigma_max, G_sigma_min, G_cond = sv_stats(
                    G_big, eps=cfg.eps_floor)
                Gp_sigma_max, Gp_sigma_min, Gp_cond = sv_stats(
                    Gp_big, eps=cfg.eps_floor)
                MMt_sigma_max, MMt_sigma_min, MMt_cond = sv_stats(
                    MMt, eps=cfg.eps_floor)

            results.append({
                "hp_trial": hp_trial,
                "z_trial": z_trial,
                "seed_hp": seed_hp,
                "seed_Z": seed_Z,

                "cost": cost_val,
                "gradZ_frob": gradZ_frob,
                "bound": float(bound),
                "ratio_grad_over_bound": float(ratio),

                "gamma1": gamma1,
                "gamma2": gamma2,
                "epsilon": eps,

                "G_frob": G_frob,
                "Gplus_frob": Gp_frob,
                "X_frob": X_frob,

                "phi_sigma_max": phi_sigma_max,
                "phi_sigma_min": phi_sigma_min,
                "phi_cond": phi_cond,

                "phi_plus_sigma_max": phiP_sigma_max,
                "phi_plus_sigma_min": phiP_sigma_min,
                "phi_plus_cond": phiP_cond,

                "G_sigma_max": G_sigma_max,
                "G_sigma_min": G_sigma_min,
                "G_cond": G_cond,

                "Gplus_sigma_max": Gp_sigma_max,
                "Gplus_sigma_min": Gp_sigma_min,
                "Gplus_cond": Gp_cond,

                "MMt_sigma_max": MMt_sigma_max,
                "MMt_sigma_min": MMt_sigma_min,
                "MMt_cond": MMt_cond,

                "nTrain": nTrain,
                "N": N,
                "ns_query": ns_query,
                "device": cfg.device,
            })

            # Avoid graph accumulation
            del Z, cost

    # ----------------------------
    # Save outputs
    # ----------------------------
    base = os.path.join(cfg.out_dir, cfg.out_name)
    csv_path = base + ".csv"
    pt_path = base + ".pt"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    torch.save({"config": cfg.__dict__, "results": results}, pt_path)

    txt_path = base + "_latex_summary.txt"
    write_latex_summary_table_txt(results, txt_path)

    print(f"\nSaved:\n- {csv_path}\n- {pt_path}\n- {txt_path}")
    print(f"Elapsed: {time.time() - t0:.2f} s")

    # Print summaries at the end (excluding device)
    summarize_results(results, exclude_keys=("device",))

    print(f"\nSaved:\n- {csv_path}\n- {pt_path}")
    print(f"Elapsed: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
