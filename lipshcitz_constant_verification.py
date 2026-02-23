"""
Lipschitz constant / gradient-norm verification for iGPK get_cost_simple.

This version follows the user's GPObservablesManager workflow:
- Xtrain: n x nTrain (one sample per trajectory, e.g., initial condition)
- Z:      nTrain x p (virtual targets on training inputs)
- X, Xplus: n x (nTrain*N) (query stacks over all trajectories and timesteps)
- ObsManager: gpk.GPObservablesManager with p observables, trained on (Xtrain, Z[:,i])
- Cost uses obs.forward(X, Z[:,i]) and obs.forward(Xplus, Z[:,i])

Outputs:
- Stores ||∇_Z L||_F and analytical upper bound per trial in CSV and .pt

Assumptions:
- ObsManager exposes:
    - add_observable(...)
    - train_observable(i, Xtrain, ytrain)
    - set_random_hyperparameters(scale=[..., ..., ...])
    - observables list with .forward(...) and .forward_G(...)
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
# Cost function (same structure as your get_cost_simple)
# NOTE: Here Z is (nTrain x p), NOT (nTrain*N x p)
# ----------------------------
def get_cost_simple(Z, X, Xplus, ObsManager, nT=1, lambda1=1.0, lambda2=1.0):
    """
    Args:
        Z:        (nTrain, p) virtual targets on Xtrain
        X:        (n, nTrain*N) query stack
        Xplus:    (n, nTrain*N) shifted query stack
        ObsManager: GPObservablesManager with .observables[i].forward(Xq, ytrain)
        nT: number of trajectories (here nT == nTrain if you use all trajectories)
    """
    p = Z.shape[1]
    ns_query = X.shape[1]  # = nTrain*N

    M = torch.empty((p, ns_query), device=X.device, dtype=X.dtype)
    Mplus = torch.empty((p, ns_query), device=X.device, dtype=X.dtype)

    for i in range(p):
        mean_i, _ = ObsManager.observables[i].forward(X, Z[:, i])
        M[i, :] = mean_i.reshape(-1)

        mean_pi, _ = ObsManager.observables[i].forward(Xplus, Z[:, i])
        Mplus[i, :] = mean_pi.reshape(-1)

    # Pseudoinverse of M via Cholesky if possible
    try:
        L = torch.linalg.cholesky(M @ M.mT)
        M_pinv = torch.cholesky_solve(M.mT, L)  # (ns_query x p)
    except RuntimeError:
        M_pinv = torch.linalg.pinv(M)
    # gram_M = M @ M.mT
    # gram_M += ((1e-6) *
    #            torch.eye(n=gram_M.shape[0], dtype=M.dtype, device=M.device))
    # L = torch.linalg.cholesky(gram_M)
    # M_pinv = torch.cholesky_solve(M.mT, L)  # (ns_query x p)

    M_pinvM = M_pinv @ M  # (ns_query x ns_query)

    cost1 = torch.linalg.matrix_norm(Mplus - (Mplus @ M_pinvM))
    cost2 = torch.linalg.matrix_norm(X - (X @ M_pinvM))

    return (lambda1 * cost1) + (lambda2 * cost2)


# ----------------------------
# Analytical bound (from your last equation)
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
    """
    ||∇_Z L||_F upper bound (last equation in iGPK_Theoretical_Analysis.txt)

    Note:
      - n_z: number of observables p
      - n_TN: number of query points (nTrain*N)
      - gamma1 = ||M||_F, gamma2 = ||Mplus||_F
      - eps = lambda_min(M M^T) clamped
      - G_frob = ||G||_F, Gplus_frob = ||G^+||_F (stacked across observables)
    """
    sqrt_nTN = math.sqrt(n_TN)
    sqrt_nz = math.sqrt(n_z)

    A = sqrt_nTN + (gamma1 ** 2) * (sqrt_nz / eps)
    pref = (2.0 * A) / (n_z * n_TN)

    term1 = (gamma2 ** 2) * A * Gplus_frob
    term2 = (2.0 * sqrt_nz * gamma1 / eps) * (1.0 + (sqrt_nz * (gamma1 ** 2) / eps)) \
        * ((gamma2 ** 2) + (X_frob ** 2)) * G_frob

    return pref * (term1 + term2)


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


# ----------------------------
# Config
# ----------------------------
@dataclass
class ExperimentConfig:
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    system_name: str = "Inhibited Predator-Prey"
    trainFrac: float = 1.0
    testFrac: float = 0.0
    clip: Optional[int] = None

    p: int = 10
    noise: float = 1e-4
    m: int = 500

    num_trials: int = 25

    # bound stabilizers
    eps_floor: float = 1e-8
    tikhonov_eps: float = 0

    lambda1: float = 1.0
    lambda2: float = 1.0

    out_dir: str = "igpk_grad_sweep_out"
    out_name: str = "grad_norm_sweep_manager"


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
    SimData, mu_vec, std_vec = gpk.utilities.normalize_data(
        SimData_raw, nTrain, N_total)

    # Build X and Xplus as in your modified script:
    # SimData[j, :, :] has length N_total+1 time points (typical in your utilities)
    # X uses first N_total points, Xplus uses last N_total points.
    n = SimData.shape[1]
    N = N_total  # number of transitions per traj
    X = torch.cat([SimData[j, :, 0:N]
                  for j in range(nTrain)], dim=1)      # n x (nTrain*N)
    Xplus = torch.cat([SimData[j, :, 1:N+1]
                      for j in range(nTrain)], dim=1)  # n x (nTrain*N)

    X = X.to(cfg.device, dtype=cfg.dtype)
    Xplus = Xplus.to(cfg.device, dtype=cfg.dtype)

    # Build Xtrain exactly like your example (one point per trajectory):
    # X is already concatenated by trajectory blocks of length N
    Xtrain = torch.cat([X[:, j*N: j*N + 1]
                       for j in range(nTrain)], dim=1)  # n x nTrain

    ns_query = X.shape[1]          # nTrain*N
    n_TN = ns_query
    n_z = cfg.p

    # Precompute ||X||_F for bound
    X_frob = float(torch.linalg.norm(X, ord="fro").detach().cpu())

    # Build ObsManager + observables (no fallback)
    ObsManager = gpk.GPObservablesManager()
    for i in range(cfg.p):
        ObsManager.add_observable(
            index=i,
            d=n,
            # IMPORTANT: training set size = nTrain (columns of Xtrain)
            ns=nTrain,
            kernel_types=["Gaussian"],
            combination="sum",
            noise=cfg.noise,
            m=cfg.m,
            device=cfg.device,
        )

    results: List[Dict[str, Any]] = []
    t0 = time.time()

    for trial in range(cfg.num_trials):
        # Random init Z (virtual targets on Xtrain)
        torch.manual_seed(trial+5)
        Z = torch.nn.Parameter(5*torch.rand(
            (nTrain, cfg.p), device=cfg.device, dtype=cfg.dtype))

        # Randomize hyperparameters (your API)
        ObsManager.set_random_hyperparameters(
            seed=42+trial**2, scale=[1.0, 1.0, None])

        # Train each observable on (Xtrain, Z[:,i]) like your example
        for i in range(cfg.p):
            ObsManager.train_observable(i, Xtrain, Z[:, i])

        # Cost + gradient
        if Z.grad is not None:
            Z.grad.zero_()

        cost = get_cost_simple(
            Z, X, Xplus, ObsManager, nT=nTrain, lambda1=cfg.lambda1, lambda2=cfg.lambda2)
        cost.backward()

        gradZ_frob = float(torch.linalg.norm(Z.grad, ord="fro").detach().cpu())
        cost_val = float(cost.detach().cpu())

        # Bound quantities
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
                    torch.eye(MMt.shape[0], device=MMt.device, dtype=MMt.dtype)
            eigs = torch.linalg.eigvalsh(MMt)
            eps = float(torch.clamp(eigs.min(), min=cfg.eps_floor).cpu())

            # Stack G and Gplus across observables (each forward_G: (ns_query, nTrain))
            G_blocks = [ObsManager.observables[i].forward_G(
                X) for i in range(cfg.p)]
            Gp_blocks = [ObsManager.observables[i].forward_G(
                Xplus) for i in range(cfg.p)]
            G_big = torch.vstack(G_blocks)    # (p*ns_query, nTrain)
            Gp_big = torch.vstack(Gp_blocks)  # (p*ns_query, nTrain)

            G_frob = float(torch.linalg.norm(G_big, ord="fro").cpu())
            Gp_frob = float(torch.linalg.norm(Gp_big, ord="fro").cpu())

            # Singular value stats for Phi-bar and G operators
            phi_sigma_max, phi_sigma_min, phi_cond = sv_stats(
                M, eps=cfg.eps_floor)
            phiP_sigma_max, phiP_sigma_min, phiP_cond = sv_stats(
                Mplus, eps=cfg.eps_floor)

            G_sigma_max, G_sigma_min, G_cond = sv_stats(
                G_big, eps=cfg.eps_floor)
            Gp_sigma_max, Gp_sigma_min, Gp_cond = sv_stats(
                Gp_big, eps=cfg.eps_floor)

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

        results.append({
            "trial": trial,
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
            "nTrain": nTrain,
            "N": N,
            "ns_query": ns_query,
            "device": cfg.device,
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
        })

        print(
            f"[trial {trial:03d}] ||∇ZL||_F={gradZ_frob:.4e} | bound={bound:.4e} | ratio={ratio:.4e}")
        print(
            f"    cond(Phi)={phi_cond:.2e}, cond(Phi+)={phiP_cond:.2e}, "
            f"cond(G)={G_cond:.2e}, cond(G+)={Gp_cond:.2e}"
        )

        # avoid graph accumulation
        del Z, cost

    # Save outputs
    base = os.path.join(cfg.out_dir, cfg.out_name)
    csv_path = base + ".csv"
    pt_path = base + ".pt"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    torch.save({"config": cfg.__dict__, "results": results}, pt_path)

    print(f"\nSaved:\n- {csv_path}\n- {pt_path}")
    print(f"Elapsed: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
