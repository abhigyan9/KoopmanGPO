## --- IMPORTS --- ###
import GPKoopman as gpk
import torch
import matplotlib.pyplot as plt
import time
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
# from botorch.acquisition.monte_carlo import qLogExpectedImprovement
from botorch.acquisition import qLogExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf

from botorch.models import SingleTaskGP
from botorch.acquisition import qLogNoisyExpectedImprovement
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors import GammaPrior, LogNormalPrior


# ---- Build a rough, noisy surrogate ----
def build_rough_noisy_gp(train_X, train_Y):
    """
    BO surrogate GP favoring roughness and nonzero noise:
      - Matérn-1/2 (absolute exponential) with ARD
      - Priors nudging lengthscales small (rougher)
      - LogNormal prior on outputscale
      - Learn noise (SingleTaskGP) with a weak prior favoring nonzero noise
    Assumes train_X normalized outside (we also apply Normalize transform).
    """
    d = train_X.shape[-1]

    # Matérn kernel with ARD and informative priors
    base_kern = MaternKernel(
        nu=1.5,                 # roughest Matérn (try 1.5 if too spiky)
        ard_num_dims=d,
        lengthscale_prior=GammaPrior(concentration=2.0, rate=10.0)  # mean=0.2 in [0,1] space
    )
    covar = ScaleKernel(
        base_kern,
        outputscale_prior=LogNormalPrior(loc=-1.0, scale=0.5)       # favors smaller outputscale but flexible
    )

    # SingleTaskGP will create a GaussianLikelihood with learnable noise.
    # Give the noise a prior that avoids collapsing to ~0 in noisy settings.
    model = SingleTaskGP(
        train_X, train_Y,
        covar_module=covar,
        input_transform=Normalize(d=d),
        outcome_transform=Standardize(m=1),
    )
    # Optional: set a weak log-normal prior on noise (keeps it away from 0)
    model.likelihood.noise_covar.register_prior(
        "noise_prior",
        LogNormalPrior(loc=-4.0, scale=1.0),   # median ~ exp(-4) ≈ 0.018 (tune)
        "noise"
    )
    return model


## --- COST FUNCTION --- ##


def get_cost_simple(Z, X, Xplus, manager, nT=1, lambda1=1.0, lambda2=1.0, lambda3=1.0):
    """
    Computes the cost function using a single differentiable GP forward pass per observable,
    merging the training and prediction steps by passing Z[:, i] directly to the forward method.

    Args:
        Z: Tensor of shape (nT*l, p), decision variable (requires grad).
        X: Tensor of shape (n, nT*N), dataset of N steps per trajectory.
        Xplus: Tensor of shape (n, nT*N), time-shifted dataset.
        Xtrain: Tensor of shape (n, r**n), gridpoints for training.
        manager: GPObservablesManager.
        nT: Number of trajectories.
        lambda1: Weighting for multi-variate NLPD
        lambda2: Weighting for Lifting Accuracy (Bhattacharyya Distance)
        lambda3: Weighting for Reconstruction
    """
    N = X.shape[1] // nT    # Number of time steps per trajectory
    p = Z.shape[1]          # Number of observables
    l = Z.shape[0] // nT    # Decision horizon
    # n = X.shape[0]          # State dimension

    # For each observable, call forward once on the full dataset X (and Xplus)
    M = torch.empty((p, N * nT), device=X.device)
    Mplus = torch.empty((p, N * nT), device=X.device)
    # diag_all = torch.empty((p, N * nT), device=X.device)
    # diag_all_plus = torch.empty((p, N * nT), device=X.device)
    # cov_all_plus = [None] * p  # store full covariance matrices for Xplus

    for i in range(p):
        mean_i, _ = manager.observables[i].forward(X, Z[:, i])
        M[i, :] = torch.transpose(mean_i, 0, -1)
        # diag_all[i] = torch.clamp(torch.diagonal(cov_i), min=1e-3)

        mean_plus_i, _ = manager.observables[i].forward(
            Xplus, Z[:, i])
        Mplus[i, :] = torch.transpose(mean_plus_i, 0, -1)
        # diag_all_plus[i] = torch.clamp(torch.diagonal(cov_i_plus), min=1e-3)

    # Compute the pseudo-inverse lifting operator and the corresponding matrices Cz and Az.
    
    try:
        L = torch.linalg.cholesky(M @ M.mT)
        M_pinv = torch.cholesky_solve(M.mT, L)
    except RuntimeError:
        M_pinv = torch.linalg.pinv(M)
    
    M_pinvM = M_pinv @ M
    
    cost1 = torch.linalg.matrix_norm(Mplus - (Mplus @ M_pinvM))
    cost2 = torch.linalg.matrix_norm(X - (X @ M_pinvM))

    return (lambda1 * cost1) + (lambda2 * cost2)


def get_cost_ACnew2(Z, X, Xplus, manager, nT=1, lambda1=1.0, lambda2=1.0, lambda3=1.0):
    """
    Computes the cost function using a single differentiable GP forward pass per observable,
    merging the training and prediction steps by passing Z[:, i] directly to the forward method.

    Args:
        Z: Tensor of shape (nT*l, p), decision variable (requires grad).
        X: Tensor of shape (n, nT*N), dataset of N steps per trajectory.
        Xplus: Tensor of shape (n, nT*N), time-shifted dataset.
        Xtrain: Tensor of shape (n, r**n), gridpoints for training.
        manager: GPObservablesManager.
        nT: Number of trajectories.
        lambda1: Weighting for multi-variate NLPD
        lambda2: Weighting for Lifting Accuracy (Bhattacharyya Distance)
        lambda3: Weighting for Reconstruction
    """
    N = X.shape[1] // nT    # Number of time steps per trajectory
    p = Z.shape[1]          # Number of observables
    l = Z.shape[0] // nT    # Decision horizon
    # n = X.shape[0]          # State dimension

    # For each observable, call forward once on the full dataset X (and Xplus)
    M = torch.empty((p, N * nT), device=X.device)
    Mplus = torch.empty((p, N * nT), device=X.device)
    diag_all = torch.empty((p, N * nT), device=X.device)
    diag_all_plus = torch.empty((p, N * nT), device=X.device)
    # cov_all_plus = [None] * p  # store full covariance matrices for Xplus

    for i in range(p):
        mean_i, cov_i = manager.observables[i].forward(X, Z[:, i])
        M[i, :] = torch.transpose(mean_i, 0, -1)
        diag_all[i] = torch.clamp(torch.diagonal(cov_i), min=1e-3)

        mean_plus_i, cov_i_plus = manager.observables[i].forward(
            Xplus, Z[:, i])
        Mplus[i, :] = torch.transpose(mean_plus_i, 0, -1)
        diag_all_plus[i] = torch.clamp(torch.diagonal(cov_i_plus), min=1e-3)

    # Compute the pseudo-inverse lifting operator and the corresponding matrices Cz and Az.
    M_pinv = torch.linalg.pinv(M)
    Cz = X @ M_pinv
    Az = Mplus @ M_pinv

    # Cost Term 1: 1-step Prediction Euclid + Trace
    NormNLPD = 0.0
    if lambda1 != 0.0:
        # --- after you have diag_all (p × NT), M (p × NT), X (n × NT), Cz (n×p), Az (p×p) ---
        num_steps = N - 2 - l
        device = X.device

        # 1) build indices for all (j,k) at once
        offsets = torch.arange(nT, device=device) * \
            N                          # (nT,)
        # (num_steps,)
        idx_base = torch.arange(num_steps, device=device)
        idx_M = (offsets[:, None] + (l+1) + idx_base[None, :]
                 ).reshape(-1)     # (nT*num_steps,)
        # shift by one for X
        idx_X = idx_M + 1

        # 2) gather into big “batch” of size B = nT*num_steps
        diag_batch = diag_all[:, idx_M]         # (p, B)
        M_batch = M[:, idx_M]           # (p, B)
        X_batch = X[:, idx_X]           # (n, B)

        # 3) compute A_z @ diag @ A_z^T in batch
        #    diag_batch.T has shape (B,p), so
        AzD = Az[None, :, :] * diag_batch.T[:,
                                            :, None]  # (B, p, p)  = Az @ diag
        A_batch = AzD @ Az.T                           # (B, p, p)

        # 4) push through Cz to get vx_batch = Cz * A_batch * Cz^T
        Cz_exp = Cz[None, :, :]                       # (1, n, p)
        B1_batch = Cz_exp @ A_batch                   # (B, n, p)
        vx_batch = (B1_batch @ Cz.T[None, :, :]).abs()  # (B, n, n)

        # 5) batched Cholesky & inverse
        # L_batch      = torch.linalg.cholesky(vx_batch)        # (B, n, n)
        # vx_inv_batch = torch.cholesky_inverse(L_batch)        # (B, n, n)

        # 6) batched error vectors
        #    first compute Cz @ Az @ M_batch  → shape (n, B)
        CtM = Cz @ (Az @ M_batch)                          # (n, B)
        err = (X_batch - CtM).T                            # (B, n)

        # 7) batched quadratic form + logdet
        #    a) quadratic form: eᵢᵀ V⁻¹ eᵢ for each i
        # qf     = torch.einsum('bi,bij,bj->b', err, vx_inv_batch, err)  # (B,)
        # (B,) | un-normalized
        qf = (err ** 2).sum(dim=1)

        #    b) log-det: use slogdet for stability
        # sign, logdet = torch.linalg.slogdet(vx_batch)        # both (B,)
        trace_batch = torch.diagonal(
            vx_batch, dim1=-2, dim2=-1).sum(dim=-1)  # (B,)
        #    (sign should be all +1 if SPD)

        NormNLPD = (qf + trace_batch).sum()

    # Cost Term 2: Lifting Accuracy (Bhattacharyya Distance)
    NormLift = 0.0
    if lambda2 != 0.0:
        # B = total (trajectory, step) pairs
        B = nT * N

        # Gather per-step variances
        # diag_all: (p, B), diag_all_plus: (p, B)
        d_k = diag_all.permute(1, 0)        # (B, p)
        d_kp = diag_all_plus.permute(1, 0)   # (B, p)

        # Dvkp_k = Az @ diag(d_k[b]) @ Az.T   (batched)
        Az_exp = Az.unsqueeze(0)                                   # (1, p, p)
        # (B, p, p) == Az @ diag(d_k)
        A_d = Az_exp * d_k.unsqueeze(1)
        Dvkp_k = A_d @ Az_exp.transpose(-1, -2)                    # (B, p, p)

        # Dvkp_kp = diag(d_kp[b]) (batched)
        Dvkp_kp = torch.diag_embed(d_kp)                            # (B, p, p)

        # σ = 0.5 * (Dvkp_k + Dvkp_kp)
        sigma = 0.5 * (Dvkp_k + Dvkp_kp)                          # (B, p, p)

        # Cholesky + inverse of σ
        # L         = torch.linalg.cholesky(sigma)                     # (B, p, p)
        # sigma_inv = torch.cholesky_inverse(L)                        # (B, p, p)

        # err = Mplus - Az @ M
        pred = Az @ M                                                # (p, B)
        err = (Mplus - pred).permute(1, 0)                          # (B, p)

        # Quadratic term: (errᵀ σ⁻¹ err) / 8  (matches your loop)
        # qf = torch.einsum('bi,bij,bj->b', err, sigma_inv, err) / 8.0 # (B,)
        qf = (err ** 2).sum(dim=1)

        # --- Diagonal-only logdet for Dvkp_k ---
        # diag(Dvkp_k)_i = sum_j Az[i,j]^2 * d_k[b,j]  ⇒ diag = d_k @ (Az⊙Az)^T
        eps = 1e-12
        Az_sq = Az.pow(2)                                     # (p, p)
        diag_prop = d_k @ Az_sq.T                                 # (B, p)
        logdet_Dvkp_k_diag = torch.log(
            diag_prop.clamp_min(eps)).sum(dim=1)  # (B,)

        # logdet(Dvkp_kp) for diagonal matrix = sum(log d_kp)
        logdet_Dvkp_kp = torch.log(d_kp.clamp_min(eps)).sum(dim=1)   # (B,)

        # logdet(σ) via slogdet (stable; same math as logdet for SPD)
        _, logdet_sigma = torch.linalg.slogdet(sigma)                 # (B,)

        # 0.5 * [ logdet(σ) - 0.5*( logdet_diag(Dvkp_k) + logdet(Dvkp_kp) ) ]
        log_term = 0.5 * (logdet_sigma - 0.5 *
                          (logdet_Dvkp_k_diag + logdet_Dvkp_kp))  # (B,)

        NormLift = (qf + log_term).sum()

    # Cost Term 3: Reconstruction Euclid + Trace
    NormRecon = 0.0
    if lambda3 != 0.0:
        # Total number of time-points across all trajectories
        B = nT * N          # B = number of (j,k) pairs

        # 1) Build batch of weighted C_z matrices: (B, n, p)
        #    Cz_exp: (1, n, p)  broadcast to (B, n, p)
        #    d_exp : (B, 1, p)  broadcast to (B, n, p)
        Cz_exp = Cz.unsqueeze(0)                          # (1, n, p)
        d_exp = diag_all.T.unsqueeze(1)                  # (B, 1, p)
        # (B, n, p) = C_z * diag(vz)
        CzD = Cz_exp * d_exp

        # 2) Form the full covariances: (B, n, n)
        vx_batch = CzD @ Cz.T                             # (B, n, n)

        # 3) Batched Cholesky + inverse
        # L_batch     = torch.linalg.cholesky(vx_batch)     # (B, n, n)
        # inv_batch   = torch.cholesky_inverse(L_batch)     # (B, n, n)

        # 4) Gather all errors in one go
        pred_batch = Cz @ M                              # (n,  B)
        err_batch = (X - pred_batch).T                   # (B,  n)

        # 5) Batched quadratic form eᵀ V⁻¹ e  →  (B,)
        # qf_batch    = torch.einsum('bi,bij,bj->b', err_batch, inv_batch, err_batch)
        # (B,) | un-normalized
        qf_batch = (err_batch ** 2).sum(dim=1)

        # 6) Batched log-determinant
        # _, logdet_batch = torch.linalg.slogdet(vx_batch)
        trace_batch = torch.diagonal(
            vx_batch, dim1=-2, dim2=-1).sum(dim=-1)  # (B,)
        # (we expect sign all +1 if SPD)

        # 7) Sum everything
        NormRecon = (qf_batch + trace_batch).sum()

    cost = (lambda1 * NormNLPD / ((N - l) * nT)) + (lambda2 *
                                                    NormLift / (N * nT)) + (lambda3 * NormRecon / (N * nT))
    return cost


def _mgr_pack_params_flat(ObsManager):
    """
    Flattens ObsManager parameters (hp1, hp2, noise, mu) into a 1-D float32 vector.
    Also returns a 'spec' list to reconstruct shapes in the same order.
    Order: [hp1_all, hp2_all, noise_all, mu_all]
    """
    spec = []  # list of tuples: (kind, shape, obs_idx, k_idx or None)
    flat_chunks = []

    # Iterate in the same way ObsManager.parameters() builds the list:
    # for each observable, extend with hp1_list, hp2_list, mu_list, noise
    # We need per-observable access to preserve shapes.
    for obs_idx, obs in ObsManager.observables.items():
        # hp1
        for k, t in enumerate(obs.hp1_list):
            spec.append(("hp1", t.shape, obs_idx, k))
            flat_chunks.append(t.detach().to(dtype=torch.float32).reshape(-1))
        # hp2
        for k, t in enumerate(obs.hp2_list):
            spec.append(("hp2", t.shape, obs_idx, k))
            flat_chunks.append(t.detach().to(dtype=torch.float32).reshape(-1))
        # noise (scalar tensor)
        t = obs.noise
        spec.append(("noise", t.shape, obs_idx, None))
        flat_chunks.append(t.detach().to(dtype=torch.float32).reshape(-1))
        # mu
        # for k, t in enumerate(obs.mu_list):
        #     spec.append(("mu", t.shape, obs_idx, k))
        #     flat_chunks.append(t.detach().to(dtype=torch.float32).reshape(-1))

    if len(flat_chunks) == 0:
        return torch.empty(0, dtype=torch.float32, device=next(iter(ObsManager.observables.values())).device), spec

    flat = torch.cat(flat_chunks, dim=0).to(dtype=torch.float32,
                                            device=next(iter(ObsManager.observables.values())).device)
    return flat, spec


def _mgr_unpack_and_set_params(ObsManager, flat_vec32, spec):
    """
    Reconstructs lists for hp1, hp2, noise, mu from flat_vec32 using 'spec'
    and writes them back through ObsManager.set_parameters(...).
    """
    # Prepare per-kind containers in the same total order as manager.set_parameters expects
    hp1_list, hp2_list, mu_list, noise_list = [], [], [], []

    idx = 0
    for kind, shape, obs_idx, k in spec:
        n = int(torch.prod(torch.tensor(shape))) if len(shape) > 0 else 1
        chunk = flat_vec32[idx:idx+n]
        idx += n
        tensor = chunk.view(shape).to(dtype=torch.float32,
                                      device=next(iter(ObsManager.observables.values())).device)
        if kind == "hp1":
            hp1_list.append(tensor)
        elif kind == "hp2":
            hp2_list.append(tensor)
        elif kind == "noise":
            noise_list.append(tensor)
        # elif kind == "mu":
        #     mu_list.append(tensor)
        else:
            raise ValueError(f"Unknown kind in spec: {kind}")

    # Write back (manager will assign in observable-wise order)
    ObsManager.set_parameters(
        hp1_list=hp1_list if len(hp1_list) else None,
        hp2_list=hp2_list if len(hp2_list) else None,
        noise_list=noise_list if len(noise_list) else None,
        mu_list=mu_list if len(mu_list) else None,
    )


def _mgr_build_hp_bounds(ObsManager, X, default_hp1=(1e-4, 2.0), default_hp2=(1e-4, 2.0),
                         default_noise=(1e-8, 1e-1), mu_margin=0.1):
    """
    Builds [2, dh] bounds aligned to the flat parameter vector produced by _mgr_pack_params_flat.
    hp1/hp2/noise use global boxes; μ uses per-dimension min/max from X with a margin.
    """
    flat, spec = _mgr_pack_params_flat(ObsManager)
    if flat.numel() == 0:
        return torch.empty(2, 0, dtype=torch.double, device=X.device), spec

    # Data bounds for mu (per state dimension)
    # X is shape [n, N_tot] (from your code): compute per-dim min/max
    x_min = X.min(dim=1).values  # [n]
    x_max = X.max(dim=1).values  # [n]
    span = (x_max - x_min)
    x_lo = x_min - mu_margin * span
    x_hi = x_max + mu_margin * span

    lbs, ubs = [], []
    n_dim = X.shape[0]

    for kind, shape, obs_idx, k in spec:
        n = int(torch.prod(torch.tensor(shape))) if len(shape) > 0 else 1
        if kind == "hp1":
            lb, ub = default_hp1
            lbs.append(torch.full(
                (n,), lb, device=X.device, dtype=torch.double))
            ubs.append(torch.full(
                (n,), ub, device=X.device, dtype=torch.double))
        elif kind == "hp2":
            lb, ub = default_hp2
            lbs.append(torch.full(
                (n,), lb, device=X.device, dtype=torch.double))
            ubs.append(torch.full(
                (n,), ub, device=X.device, dtype=torch.double))
        elif kind == "noise":
            lb, ub = default_noise
            lbs.append(torch.full(
                (n,), lb, device=X.device, dtype=torch.double))
            ubs.append(torch.full(
                (n,), ub, device=X.device, dtype=torch.double))
        elif kind == "mu":
            # shape typically (d,1); bound each element by per-dim x_lo/x_hi (tile across kernels)
            if len(shape) == 2 and shape[1] == 1 and shape[0] == n_dim:
                lbs.append(x_lo.to(dtype=torch.double))
                ubs.append(x_hi.to(dtype=torch.double))
            else:
                # fallback: just use a wide box
                lbs.append(torch.full(
                    (n,), -3.0, device=X.device, dtype=torch.double))
                ubs.append(torch.full(
                    (n,),  3.0, device=X.device, dtype=torch.double))
        else:
            raise ValueError(f"Unknown kind: {kind}")

    LB = torch.cat(lbs).unsqueeze(0)  # [1, dh]
    UB = torch.cat(ubs).unsqueeze(0)  # [1, dh]
    return torch.cat([LB, UB], dim=0), spec  # [2, dh], spec


def _bo_optimize_Z(
    Z_shape,                      # tuple: (l*nTrain, p)
    eval_cost_fn,                 # callable: Z_float32 -> scalar tensor()
    device,
    bounds=(0.0, 1.0),            # per-dimension box
    n_init: int = 64,
    n_iter: int = 128,
    q: int = 1,
    seed: int = 1234,
):
    """
    Bayesian Optimization on flattened Z using a SingleTaskGP and qEI.
    Internals use float64 for the GP, but objective eval remains float32.
    Returns
    -------
    Z_best : torch.Tensor with shape Z_shape (float32 on device)
    cost_hist : list[float] of best-so-far after each eval
    """
    torch.manual_seed(seed)
    d = Z_shape[0] * Z_shape[1]
    lb, ub = float(bounds[0]), float(bounds[1])

    # Build [2, d] bounds in double for BoTorch
    bounds_d = torch.stack([
        torch.full((d,), lb, device=device, dtype=torch.double),
        torch.full((d,), ub, device=device, dtype=torch.double),
    ])

    # 1) Sobol init in [0,1]^d (double for BoTorch)
    sobol = torch.quasirandom.SobolEngine(
        dimension=d, scramble=True, seed=seed)
    X_init = sobol.draw(n_init).to(
        device=device, dtype=torch.double)  # [n_init, d]
    if (lb, ub) != (0.0, 1.0):
        X_init = lb + (ub - lb) * X_init
    X_init = X_init.clamp(lb, ub)

    # Evaluate initial design with your float32 objective
    Y_list = []
    best_so_far = float('inf')
    best_x = None
    cost_hist = []

    with torch.no_grad():
        for i in range(n_init):
            # cast candidate to float32 for your gpk cost
            z32 = X_init[i].to(dtype=torch.float32).view(*Z_shape)
            y = eval_cost_fn(z32).detach().view(
                1, 1)  # returns float32 scalar tensor
            # store in double for the GP
            y_d = y.to(dtype=torch.double)
            Y_list.append(y_d)
            yi = float(y.item())
            if yi < best_so_far:
                best_so_far = yi
                best_x = X_init[i].clone()
            cost_hist.append(best_so_far)

    train_X = X_init                               # [N, d], double
    train_Y = torch.cat(Y_list, dim=0)             # [N, 1], double

    # 2) BO loop with qEI
    for iter_num in range(n_iter):
        # Fit GP with scaling transforms (helps SciPy and numerics)
        gp = SingleTaskGP(
            train_X, train_Y,
            input_transform=Normalize(d),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # MC sampler (old/new API compatibility)
        try:
            sampler = SobolQMCNormalSampler(num_samples=256)
        except TypeError:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))

        acqf = qLogExpectedImprovement(
            model=gp, best_f=train_Y.min(), sampler=sampler)

        # Try to optimize the acquisition; if SciPy hiccups, fallback to random Sobol
        try:
            cand, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds_d,
                q=q,
                num_restarts=15,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200},
            )
        except Exception:
            # Fallback: sample candidates and choose the best by acq value
            rand_cands = torch.quasirandom.SobolEngine(
                d, scramble=True, seed=seed+7).draw(256)
            rand_cands = lb + (ub - lb) * \
                rand_cands.to(device=device, dtype=torch.double)
            rand_cands = rand_cands.clamp(lb, ub)
            with torch.no_grad():
                acq_vals = acqf(rand_cands.unsqueeze(-2))  # shape [256, 1]
            top_idx = torch.topk(acq_vals.view(-1), k=q).indices
            # shape [q, 1, d] -> match optimize_acqf out
            cand = rand_cands[top_idx].unsqueeze(1)
            cand = cand.squeeze(1)

        cand = cand.clamp(lb, ub)

        # Evaluate candidates with your float32 objective
        with torch.no_grad():
            Y_new = []
            for j in range(q):
                z32 = cand[j].to(dtype=torch.float32).view(*Z_shape)
                y = eval_cost_fn(z32).detach().view(1, 1)          # float32
                # store as double
                y_d = y.to(dtype=torch.double)
                Y_new.append(y_d)
                yi = float(y.item())
                if yi < best_so_far:
                    best_so_far = yi
                    best_x = cand[j].detach().clone()
                cost_hist.append(best_so_far)
            Y_new = torch.cat(Y_new, dim=0)  # [q,1], double
        
        print(f'BO Iteration {iter_num} completed with minimum cost = {Y_new.min()}')
        # Augment data (double)
        train_X = torch.cat([train_X, cand], dim=0)
        train_Y = torch.cat([train_Y, Y_new], dim=0)

    # Return best in float32 for downstream gpk code
    Z_best32 = best_x.to(dtype=torch.float32).view(*Z_shape)
    return Z_best32, cost_hist


def _bo_hp_outer_gdZ_inner(
    # torch.Tensor, shape (l*nTrain, p), float32 on device
    Z_init,
    ObsManager,                   # GPObservablesManager
    X, Xplus, nT,                 # training tensors
    # [2, dh] double, from _mgr_build_hp_bounds(...)
    hp_bounds_mat,
    lam1, lam2, lam3,             # weights for get_cost_ACnew2
    device,
    n_outer_iter=10,              # ← 10 iterations
    q_batch=20,                   # ← 20 samples per iteration
    inner_steps=200,              # ← 200 GD steps on Z
    lr_Z=0.02,
    seed=1234,
    warm_start=True
):
    """
    Bayesian Optimization over flattened hyperparameters (hp) with a nested
    inner loop: for each hp candidate, run gradient descent on Z for 'inner_steps'
    and use the minimum get_cost_ACnew2 reached as the objective value.

    Returns: best_Z32, best_hp32, history dict
    """
    torch.manual_seed(seed)
    dZ = Z_init.shape[0] * Z_init.shape[1]
    # 1) Flatten current hp and prepare bounds
    hp0_flat32, hp_spec = _mgr_pack_params_flat(ObsManager)
    if hp0_flat32.numel() == 0:
        raise ValueError("No hyperparameters in ObsManager.")
    assert hp_bounds_mat.shape == (
        2, hp0_flat32.numel()), "hp_bounds_mat must be [2, dh]"

    dh = hp0_flat32.numel()
    bounds_d = hp_bounds_mat.to(device=device, dtype=torch.double)
    lb_hp, ub_hp = bounds_d[0], bounds_d[1]

    # -- storage for BO train data (double for BoTorch)
    train_X = torch.empty(0, dh, dtype=torch.double, device=device)
    train_Y = torch.empty(0, 1,  dtype=torch.double, device=device)

    best_val = float("inf")
    best_hp = None
    best_Z = None
    cost_trace = []

    # Warm-start template for Z per candidate
    # Z_template = Z_init.detach().clone()

    def eval_candidate_hp(x_hp_double, z0):
        """
        x_hp_double: [dh] double tensor on device
        Steps:
          1) set hp in manager
          2) inner GD on Z for 'inner_steps' (starting from Z_template)
          3) return min cost reached, and best Z found for that hp
        """
        # 1) set hp
        hp32 = x_hp_double.to(dtype=torch.float32)
        _mgr_unpack_and_set_params(ObsManager, hp32, hp_spec)

        # 2) inner GD on Z
        Z = torch.nn.Parameter(z0.detach().clone())
        # optZ = torch.optim.SGD([Z], lr=lr_Z, momentum=0.75, nesterov=True)
        optZ = torch.optim.Adam([Z], lr=lr_Z)

        best_inner = float("inf")
        best_Z_inner = None

        with torch.enable_grad():
            for _ in range(inner_steps):
                optZ.zero_grad(set_to_none=True)
                cost = get_cost_simple(
                    Z, X, Xplus, ObsManager,
                    nT=nT, lambda1=lam1, lambda2=lam2, lambda3=lam3
                )
                cost.backward()
                optZ.step()

                cval = float(cost.detach().item())
                if cval < best_inner:
                    best_inner = cval
                    best_Z_inner = Z.detach().clone()

        return best_inner, best_Z_inner

    # ===== BO loop =====
    # Initial design via Sobol: evaluate q_batch points
    sobol = torch.quasirandom.SobolEngine(dh, scramble=True, seed=seed)
    X0u = sobol.draw(q_batch).to(
        device=device, dtype=torch.double)  # [q_batch, dh] in [0,1]
    X0 = lb_hp + (ub_hp - lb_hp) * X0u
    with torch.no_grad():
        Y0_vals = []
        z_warm = Z_init.detach().clone()  # seed warm start
        for i in range(q_batch):
            z0 = (best_Z if (warm_start and best_Z is not None)
                  else z_warm) if warm_start else Z_init
            yi, Zi = eval_candidate_hp(X0[i], z0)
            train_X = torch.cat([train_X, X0[i:i+1]], dim=0)
            train_Y = torch.cat([train_Y, torch.tensor(
                [[yi]], dtype=torch.double, device=device)], dim=0)

            # global best tracking
            if yi < best_val:
                best_val, best_hp, best_Z = yi, X0[i].detach(
                ).clone(), Zi.detach().clone()
                if warm_start:  # promote global best to new warm start
                    z_warm = best_Z.detach().clone()
            cost_trace.append(best_val)
            Y0_vals.append(yi)

    # Iterations
    for inner_iter in range(n_outer_iter):
        gp = SingleTaskGP(
            train_X, train_Y,
            input_transform=Normalize(d=dh),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        try:
            sampler = SobolQMCNormalSampler(num_samples=256)
        except TypeError:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))

        acq = qLogExpectedImprovement(
            model=gp, best_f=train_Y.min(), sampler=sampler)

        cand, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds_d,
            q=q_batch,
            num_restarts=15,
            raw_samples=512,
            options={"batch_limit": 5, "maxiter": 200},
        )
        cand = cand.clamp(lb_hp, ub_hp)

        for j in range(q_batch):
            z0 = (best_Z if (warm_start and best_Z is not None)
                  else z_warm) if warm_start else Z_init
            yj, Zj = eval_candidate_hp(cand[j], z0)

            train_X = torch.cat([train_X, cand[j:j+1]], dim=0)
            train_Y = torch.cat([train_Y, torch.tensor(
                [[yj]], dtype=torch.double, device=device)], dim=0)

            if yj < best_val:
                best_val, best_hp, best_Z = yj, cand[j].detach(
                ).clone(), Zj.detach().clone()
                if warm_start:
                    z_warm = best_Z.detach().clone()

            cost_trace.append(best_val)
        
        print(f'Finished BO Iteration {inner_iter} with Min Loss {cost_trace[-1]:.2e}')

    eval_costs = torch.tensor(cost_trace, device=device)
    per_iter_idx = [((i+1)*q_batch - 1) for i in range(n_outer_iter + 1)]
    per_iter_best = eval_costs[per_iter_idx].detach().cpu().tolist()

    return best_Z, best_hp.to(dtype=torch.float32), {"cost": eval_costs, "per_iter_best": per_iter_best,
                                                     "n_outer": n_outer_iter, "q_batch": q_batch}


def _bo_optimize_Z_and_hp(
    Z_shape,                      # tuple: (l*nTrain, p)
    eval_cost_fn,                 # callable: (Z32, hp32) -> scalar tensor()
    # callables: get_hp() -> hp32 [dh], set_hp(hp32)
    get_hp, set_hp,
    device,
    Z_bounds=(0.0, 1.0),          # tuple for all Z dims
    hp_bounds_mat=None,           # [2, dh] per-dimension bounds for hp
    n_init: int = 32,
    n_iter: int = 128,
    q: int = 1,
    seed: int = 1234,
):
    """
    Joint BO over flattened Z and hp. BoTorch runs in float64; cost in float32.
    Returns
    -------
    Z_best32 : Z tensor with shape Z_shape (float32)
    hp_best32: flat hp tensor (float32)
    cost_hist: list[float]
    """
    torch.manual_seed(seed)
    dZ = Z_shape[0] * Z_shape[1]
    hp0_32 = get_hp()
    assert hp0_32.dim() == 1, "hp getter must return a flat 1-D tensor"
    dh = hp0_32.numel()

    # Build per-dim bounds [2, dZ+dh] in double
    lbZ, ubZ = map(float, Z_bounds)
    bounds_list_lb = [lbZ] * dZ
    bounds_list_ub = [ubZ] * dZ

    if hp_bounds_mat is None:
        raise ValueError("hp_bounds_mat [2, dh] required for BO_Z_and_hp")

    assert hp_bounds_mat.shape == (2, dh), "hp_bounds_mat must be [2, dh]"
    lb_hp = hp_bounds_mat[0].tolist()
    ub_hp = hp_bounds_mat[1].tolist()

    bounds_lb = torch.tensor(bounds_list_lb + lb_hp,
                             dtype=torch.double, device=device)
    bounds_ub = torch.tensor(bounds_list_ub + ub_hp,
                             dtype=torch.double, device=device)
    bounds_d = torch.stack([bounds_lb, bounds_ub], dim=0)  # [2, dTot]
    dTot = dZ + dh

    # 1) Sobol init within bounds
    sobol = torch.quasirandom.SobolEngine(
        dimension=dTot, scramble=True, seed=seed)
    # [n_init, dTot] in [0,1]
    U = sobol.draw(n_init).to(device=device, dtype=torch.double)
    X_init = bounds_lb + (bounds_ub - bounds_lb) * U
    X_init = X_init.clamp(bounds_lb, bounds_ub)

    # Evaluate initial design
    Y_list = []
    best_so_far = float('inf')
    best_x = None
    cost_hist = []

    with torch.no_grad():
        for i in range(n_init):
            xi = X_init[i]
            Z_part32 = xi[:dZ].to(dtype=torch.float32).view(*Z_shape)
            hp_part32 = xi[dZ:].to(dtype=torch.float32).view(-1)

            # set hp in ObsManager (float32)
            set_hp(hp_part32)
            y = eval_cost_fn(Z_part32, hp_part32)  # returns float32 scalar
            y_d = y.detach().to(dtype=torch.double).view(1, 1)
            Y_list.append(y_d)
            yi = float(y.item())
            if yi < best_so_far:
                best_so_far = yi
                best_x = xi.clone()
            # cost_hist.append(best_so_far)

    train_X = X_init                               # [N, dTot], double
    train_Y = torch.cat(Y_list, dim=0)             # [N, 1], double

    # 2) BO loop
    for iter_num in range(n_iter):
        # gp = SingleTaskGP(
        #     train_X, train_Y,
        #     input_transform=Normalize(dTot),
        #     outcome_transform=Standardize(m=1),
        # )
        # mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        # fit_gpytorch_mll(mll)
        gp = build_rough_noisy_gp(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # Sampler (API compatibility)
        try:
            sampler = SobolQMCNormalSampler(num_samples=256)
        except TypeError:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))

        # acqf = qLogExpectedImprovement(
        #     model=gp, best_f=train_Y.min(), sampler=sampler)
        acqf = qLogNoisyExpectedImprovement(model=gp, X_baseline=train_X)

        # Optimize acqf; fallback to Sobol if SciPy fails
        try:
            cand, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds_d,
                q=q,
                num_restarts=15,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200},
            )
        except Exception:
            rand_cands = torch.quasirandom.SobolEngine(
                dTot, scramble=True, seed=seed+9).draw(256)
            rand_cands = bounds_lb + \
                (bounds_ub - bounds_lb) * \
                rand_cands.to(device=device, dtype=torch.double)
            rand_cands = rand_cands.clamp(bounds_lb, bounds_ub)
            with torch.no_grad():
                acq_vals = acqf(rand_cands.unsqueeze(-2))  # [256,1]
            top_idx = torch.topk(acq_vals.view(-1), k=q).indices
            cand = rand_cands[top_idx]                     # [q, dTot]

        cand = cand.clamp(bounds_lb, bounds_ub)

        with torch.no_grad():
            Y_new = []
            for j in range(q):
                xj = cand[j]
                Z_part32 = xj[:dZ].to(dtype=torch.float32).view(*Z_shape)
                hp_part32 = xj[dZ:].to(dtype=torch.float32).view(-1)
                set_hp(hp_part32)
                y = eval_cost_fn(Z_part32, hp_part32).detach().view(
                    1, 1)   # float32
                y_d = y.to(dtype=torch.double)
                Y_new.append(y_d)
                yi = float(y.item())
                if yi < best_so_far:
                    best_so_far = yi
                    best_x = xj.detach().clone()
                cost_hist.append(best_so_far)
            Y_new = torch.cat(Y_new, dim=0)

        train_X = torch.cat([train_X, cand], dim=0)
        train_Y = torch.cat([train_Y, Y_new], dim=0)
        print(f'Finished BO Iteration {iter_num} with cost {best_so_far}')

    # Unpack best
    Z_best32 = best_x[:dZ].to(dtype=torch.float32).view(*Z_shape)
    hp_best32 = best_x[dZ:].to(dtype=torch.float32).view(-1)

    eval_costs = torch.tensor(cost_hist, device=device)
    per_iter_idx = [((i+1)*q - 1) for i in range(n_iter + 1)]
    per_iter_best = eval_costs[per_iter_idx].detach().cpu().tolist()

    return Z_best32, hp_best32, {"cost": eval_costs, "per_iter_best": per_iter_best,
                                                     "n_outer": n_iter, "q_batch": q}


def get_iGPK(
    # normalized (and optionally noised) data, (num_traj, n, N+1)
    SimData,
    nTrain, nTest,
    lifting_order,
    # [max_iter, (optional phase_len_hp), (optional phase_len_z), (optional reserve_final_z)]
    iters_list,
    learn_rate,
    opt_weights,                 # [lambda1, lambda2, lambda3]
    routine="Z_only",            # "Z_only" or "SpacedOpt"
    train_method="Horizon",      # "Horizon" or "K-Means"
    device="cuda:0",
    seed=1234
):
    """
    Train iGPK, build Koopman (A, C), simulate train/test, and return predictions, covariances, NRMSE.

    NOTE: Data loading & noise addition remain outside. Pass prepped SimData in.
    """
    torch.manual_seed(seed)
    SimData = SimData.float().to(device)

    # Shapes & basic splits
    n = SimData.shape[1]
    N = SimData.shape[2] - 1
    p = lifting_order
    l = 1  # lifting horizon fixed to 1 (kept from your script)

    # Build concatenated matrices and ICs from SimData
    Xall = torch.cat([SimData[j, :, :]
                     for j in range(nTrain)], dim=1)     # n x (nTrain*(N+1))
    X = torch.cat([SimData[j, :, 0:N]
                  for j in range(nTrain)], dim=1)     # n x (nTrain*N)
    Xplus = torch.cat([SimData[j, :, 1:]
                      for j in range(nTrain)], dim=1)     # n x (nTrain*N)

    ICsetTrain = torch.cat([SimData[j, :, 0].view(n, 1)
                           for j in range(nTrain)], dim=1)
    ICsetTest = torch.cat([SimData[j, :, 0].view(n, 1)
                          for j in range(nTrain, nTrain + nTest)], dim=1)

    # Initialize manager & decision variable Z (training grid)
    if train_method == "Horizon":
        Xtrain = torch.cat([X[:, j*N: j*N + l]
                           for j in range(nTrain)], dim=1)  # n x (nTrain*l)
        Z = torch.nn.Parameter(torch.rand(Xtrain.shape[1], p, device=device))
        ObsManager = gpk.GPObservablesManager()
        for i in range(p):
            ObsManager.add_observable(
                index=i, d=n, ns=l*nTrain, kernel_types=['Gaussian'],
                combination='sum', noise=1e-4, m=500, device=device
            )
        for i in range(p):
            ObsManager.train_observable(i, Xtrain, Z[:, i])
        ObsManager.set_random_hyperparameters(scale=[1.0, 1.0, None])

    elif train_method == "K-Means":
        Xtrain = torch.cat([X[:, j*N: j*N + l] for j in range(nTrain)], dim=1)
        Z = torch.nn.Parameter(torch.rand(Xtrain.shape[1], p, device=device))
        ObsManager = gpk.GPObservablesManager()
        centroids = gpk.get_kmeans(X, num_centers=p)
        for i in range(p):
            ObsManager.add_observable(
                index=i, d=n, ns=l*nTrain,
                kernel_types=['ExplicitAttractor', 'Gaussian'],
                combination='sum', noise=1e-4, m=500, device=device
            )
        for i in range(p):
            ObsManager.train_observable(i, Xtrain, Z[:, i])
        ObsManager.set_random_hyperparameters(scale=[1.0, 2.0, None])
        mu_centroids = [centroids[:, i:i+1] for i in range(centroids.shape[1])]
        mu_centroids.extend(mu_centroids)
        ObsManager.set_parameters(mu_list=mu_centroids)
    else:
        raise ValueError(f"Unrecognized train_method: {train_method}")

    # === Optimization ===
    max_iter = iters_list[0]
    lam1, lam2, lam3 = opt_weights
    iter = 0
    cost_history = []

    # NEW: define a closure to evaluate cost at a given Z (no autograd needed for BO)
    def _eval_cost_at(Z_current):
        return get_cost_ACnew2(Z_current, X, Xplus, ObsManager,
                               nT=nTrain, lambda1=lam1, lambda2=lam2, lambda3=lam3)

    def _eval_cost_at_Zhp(Z_current, _hp_vec32):
        # hp was already set by the BO helper prior to calling this
        return get_cost_ACnew2(Z_current, X, Xplus, ObsManager,
                               nT=nTrain, lambda1=lam1, lambda2=lam2, lambda3=lam3)

    def _set_requires(params, flag: bool):
        for p_ in params:
            if p_ is not None:
                p_.requires_grad_(flag)

    if routine == "SpacedOpt":
        phase_len_hp = iters_list[1] if len(iters_list) > 1 else 100
        phase_len_z = iters_list[2] if len(iters_list) > 2 else 100
        reserve_final_z = iters_list[3] if len(iters_list) > 3 else 200

        hp_params = []
        for obs in ObsManager.observables.values():
            hp_params += list(getattr(obs, "hp1_list", []))
            hp_params += list(getattr(obs, "hp2_list", []))

        lr_Z, lr_HP = learn_rate, learn_rate * 0.5
        optZ = torch.optim.Adam([Z],       lr=lr_Z)
        optHP = torch.optim.Adam(hp_params, lr=lr_HP) if hp_params else None

        switch_point = max(0, max_iter - reserve_final_z)

        while iter < switch_point:
            # HP phase
            if optHP and phase_len_hp > 0:
                _set_requires([Z], False)
                _set_requires(hp_params, True)
                for _ in range(min(phase_len_hp, switch_point - iter)):
                    optHP.zero_grad(set_to_none=True)
                    cost = get_cost_ACnew2(Z, X, Xplus, ObsManager,
                                           nT=nTrain, lambda1=lam1, lambda2=lam2, lambda3=lam3)
                    cost.backward()
                    optHP.step()
                    cost_history.append(cost.item())
                    iter += 1
                    if iter >= switch_point:
                        break
            if iter >= switch_point:
                break

            # Z phase
            if phase_len_z > 0:
                _set_requires([Z], True)
                _set_requires(hp_params, False)
                for _ in range(min(phase_len_z, switch_point - iter)):
                    optZ.zero_grad(set_to_none=True)
                    cost = get_cost_ACnew2(Z, X, Xplus, ObsManager,
                                           nT=nTrain, lambda1=lam1, lambda2=lam2, lambda3=lam3)
                    cost.backward()
                    optZ.step()
                    cost_history.append(cost.item())
                    iter += 1
                    if iter >= switch_point:
                        break

        # Final Z-only
        _set_requires([Z], True)
        _set_requires(hp_params, False)
        while iter < max_iter:
            optZ.zero_grad(set_to_none=True)
            cost = get_cost_ACnew2(Z, X, Xplus, ObsManager,
                                   nT=nTrain, lambda1=lam1, lambda2=lam2, lambda3=lam3)
            cost.backward()
            optZ.step()
            cost_history.append(cost.item())
            iter += 1

    elif routine == "Z_only":
        optimizer = torch.optim.SGD(
            [Z], lr=learn_rate, momentum=0.75, nesterov=True)
        while iter < max_iter:
            optimizer.zero_grad()
            cost = get_cost_ACnew2(Z, X, Xplus, ObsManager,
                                   nT=nTrain, lambda1=lam1, lambda2=lam2, lambda3=lam3)
            cost.backward()
            optimizer.step()
            cost_history.append(cost.item())
            iter += 1

    elif routine == "BO_Z":
        # Bounds for Z: keep same range as your random init
        bo_Z, bo_hist = _bo_optimize_Z(
            Z_shape=(Z.shape[0], Z.shape[1]),
            eval_cost_fn=_eval_cost_at,
            device=SimData.device,
            bounds=(0., 1.0),     # change if you need a different box
            n_init=iters_list[1] if len(iters_list) > 1 else 32,
            n_iter=iters_list[2] if len(iters_list) > 2 else 128,
            q=1,
            seed=seed,
        )
        Z = torch.nn.Parameter(bo_Z.detach().clone())  # lock in best Z
        cost_history = bo_hist

    elif routine == "BO_ZnHP":
        # Build a joint objective that evaluates cost at (Z, hp)
        def _eval_cost_at_Zhp(Z_current, _hp_unused):
            # hp is already set in the manager by the BO loop before calling this
            return get_cost_ACnew2(
                Z_current, X, Xplus, ObsManager,
                nT=nTrain, lambda1=lam1, lambda2=lam2, lambda3=lam3
            )

        # 1) Flatten current hps and build bounds aligned to that flat vector
        hp0_flat32, hp_spec = _mgr_pack_params_flat(ObsManager)
        if hp0_flat32.numel() == 0:
            raise ValueError(
                "GPObservablesManager has no parameters to optimize.")

        # Per-dimension hp bounds [2, dh]; μ bounds are data-driven from X
        hp_bounds_mat_d, hp_spec2 = _mgr_build_hp_bounds(ObsManager, X)
        assert hp_spec2 == hp_spec, "Internal spec mismatch for hp bounds."

        # 2) Run joint BO over Z and hp (we reuse your existing helper)
        Z_opt32, hp_opt32, bo_hist = _bo_optimize_Z_and_hp(
            Z_shape=(Z.shape[0], Z.shape[1]),
            eval_cost_fn=_eval_cost_at_Zhp,
            get_hp=lambda: _mgr_pack_params_flat(ObsManager)[0],
            set_hp=lambda hp_vec32: _mgr_unpack_and_set_params(
                ObsManager, hp_vec32, hp_spec),
            device=SimData.device,
            # adjust if you want a different box for Z
            Z_bounds=(0.0, 1.0),
            hp_bounds_mat=hp_bounds_mat_d,  # [2, dh] double
            n_init=iters_list[1] if len(iters_list) > 1 else 32,
            n_iter=iters_list[2] if len(iters_list) > 2 else 128,
            q=1,
            seed=seed,
        )
        # 3) Lock in best Z and hp
        # keep float32 for your gpk
        Z = torch.nn.Parameter(Z_opt32.detach().clone())
        _mgr_unpack_and_set_params(
            ObsManager, hp_opt32.detach().clone(), hp_spec)
        cost_history = bo_hist["per_iter_best"]

    elif routine == "BO_hp_and_GD_Z":
        # Configure: iters_list = [_, n_outer_iter, q_batch, inner_steps]
        n_outer_iter = iters_list[1] if len(
            iters_list) > 1 else 10    # default 10
        q_batch = iters_list[2] if len(iters_list) > 2 else 20    # default 20
        inner_steps = iters_list[3] if len(
            iters_list) > 3 else 200   # default 200

        # Build hp bounds from current manager + data-driven μ bounds
        hp0_flat32, hp_spec = _mgr_pack_params_flat(ObsManager)
        if hp0_flat32.numel() == 0:
            raise ValueError("No hyperparameters present to optimize.")
        hp_bounds_mat, hp_spec2 = _mgr_build_hp_bounds(ObsManager, X)
        assert hp_spec == hp_spec2, "HP shape spec mismatch."

        # Run BO (hp) ⟶ inner GD (Z) routine
        Z_best32, hp_best32, hist = _bo_hp_outer_gdZ_inner(
            Z_init=Z.detach().clone(),
            ObsManager=ObsManager,
            X=X, Xplus=Xplus, nT=nTrain,
            hp_bounds_mat=hp_bounds_mat,
            lam1=lam1, lam2=lam2, lam3=lam3,
            device=SimData.device,
            n_outer_iter=n_outer_iter,
            q_batch=q_batch,
            inner_steps=inner_steps,
            lr_Z=learn_rate,
            seed=seed
        )

        # Lock in the best Z and hyperparameters
        Z = torch.nn.Parameter(Z_best32.detach().clone())
        _mgr_unpack_and_set_params(
            ObsManager, hp_best32.detach().clone(), hp_spec)

        # Keep cost history (best-so-far per evaluation)
        cost_history = hist["per_iter_best"]

    else:
        raise ValueError(f"Invalid routine: {routine}")

    # Plot Cost History
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('log(Cost)', color=color)
    ax1.plot(torch.log10(torch.abs(torch.tensor(cost_history))), color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Cost', color=color)
    ax2.plot(cost_history, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()

    # === Retrain GPs at optimal Z & (optionally) optimize hp ===
    optimal_Z = Z.detach()
    for i in range(p):
        ObsManager.train_observable(i, Xtrain, optimal_Z[:, i])

    ObsManager.optimize_hyperparameters(
        opt_mu=False, opt_sigma=True, max_iter=100)

    # === Koopman A, C ===
    ObsList = [i for i in range(p)]
    A, C = gpk.getKoopman(ObsManager, ObsList, Xall, nTrain, stateAug=False)

    # === Simulate & evaluate ===
    #   Train split (offset 0), Test split (offset nTrain)
    XhatTrain, XcvTrain, TrainNRMSE = gpk.sim_and_eval(
        ObsManager, A, C, ICsetTrain, SimData, traj_offset=0)
    XhatTest,  XcvTest,  TestNRMSE = gpk.sim_and_eval(
        ObsManager, A, C, ICsetTest,  SimData, traj_offset=nTrain)

    # === Package results ===
    return {
        "ObsManager": ObsManager,
        "A": A, "C": C,
        "ICsetTrain": ICsetTrain.detach().cpu(),
        "ICsetTest":  ICsetTest.detach().cpu(),
        "Train": {
            "Xhat": XhatTrain,           # (nTrain, n, N)
            "Xcv":  XcvTrain,            # (nTrain, n, n, N)
            "NRMSE": TrainNRMSE          # (nTrain, n)
        },
        "Test": {
            "Xhat": XhatTest,            # (nTest, n, N)
            "Xcv":  XcvTest,             # (nTest, n, n, N)
            "NRMSE": TestNRMSE           # (nTest, n)
        },
        "history": {
            "cost": torch.tensor(cost_history).cpu()
        }
    }


if __name__ == "__main__":
    system_name = 'Simple Pendulum'
    train_frac, test_frac = 0.4, 0.2
    clip = None
    lifted_order = 6
    noise_type = 'gaussian'
    iters_list = [0, 10, 10, 200]
    routine = "BO_ZnHP" # BO_Z | BO_ZnHP | BO_hp_and_GD_Z
    # 1) Load + normalize
    SimData_raw, ts, num_traj, N, nTrain, nTest = gpk.load_SimData(
        system_name, train_frac, test_frac, clip=clip)
    SimData_clean, mu_vec, std_vec = gpk.normalize_data(
        SimData_raw, nTrain, N)

    # 2) Noise
    SimData = gpk.add_noise(SimData_clean, noise_type=noise_type,
                            intensity=0., seed=1234)

    print(f'==== Starting iGPK Model Identification ====')
    t0 = time.perf_counter()
    results = get_iGPK(SimData, nTrain, nTest, lifted_order,
                       iters_list, learn_rate=0.05,
                       opt_weights=[1.0, 1.0, 1.0], routine=routine,
                       train_method="Horizon")
    t_BO = time.perf_counter() - t0
    print(
        f'Bayesian Optimization with {iters_list[1]}-iterations, {iters_list[2]}-samples, finished in {t_BO:.2f} seconds.')


    # unpack iGPK
    A_igpk, C_igpk = results["A"], results["C"]
    # ICsetTrain, ICsetTest = results["ICsetTrain"], results["ICsetTest"]
    XhatTrain, XcvhatTrain, TrainNRMSE = results["Train"][
        "Xhat"], results["Train"]["Xcv"], results["Train"]["NRMSE"]
    XhatTest,  XcvhatTest,  TestNRMSE = results["Test"][
        "Xhat"],  results["Test"]["Xcv"],  results["Test"]["NRMSE"]

    gpk.plot_eigen(A_igpk)

    gpk.plot_NRMSE_metrics([TrainNRMSE*100], [TestNRMSE*100], ['iGPK-BO'])

    # 6) indices + timebase
    idx_trainMIN = torch.argmin(TrainNRMSE.mean(dim=1))
    idx_testMIN = torch.argmin(TestNRMSE.mean(dim=1))
    idx_testMAX = torch.argmax(TestNRMSE.mean(dim=1))
    # same shape you used (see your callsite) :contentReference[oaicite:11]{index=11}
    time_arr = torch.arange(0., ts * (SimData.shape[2] - 1), ts)

    # 7) pack models for overlay plot
    models = [
        {"name": "iGPK-BO", "train": {"Xhat": XhatTrain, "Xcvhat": XcvhatTrain},
            "test": {"Xhat": XhatTest, "Xcvhat": XcvhatTest}}
    ]

    # a) 3 trajectory overlays
    for (which, idx, split, sim_offset, suffix) in [
        ("best-train", idx_trainMIN, "train", 0,         "Best_Train"),
        ("best-test",  idx_testMIN,  "test",  nTrain,    "Best_Test"),
        ("worst-test", idx_testMAX,  "test",  nTrain,    "Worst_Test"),
    ]:
        gpk.compare_model_predictions(
            time=time_arr, models=models, SimData=SimData, idx=idx, N=(
                SimData.shape[2]-1),
            system_name=system_name, title_suffix=suffix, split=split, sim_offset=sim_offset,
            compare_to="SimData_clean", SimData_clean=SimData_clean, sigma=1.0
        )

    plt.show()
