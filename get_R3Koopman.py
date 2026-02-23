import numpy as np
import torch


def get_R3Koopman(SimData, nTrain, nTest, lifting_order, device="cpu"):
    """
    Kernel Koopman learning using kooplearn.kernel.KernelRidge.

    SimData: torch.Tensor, shape (num_trajectories, state_dim, num_steps+1)
    nTrain:  number of trajectories used for training (taken from the front)
    nTest:   number of trajectories used for testing  (taken right after the training block)
    lifting_order: used here as kooplearn's n_components (rank / retained components)
    device: torch device for inputs/outputs (kooplearn itself runs on CPU numpy)

    Returns dict with:
      results["A"], results["C"]
      results["Train"]["Xhat"], results["Train"]["Xcv"], results["Train"]["NRMSE"]
      results["Test"]["Xhat"],  results["Test"]["Xcv"],  results["Test"]["NRMSE"]

    Notes / caveat (important):
    - kooplearn's fit() API expects ONE trajectory array of shape (n_samples, n_features). It does not
      (as far as the public docs show) accept a list of disjoint trajectories in a single fit call.
      This implementation concatenates the training trajectories in time, which introduces (nTrain-1)
      artificial “boundary transitions” between trajectories.
      In practice, if N is large, this tends to be negligible; if you care, we can switch to a
      “fit-per-trajectory then aggregate” strategy or implement a pairwise estimator.
    - kooplearn does not expose predictive covariance for KernelRidge predictions, so Xcv is None.
    """

    # ----------------------------
    # Validate + move to device
    # ----------------------------
    if not isinstance(SimData, torch.Tensor):
        raise TypeError(
            "SimData must be a torch.Tensor of shape (num_trajectories, state_dim, num_steps+1).")

    SimData = SimData.float().to(device)

    n_traj, n, Tp1 = SimData.shape
    N = Tp1 - 1
    p = int(lifting_order)
    lag_time = 1

    if nTrain < 1 or nTrain > n_traj:
        raise ValueError(f"nTrain must be in [1, {n_traj}], got {nTrain}.")
    if nTest < 1:
        raise ValueError("nTest must be >= 1.")
    if (nTrain + nTest) > n_traj:
        raise ValueError(
            f"Need nTrain+nTest <= num_trajectories ({n_traj}), got {nTrain+nTest}.")
    if N < 1:
        raise ValueError(f"Need num_steps >= 1, got N={N}.")

    # ----------------------------
    # Build concatenated train trajectory for kooplearn (CPU numpy)
    #   SimData[j] is (n, N+1)  -> transpose to (N+1, n)
    #   concat along time -> (nTrain*(N+1), n)
    # ----------------------------
    Xtrain_traj_torch = torch.cat(
        # (nTrain*(N+1), n)
        [SimData[j, :, :].T for j in range(nTrain)], dim=0)
    Xtrain_traj = Xtrain_traj_torch.detach().cpu().numpy()

    # ----------------------------
    # Fit kooplearn KernelRidge
    # ----------------------------
    # kooplearn docs: fit expects (n_samples, n_features)
    from kooplearn.kernel import KernelRidge

    model = KernelRidge(
        n_components=p,
        lag_time=lag_time,
        reduced_rank=True,
        kernel="rbf",
        alpha=1e-6,
        random_state=0,
    )
    model.fit(Xtrain_traj)

    # ----------------------------
    # Get eigen-decomposition and build a compact (A, C) surrogate:
    #   z_{k+1} = A z_k ,   x_k ≈ C z_k
    # using right eigenfunctions evaluated on training points.
    # ----------------------------
    eigvals, right_eigs = model.eig(
        eval_right_on=Xtrain_traj, eval_left_on=None)
    Z = np.asarray(right_eigs)  # (nTrain*(N+1), p) potentially complex

    A = np.diag(np.asarray(eigvals))  # (p, p)

    # Least squares: Z @ Ct ≈ Xtrain_traj  => Ct is (p, n)
    Ct, *_ = np.linalg.lstsq(Z, Xtrain_traj, rcond=None)
    C = Ct.T  # (n, p)

    # If data is real, drop tiny imaginary components
    if np.isrealobj(Xtrain_traj):
        A = np.real_if_close(A, tol=1e6)
        C = np.real_if_close(C, tol=1e6)

    # ----------------------------
    # Rollout helper (uses kooplearn predict; returns full trajectory)
    # ----------------------------
    def rollout_from_x0(x0_np, horizon):
        # x0_np: (n,)
        out = np.zeros((horizon + 1, n), dtype=float)
        out[0] = x0_np
        x0_batch = x0_np.reshape(1, -1)
        for t in range(1, horizon + 1):
            # returns final predicted state at step t
            xt = model.predict(x0_batch, n_steps=t, observable=False)
            out[t] = np.asarray(xt).reshape(-1)
        return out  # (horizon+1, n)

    # ----------------------------
    # NRMSE helper (normalized by mean-centered signal energy)
    # ----------------------------
    def nrmse_per_state(true, pred, eps=1e-12):
        """
        true, pred: (T, n)
        Returns: (n,)  NRMSE per state
        """
        true = np.asarray(true)
        pred = np.asarray(pred)

        err = np.sqrt(np.mean((true - pred)**2, axis=0))          # (n,)
        denom = np.sqrt(np.mean((true - true.mean(axis=0))**2, axis=0))  # (n,)
        return err / (denom + eps)

    # ----------------------------
    # Predict on TRAIN trajectories (separately) and aggregate
    # Shapes returned to you: (nTraj, n, N+1) as torch on device
    # ----------------------------
    XhatTrain_list = []
    train_nrmse_list = []
    for j in range(nTrain):
        x_true = SimData[j, :, :].T.detach().cpu().numpy()     # (N+1, n)
        x0 = x_true[0]
        x_pred = rollout_from_x0(x0, horizon=N)                # (N+1, n)
        XhatTrain_list.append(torch.from_numpy(x_pred.T))      # (n, N+1)
        train_nrmse_list.append(torch.from_numpy(
            nrmse_per_state(x_true, x_pred)))

    XhatTrain = torch.stack(XhatTrain_list, dim=0).to(
        device)  # (nTrain, n, N+1)
    TrainNRMSE = torch.stack(train_nrmse_list, dim=0).to(device)  # (nTrain, n)

    # ----------------------------
    # Predict on TEST trajectories (block right after training trajectories)
    # ----------------------------
    XhatTest_list = []
    test_nrmse_list = []
    for j in range(nTrain, nTrain + nTest):
        x_true = SimData[j, :, :].T.detach().cpu().numpy()     # (N+1, n)
        x0 = x_true[0]
        x_pred = rollout_from_x0(x0, horizon=N)                # (N+1, n)
        XhatTest_list.append(torch.from_numpy(x_pred.T))       # (n, N+1)
        test_nrmse_list.append(torch.from_numpy(
            nrmse_per_state(x_true, x_pred)))

    XhatTest = torch.stack(XhatTest_list, dim=0).to(
        device)    # (nTest, n, N+1)
    TestNRMSE = torch.stack(test_nrmse_list, dim=0).to(device)  # (nTest, n)

    # A, C to torch on device (keep them as float if real)
    A_torch = torch.from_numpy(np.asarray(A)).to(device)
    C_torch = torch.from_numpy(np.asarray(C)).to(device)

    results = {
        "A": A_torch,
        "C": C_torch,
        "Train": {"Xhat": XhatTrain, "Xcv": None, "NRMSE": TrainNRMSE},
        "Test":  {"Xhat": XhatTest,  "Xcv": None, "NRMSE": TestNRMSE},
        "model": model,  # keep the fitted kooplearn object if you want eig/modes later
    }
    return results


# ==========================================================
# Small Test Snippet for get_KKR
# ==========================================================
if __name__ == "__main__":

    import math
    import torch
    import GPKoopman as gpk

    # -----------------------------
    # Configuration
    # -----------------------------
    system_name = "Inhibited Predator-Prey"   # change if needed
    trainFrac = 0.5
    testFrac = 0.5
    clip = None               # optionally limit horizon
    lifting_order = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nRunning KKR test on system: {system_name}")
    print(f"Using device: {device}")

    # -----------------------------
    # Load + Normalize
    # -----------------------------
    SimData_raw, ts, num_traj, N, nTrain, nTest = gpk.load_SimData(
        system_name, trainFrac, testFrac, clip
    )

    SimData_norm, mu_vec, std_vec = gpk.normalize_data(
        SimData_raw, nTrain, N
    )

    print(f"Num trajectories: {num_traj}")
    print(f"Train trajectories: {nTrain}")
    print(f"Test trajectories:  {nTest}")
    print(f"State dimension:    {SimData_norm.shape[1]}")
    print(f"Horizon (N):        {N}")

    # -----------------------------
    # Run Kernel Koopman
    # -----------------------------
    results = get_R3Koopman(
        SimData=SimData_norm,
        nTrain=nTrain,
        nTest=nTest,
        lifting_order=lifting_order,
        device=device
    )

    A, C = results["A"], results["C"]

    XhatTrain = results["Train"]["Xhat"]
    TrainNRMSE = results["Train"]["NRMSE"]

    XhatTest = results["Test"]["Xhat"]
    TestNRMSE = results["Test"]["NRMSE"]

    # -----------------------------
    # Diagnostics
    # -----------------------------
    print("\n===== Kernel Koopman Results =====")
    print(f"A shape: {A.shape}")
    print(f"C shape: {C.shape}")

    print(f"Train prediction shape: {XhatTrain.shape}")
    print(f"Test  prediction shape: {XhatTest.shape}")

    print("Train per-state mean NRMSE:", TrainNRMSE.mean(dim=0))
    print("Test  per-state mean NRMSE:", TestNRMSE.mean(dim=0))

    # Spectral sanity check
    if A.is_complex():
        eigvals = torch.linalg.eigvals(A).cpu().numpy()
    else:
        eigvals = torch.linalg.eigvals(A.float()).cpu().numpy()

    spectral_radius = max(abs(eigvals))
    print(f"\nSpectral radius of A: {spectral_radius:.6f}")

    print("\nTest completed successfully.\n")
