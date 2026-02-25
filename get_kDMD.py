import numpy as np
import torch


def get_kDMD(
    SimData: torch.Tensor,
    nTrain: int,
    nTest: int,
    lifting_order: int,
    device: torch.device | None = None,
):
    """
    Kernel Dynamic Mode Decomposition (KDMD) via pykoopman.

    Args:
        SimData: torch.Tensor, shape (num_trajectories, state_dim, num_steps+1)
        nTrain:  number of trajectories used for training (taken from the front)
        nTest:   number of trajectories used for testing  (taken right after the training block)
        lifting_order: used here as KDMD svd_rank (retained components / effective rank)
        device: torch device for outputs (CPU by default)

    Returns:
        results dict with:
          results["A"], results["C"]
          results["Train"]["Xhat"], results["Train"]["NRMSE"]
          results["Test"]["Xhat"],  results["Test"]["NRMSE"]

    Notes:
      - Requires: pykoopman, scikit-learn
      - KDMD API usage follows pykoopman tutorials and repo docs. :contentReference[oaicite:0]{index=0}
    """
    if device is None:
        device = torch.device("cpu")

    # ---------- sanity checks ----------
    if SimData.ndim != 3:
        raise ValueError(
            f"SimData must be 3D (nTraj, n, T+1). Got {tuple(SimData.shape)}")
    nTraj, n, Tp1 = SimData.shape
    if nTrain + nTest > nTraj:
        raise ValueError(
            f"nTrain+nTest={nTrain+nTest} exceeds num_trajectories={nTraj}")

    T = Tp1 - 1
    if T < 1:
        raise ValueError("Need at least 2 time samples (num_steps+1 >= 2).")

    # ---------- import pykoopman bits ----------
    try:
        import pykoopman as pk
        from pykoopman.regression import KDMD
        from sklearn.gaussian_process.kernels import RBF
    except Exception as e:
        raise ImportError(
            "Missing dependency. Install with something like:\n"
            "  pip install pykoopman scikit-learn\n"
            f"Original import error: {repr(e)}"
        )

    # ---------- helpers ----------
    def _stack_snapshot_pairs(batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        batch: (nB, n, T+1)
        Returns:
          X: (nB*T, n), Y: (nB*T, n)
        """
        # X_k = x(t), Y_k = x(t+1), stack across trajectories and time
        X = batch[:, :, :-1].permute(0, 2, 1).reshape(-1,
                                                      n).detach().cpu().numpy()
        Y = batch[:, :,  1:].permute(
            0, 2, 1).reshape(-1, n).detach().cpu().numpy()
        return X, Y

    def _rollout(model, x0: np.ndarray, steps: int) -> np.ndarray:
        """
        model: fitted pk.Koopman
        x0: (n,) numpy
        steps: number of steps to predict (T)
        Returns trajectory: (steps+1, n)
        """
        traj = np.zeros((steps + 1, n), dtype=float)
        traj[0] = x0
        x = x0.reshape(1, -1)
        for k in range(steps):
            # returns shape (1, n) for Koopman models in pykoopman tutorials
            x = model.predict(x)
            traj[k + 1] = x.reshape(-1)
        return traj

    def _nrmse_per_traj_state(x_true: torch.Tensor, x_hat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        x_true, x_hat: (nB, n, T+1)
        returns: (nB, n) NRMSE per trajectory per state
        Definition: RMSE / RMS(true)
        """
        err = x_true - x_hat
        rmse = torch.sqrt(torch.mean(err**2, dim=-1)
                          )                     # (nB, n)
        denom = torch.sqrt(torch.mean(x_true**2, dim=-1)
                           ).clamp_min(eps)  # (nB, n)
        return rmse / denom

    # ---------- split train/test ----------
    train_batch = SimData[:nTrain]
    test_batch = SimData[nTrain:nTrain + nTest]

    # ---------- fit KDMD on stacked (X,Y) ----------
    Xtr, Ytr = _stack_snapshot_pairs(train_batch)

    # KDMD setup (mirrors tutorial usage: KDMD(svd_rank=..., kernel=RBF(...), ...)). :contentReference[oaicite:1]{index=1}
    regressor = KDMD(
        svd_rank=int(lifting_order),
        kernel=RBF(length_scale=1.0),
        forward_backward=False,
        tikhonov_regularization=1e-12,
    )
    model = pk.Koopman(regressor=regressor)
    model.fit(Xtr, Ytr)

    # ---------- extract A, C (best-effort across pykoopman versions) ----------
    # In pykoopman, the Koopman matrix is typically exposed; "C" varies by model/observable choice.
    A = None
    for attr in ["koopman_matrix", "A", "K"]:
        if hasattr(model, attr):
            A = getattr(model, attr)
            break
    if A is None and hasattr(model, "regressor_") and hasattr(model.regressor_, "koopman_matrix"):
        A = model.regressor_.koopman_matrix

    # "C" is commonly the modes matrix when the state is part of the observable library.
    # If unavailable, return identity as a conservative placeholder in the state space.
    C = None
    for attr in ["modes", "C", "output_matrix"]:
        if hasattr(model, attr):
            C = getattr(model, attr)
            break
    if C is None:
        C = np.eye(n)

    # ---------- multi-step rollouts for Train/Test ----------
    def _predict_block(batch: torch.Tensor) -> torch.Tensor:
        nB = batch.shape[0]
        Xhat = torch.zeros((nB, n, Tp1), dtype=torch.float32)
        for i in range(nB):
            x0 = batch[i, :, 0].detach().cpu().numpy()
            traj_hat = _rollout(model, x0=x0, steps=T)  # (T+1, n)
            Xhat[i] = torch.from_numpy(traj_hat.T).to(torch.float32)
        return Xhat

    Xhat_train = _predict_block(train_batch)
    Xhat_test = _predict_block(test_batch)

    # ---------- NRMSE ----------
    TrainNRMSE = _nrmse_per_traj_state(train_batch.to(
        torch.float32), Xhat_train)  # (nTrain, n)
    TestNRMSE = _nrmse_per_traj_state(test_batch.to(
        torch.float32),  Xhat_test)   # (nTest,  n)

    # ---------- pack outputs (move tensors to requested device) ----------
    results = {
        "A": torch.tensor(np.array(A), dtype=torch.float32, device=device),
        "C": torch.tensor(np.array(C), dtype=torch.float32, device=device),
        "Train": {
            "Xhat": Xhat_train.to(device),
            "NRMSE": TrainNRMSE.to(device),
        },
        "Test": {
            "Xhat": Xhat_test.to(device),
            "NRMSE": TestNRMSE.to(device),
        },
    }
    return results


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
    results = get_kDMD(
        SimData=SimData_norm,
        nTrain=nTrain,
        nTest=nTest,
        lifting_order=lifting_order,
        device=device
    )

    XhatTrain = results["Train"]["Xhat"]
    TrainNRMSE = results["Train"]["NRMSE"]

    XhatTest = results["Test"]["Xhat"]
    TestNRMSE = results["Test"]["NRMSE"]

    # -----------------------------
    # Diagnostics
    # -----------------------------
    print("\n===== Kernel Koopman Results =====")

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
