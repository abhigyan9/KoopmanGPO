import itertools
import torch
from .autonomous import sim_LTI
from .utilities import get_kmeans, sim_and_eval
from .GPObs import GPObservablesManager
from typing import Tuple
from __future__ import annotations

# Extended Dynamic Mode Decomposition

# eDMD based on Polynomial Combination Observables


def generate_basis(x, degree):
    """
    Generate a polynomial basis for a 1D state vector x up to a given degree.

    Args:
        x (torch.Tensor): A 1D tensor of shape (d,).
        degree (int): Maximum degree of the polynomial basis.

    Returns:
        torch.Tensor: A 1D tensor containing the computed polynomial basis.
    """
    # Ensure x is 1D (if it's (d,1) or similar, squeeze it)
    if x.dim() > 1:
        x = x.squeeze()

    # Use the number of elements in the state vector, not the total elements.
    d = x.shape[0]
    # constant term
    basis = [torch.tensor(1.0, dtype=x.dtype, device=x.device)]

    for deg in range(1, degree + 1):
        for indices in itertools.combinations_with_replacement(range(d), deg):
            term = torch.prod(x[list(indices)])
            basis.append(term)

    return torch.stack(basis)


def generate_basis_batch(X, degree):
    """
    Generate a polynomial basis for each state vector in a batch.

    Args:
        X (torch.Tensor): A 2D tensor with shape (state_dim, num_states).
        degree (int): Maximum degree of the polynomial basis.

    Returns:
        torch.Tensor: A 2D tensor where each column is the polynomial basis of the corresponding state vector.
    """
    state_dim, num_states = X.shape
    # Compute the size of the basis by generating it once
    single_basis = generate_basis(X[:, 0], degree)
    basis_size = single_basis.numel()

    # Pre-allocate tensor for efficiency
    basis_tensor = torch.empty(
        basis_size, num_states, dtype=X.dtype, device=X.device)

    for i in range(num_states):
        basis_tensor[:, i] = generate_basis(X[:, i], degree)

    return basis_tensor


def eDMD_poly(SimData, nTrain, nTest, poly_deg=1):

    SimData = SimData.float()
    n, N = SimData.shape[1], SimData.shape[2] - 1

    X = torch.cat([SimData[j, :, 0:N] for j in range(nTrain)],
                  dim=1)       # Concatenated Data matrix
    # Time-shifted Data matrix
    Xplus = torch.cat([SimData[j, :, 1:] for j in range(nTrain)], dim=1)
    ICsetTrain = torch.cat([SimData[j, :, 0].view(n, 1)
                           for j in range(nTrain)], dim=1)
    ICsetTest = torch.cat([SimData[j, :, 0].view(n, 1)
                          for j in range(nTrain, nTrain + nTest)], dim=1)

    # Generate Polynomial Basis Function
    def phi_batch(X): return generate_basis_batch(X, poly_deg)

    # Lift state with generated polynomial basis functions
    phi_x = phi_batch(X)
    phi_xplus = phi_batch(Xplus)

    # Compute eDMD matrices
    A_edmd = phi_xplus @ torch.linalg.pinv(phi_x)
    C_edmd = X @ torch.linalg.pinv(phi_x)
    p = C_edmd.shape[1]

    # ---------- Training ----------
    ZedTrain = torch.empty((nTrain, p, N))
    XedTrain = torch.empty((nTrain, n, N))
    TrainNRMSE_eDMD = torch.empty((nTrain, n))

    for j in range(nTrain):
        ZedTrain[j, :, 0] = phi_batch(ICsetTrain[:, j].view(n, 1)).view(p,)
        ZedTrain[j, :, :], XedTrain[j, :, :] = sim_LTI(
            ZedTrain[j, :, 0], A_edmd, C_edmd, num_steps=N, ts=None, x0cv=None
        )

        errors = XedTrain[j] - SimData[j, :, :N]
        rmse = torch.sqrt(torch.mean(errors**2, dim=1))

        true_vals = SimData[j, :, :N]
        range_vals = true_vals.max(dim=1).values - true_vals.min(dim=1).values
        range_vals = torch.where(
            range_vals == 0, torch.ones_like(range_vals), range_vals)

        TrainNRMSE_eDMD[j] = rmse / range_vals

    # ---------- Testing ----------
    ZedTest = torch.empty((nTest, p, N))
    XedTest = torch.empty((nTest, n, N))
    TestNRMSE_eDMD = torch.empty((nTest, n))

    for j in range(nTest):
        ZedTest[j, :, 0] = phi_batch(ICsetTest[:, j].view(n, 1)).view(p,)
        ZedTest[j, :, :], XedTest[j, :, :] = sim_LTI(
            ZedTest[j, :, 0], A_edmd, C_edmd, num_steps=N, ts=None, x0cv=None
        )

        errors = XedTest[j] - SimData[nTrain+j, :, :N]
        rmse = torch.sqrt(torch.mean(errors**2, dim=1))

        true_vals = SimData[nTrain+j, :, :N]
        range_vals = true_vals.max(dim=1).values - true_vals.min(dim=1).values
        range_vals = torch.where(
            range_vals == 0, torch.ones_like(range_vals), range_vals)

        TestNRMSE_eDMD[j] = rmse / range_vals

    XedTrain, XedTest = XedTrain.detach(), XedTest.detach()
    TestNRMSE_eDMD, TrainNRMSE_eDMD = TestNRMSE_eDMD.detach(), TrainNRMSE_eDMD.detach()

    return A_edmd, C_edmd, XedTrain, XedTest, TrainNRMSE_eDMD, TestNRMSE_eDMD


# eDMD based on Radial Basis Function Observables


def rbf_observable(x, centers, width=None, rbf_type='gaussian', state_aug=False):
    """
    Computes RBF values for a set of points x given multiple centers, with a choice
    between a Gaussian RBF or a thin-plate spline RBF.

    Args:
        x (torch.Tensor): Input points of shape (d, samples), where d is the dimension 
                          and samples is the number of points.
        centers (torch.Tensor): RBF centers of shape (d, m), where m is the number of RBFs.
        width (float or torch.Tensor, optional): For Gaussian RBF, either a scalar or a 1D tensor 
                          of length m specifying the width for each RBF. This is ignored for 
                          the thin-plate spline RBF.
        rbf_type (str): Type of RBF to compute. Options are 'gaussian' or 'thin_plate'.
                        Default is 'gaussian'.

    Returns:
        torch.Tensor: RBF values of shape (m, samples), where each row corresponds to one RBF.
    """
    # Compute pairwise differences between each center and each sample.
    # x has shape (d, samples), centers has shape (d, m).
    # After unsqueezing:
    #   x -> (d, 1, samples)
    #   centers -> (d, m, 1)
    # Their difference broadcasts to shape (d, m, samples)
    diff = x.unsqueeze(1) - centers.unsqueeze(2)

    # Compute squared Euclidean distance for each center-sample pair.
    # Resulting shape: (m, samples)
    dist_sq = torch.sum(diff**2, dim=0)

    if rbf_type.lower() == 'gaussian':
        if width is None:
            raise ValueError("A width must be provided for the Gaussian RBF.")
        # Convert width to a tensor if necessary.
        if not torch.is_tensor(width):
            width = torch.tensor(
                width, dtype=dist_sq.dtype, device=dist_sq.device)

        # Reshape width for proper broadcasting: (m, 1) if it's a 1D tensor.
        if width.ndim == 0:
            widths = width
        elif width.ndim == 1:
            widths = width.view(-1, 1)
        else:
            raise ValueError(
                "width must be a scalar or a 1D tensor of length m.")

        # Compute the Gaussian RBF: exp(-||x - c||^2 / (2 * width^2))
        phi = torch.exp(-dist_sq / (2 * widths**2))

    elif rbf_type.lower() == 'thin_plate':
        # For the thin-plate spline, ignore width.
        # Add a small constant to avoid log(0)
        r = torch.sqrt(dist_sq + 1e-4)
        # Compute the thin-plate spline RBF: ||x - c||^2 * log(||x - c||)
        phi = dist_sq * torch.log(r)

    else:
        raise ValueError("rbf_type must be either 'gaussian' or 'thin_plate'")

    if state_aug:
        return torch.vstack((x, phi))
    else:
        return phi


def eDMD_RBF_kmeans(SimData, nTrain, nTest, num_centers, width=None, rbf_type='gaussian', state_aug=False):
    """
    Extended Dynamic Mode Decomposition (eDMD) using RBF observables defined via rbf_observable.

    Args:
        SimData (torch.Tensor): Simulation data of shape (num_trajectories, state_dim, num_time_steps).
        nTrain (int): Number of trajectories used for training.
        nTest (int): Number of trajectories used for testing.
        centers (torch.Tensor): RBF centers of shape (state_dim, m) where m is the number of RBFs.
        width (float or torch.Tensor, optional): For Gaussian RBF, either a scalar or a 1D tensor of length m.
            Ignored for thin-plate spline RBF.
        rbf_type (str): Type of RBF to use: 'gaussian' or 'thin_plate'.

    Returns:
        A_edmd (torch.Tensor): The estimated system matrix in the lifted space.
        C_edmd (torch.Tensor): The reconstruction matrix from the lifted space to the state space.
        XedTrain (torch.Tensor): Predicted training trajectories.
        XedTest (torch.Tensor): Predicted testing trajectories.
        TrainRMSE_eDMD (torch.Tensor): RMSE for each state in each training trajectory.
        TestRMSE_eDMD (torch.Tensor): RMSE for each state in each testing trajectory.
    """
    SimData = SimData.float()
    n, N = SimData.shape[1], SimData.shape[2] - 1

    # Concatenate data for training: X and its time-shifted version Xplus.
    X = torch.cat([SimData[j, :, 0:N] for j in range(nTrain)], dim=1)
    Xplus = torch.cat([SimData[j, :, 1:] for j in range(nTrain)], dim=1)
    # Initial conditions for training and testing.
    ICsetTrain = torch.cat([SimData[j, :, 0].view(n, 1)
                           for j in range(nTrain)], dim=1)
    ICsetTest = torch.cat([SimData[j, :, 0].view(n, 1)
                          for j in range(nTrain, nTrain + nTest)], dim=1)

    centers = get_kmeans(X, num_centers)

    # Lift the data.
    # shape: (m, total training samples)
    phi_x = rbf_observable(X, centers, width, rbf_type, state_aug)
    phi_xplus = rbf_observable(Xplus, centers, width, rbf_type, state_aug)

    # Compute the eDMD matrices using a pseudo-inverse.
    A_edmd = phi_xplus @ torch.linalg.pinv(phi_x)
    p = A_edmd.shape[0]  # p is the number of lifted observables (typically m).
    if state_aug:
        C_edmd = torch.zeros((n, p))
        for i in range(n):
            C_edmd[i, i] = 1.
    else:
        C_edmd = X @ torch.linalg.pinv(phi_x)

    # Evaluate on the training set.
    ZedTrain = torch.empty((nTrain, p, N))
    XedTrain = torch.empty((nTrain, n, N))
    TrainRMSE_eDMD = torch.empty((nTrain, n))

    for j in range(nTrain):
        # Lift the initial condition for the j-th training trajectory.
        ZedTrain[j, :, 0] = rbf_observable(ICsetTrain[:, j].view(
            n, 1), centers, width, rbf_type, state_aug).view(p,)
        # Simulate the lifted linear system.
        ZedTrain[j, :, :], XedTrain[j, :, :] = sim_LTI(ZedTrain[j, :, 0], A_edmd, C_edmd,
                                                       num_steps=N, ts=None, x0cv=None)
        # Compute RMSE on the j-th trajectory.
        TrainRMSE_eDMD[j, :] = torch.sqrt(torch.mean(
            (XedTrain[j, :, :] - SimData[j, :, :N])**2, dim=1))

    # Evaluate on the test set.
    ZedTest = torch.empty((nTest, p, N))
    XedTest = torch.empty((nTest, n, N))
    TestRMSE_eDMD = torch.empty((nTest, n))

    for j in range(nTest):
        ZedTest[j, :, 0] = rbf_observable(ICsetTest[:, j].view(
            n, 1), centers, width, rbf_type, state_aug).view(p,)
        ZedTest[j, :, :], XedTest[j, :, :] = sim_LTI(ZedTest[j, :, 0], A_edmd, C_edmd,
                                                     num_steps=N, ts=None, x0cv=None)
        TestRMSE_eDMD[j, :] = torch.sqrt(torch.mean(
            (XedTest[j, :, :] - SimData[nTrain+j, :, :N])**2, dim=1))

    # Detach results.
    XedTrain, XedTest = XedTrain.detach(), XedTest.detach()
    TrainRMSE_eDMD, TestRMSE_eDMD = TrainRMSE_eDMD.detach(), TestRMSE_eDMD.detach()

    return A_edmd, C_edmd, XedTrain, XedTest, TrainRMSE_eDMD, TestRMSE_eDMD


# Subspace Identification driven GP-Koopman
# adapted from Holocomb and Bitmead 2017, Lian and Jones 2019, Loya et al. 2023
def SSID(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    delay: int,
    sys_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform subspace system identification on multi–trajectory data.

    Parameters
    ----------
    inputs : torch.Tensor
        Tensor of shape ``(nT, nu, N)`` containing input trajectories.  The
        first dimension indexes the trajectory, the second dimension indexes
        the input channels, and the third dimension indexes discrete time.
    outputs : torch.Tensor
        Tensor of shape ``(nT, ny, N)`` containing output trajectories.
    delay : int
        Number of time‐delayed embeddings to include when constructing the
        mosaic–Hankel matrices.  A value of ``0`` corresponds to using
        instantaneous outputs without delay; in this case the Hankel
        construction simply stacks the outputs at each time index.
    sys_dim : int
        Desired dimension of the lifted state.  This corresponds to the
        number of dominant singular values retained from the SVD and
        therefore determines the rank of the identified linear model.

    Returns
    -------
    A : torch.Tensor
        State transition matrix of shape ``(sys_dim, sys_dim)``.
    B : torch.Tensor
        Input matrix of shape ``(sys_dim, nu)``.
    C : torch.Tensor
        Output matrix of shape ``(ny, sys_dim)``.
    D : torch.Tensor
        Feedthrough matrix of shape ``(ny, nu)``.
    z0 : torch.Tensor
        Initial lifted states for each trajectory, shape ``(sys_dim, nT)``.

    Notes
    -----
    The function follows the multi–record subspace identification
    formulation presented in Holcomb & Bitmead (2017) and
    extended to Koopman operator identification by Loya et al. (2023).
    All matrix operations are carried out with `torch` and use the
    Moore–Penrose pseudoinverse for robustness.
    """

    # Basic dimensions
    nT, nu, N = inputs.shape  # number of trajectories, input dimension, time horizon
    _, ny, N_y = outputs.shape  # output dimension
    assert N_y == N, "inputs and outputs must have the same time horizon"

    device = inputs.device
    dtype = inputs.dtype

    # The number of columns per trajectory in the mosaic–Hankel matrices
    cols_per_traj = N - delay
    if cols_per_traj <= 0:
        raise ValueError(
            "The delay must be less than the trajectory length (N)."
        )

    # -------------------------------------------------------------------
    # Step 1: Construct mosaic–Hankel matrices for outputs (Y_l) and inputs (U_l)
    #         with (delay + 1) block rows and (N - delay) columns per trajectory.
    # Y_l has shape ((delay + 1) * ny, nT * cols_per_traj)
    # U_l has shape ((delay + 1) * nu, nT * cols_per_traj)
    blk_rows_y = (delay + 1) * ny
    blk_rows_u = (delay + 1) * nu
    total_cols = nT * cols_per_traj

    # Preallocate Hankel matrices
    Y_l = torch.empty((blk_rows_y, total_cols), dtype=dtype, device=device)
    U_l = torch.empty((blk_rows_u, total_cols), dtype=dtype, device=device)

    # Fill the Hankel matrices trajectory by trajectory
    for j in range(nT):
        # Starting column index for this trajectory in the global Hankel matrix
        col_offset = j * cols_per_traj
        # Build Hankel blocks for outputs and inputs
        # Each column corresponds to a time index k and contains stacked outputs/inputs
        # from k through k + delay
        for k in range(cols_per_traj):
            # Index of the column in the global matrix
            col_idx = col_offset + k
            # Collect delayed outputs for this trajectory and time
            if delay == 0:
                # No delay: just take the output at time k
                Y_l[:, col_idx] = outputs[j, :, k]
                U_l[:, col_idx] = inputs[j, :, k]
            else:
                # Stack outputs from k through k + delay into a column
                # Each block is of length ny (or nu for inputs)
                # Flatten the stacked blocks into a single vector
                Y_col_blocks = []
                U_col_blocks = []
                for d in range(delay + 1):
                    Y_col_blocks.append(outputs[j, :, k + d])  # shape (ny,)
                    U_col_blocks.append(inputs[j, :, k + d])  # shape (nu,)
                Y_l[:, col_idx] = torch.cat(Y_col_blocks, dim=0)
                U_l[:, col_idx] = torch.cat(U_col_blocks, dim=0)

    # -------------------------------------------------------------------
    # Step 2: Compute projection onto the orthogonal complement of the row space of U_l.
    #         This projection removes input influences when extracting the
    #         extended observability matrix.  The formula is
    #             Π = I - U_l.T @ (U_l @ U_l.T)^† @ U_l,
    #         where I is an identity matrix of size (total_cols x total_cols).
    # Note: U_l @ U_l.T is of size (blk_rows_u x blk_rows_u), which is
    # typically small compared to (total_cols x total_cols).  Using the
    # pseudoinverse ensures robustness when U_l is not full rank【83719996940448†L253-L264】.
    # Compute the pseudoinverse of U_l @ U_l.T
    UUT = U_l @ U_l.T
    UUT_pinv = torch.linalg.pinv(UUT)
    # Projection operator Π (total_cols x total_cols)
    # We compute U_l.T @ UUT_pinv @ U_l first to avoid forming a huge identity prematurely.
    # The result is symmetric idempotent and projects onto the row space of U_l.
    proj_rows = U_l.T @ UUT_pinv @ U_l  # shape (total_cols, total_cols)
    I_cols = torch.eye(total_cols, dtype=dtype, device=device)
    Pi_ul = I_cols - proj_rows

    # -------------------------------------------------------------------
    # Step 3: Perform SVD on the projected output Hankel matrix and extract A and C.
    #         The extended observability matrix Γ̂ is obtained as
    #             Γ̂ = Q1 Σ1^{1/2},
    #         where Q1 and Σ1 contain the leading singular vectors/values
    #         【83719996940448†L265-L299】.  The first ny rows of Γ̂ give C and
    #         shifted blocks give A via least squares【83719996940448†L293-L300】.
    Y_proj = Y_l @ Pi_ul  # shape (blk_rows_y, total_cols)
    # Compute SVD; we only need the left singular vectors and singular values.
    # full_matrices=False yields economic SVD, which is sufficient here.
    U_svd, S_svd, _ = torch.linalg.svd(Y_proj, full_matrices=False)
    # Take the leading sys_dim singular vectors/values
    if sys_dim > U_svd.shape[1]:
        raise ValueError(
            f"sys_dim={sys_dim} exceeds the maximum possible rank {U_svd.shape[1]}."
        )
    U_r = U_svd[:, :sys_dim]
    S_r = S_svd[:sys_dim]
    # Form Γ̂_r = U_r * sqrt(S_r)
    sqrt_S = torch.sqrt(S_r)
    Gamma_r = U_r * sqrt_S.view(1, -1)

    # Extract C from the first block of rows (ny rows)
    C = Gamma_r[:ny, :]

    # Extract A by solving Γ̂_r(0:(delay*ny)-1) * A ≈ Γ̂_r(ny:(delay+1)*ny - 1)
    if delay == 0:
        # Without delay, there is no shift to estimate A; use identity
        A = torch.eye(sys_dim, dtype=dtype, device=device)
    else:
        # Rows corresponding to the first delay blocks
        top_rows = Gamma_r[: delay * ny, :]
        # Rows corresponding to the next delay blocks
        bottom_rows = Gamma_r[ny: (delay + 1) * ny, :]
        # Solve for A in least squares sense
        A = torch.linalg.pinv(top_rows) @ bottom_rows

    # -------------------------------------------------------------------
    # Step 4: Regression to estimate B, D and z0 for each trajectory.
    #         We assemble a linear system of the form
    #             y_vec = Φ @ θ,
    #         where θ contains vec(B), vec(D), and the stacked z0_j's.
    #         See Eq. (15)–(25) in Holcomb & Bitmead (2017)【83719996940448†L304-L389】.
    # Precompute powers of A and the products C A^t for t = 0 … (N-1).
    # This avoids repeated matrix multiplications inside the loops.
    # A_power[t] = A^t
    A_powers = []
    I_sys = torch.eye(sys_dim, dtype=dtype, device=device)
    A_powers.append(I_sys)
    for t in range(1, N):
        A_powers.append(A_powers[-1] @ A)
    # C_A_powers[t] = C @ A^t (ny x sys_dim)
    C_A_powers = [C @ A_powers[t] for t in range(N)]

    # Number of unknown parameters
    # vec(B): sys_dim * nu
    # vec(D): ny * nu
    # z0 stacked: sys_dim * nT
    n_B = sys_dim * nu
    n_D = ny * nu
    n_z = sys_dim * nT
    n_params = n_B + n_D + n_z

    # Lists to accumulate rows of the regression matrix Φ and output vector
    # Each entry corresponds to ny rows (one per output channel) for a given (trajectory j, time t).
    reg_rows = []  # list of 2D tensors of shape (ny, n_params)
    y_rows = []    # list of 1D tensors of shape (ny,)

    # Identity matrix of size ny (for constructing phi_D)
    I_ny = torch.eye(ny, dtype=dtype, device=device)

    # Loop over trajectories and time indices to assemble the regression matrix
    for j in range(nT):
        # Precompute contributions of this trajectory's inputs for phi_B to avoid redundant loops.
        # For each time t and each past index k we need to accumulate C A^{t-1-k} scaled by u[j,:,k].
        # We'll compute phi_B_j[t] on the fly.
        for t in range(N):
            # Prepare phi_B_j[t]: shape (ny, sys_dim * nu)
            # Initialize to zeros
            phi_B = torch.zeros((ny, sys_dim * nu), dtype=dtype, device=device)
            # Sum over past inputs up to t-1
            for k in range(t):
                # C A^{t-1-k}
                M = C_A_powers[t - 1 - k]  # shape (ny, sys_dim)
                # Input at time k for trajectory j
                u_k = inputs[j, :, k]  # shape (nu,)
                # Accumulate contributions for each input channel
                # Each input channel q multiplies the block M
                for q in range(nu):
                    phi_B[:, q * sys_dim: (q + 1) * sys_dim] += M * u_k[q]

            # Prepare phi_D_j[t]: shape (ny, ny * nu) = (ny, nu * ny)
            # phi_D = u_t^T ⊗ I_p
            u_t = inputs[j, :, t]  # shape (nu,)
            phi_D = torch.kron(u_t.view(1, -1), I_ny)  # shape (ny, nu*ny)

            # Prepare phi_z0_j[t]: shape (ny, sys_dim * nT)
            # Only the block corresponding to trajectory j is non-zero
            phi_z0_block = torch.zeros(
                (ny, sys_dim * nT), dtype=dtype, device=device)
            phi_z0_block[
                :, j * sys_dim: (j + 1) * sys_dim
            ] = C_A_powers[t]  # shape (ny, sys_dim)

            # Concatenate phi_B, phi_D, phi_z0_block horizontally
            reg_row = torch.cat((phi_B, phi_D, phi_z0_block),
                                dim=1)  # (ny, n_params)
            reg_rows.append(reg_row)
            # Corresponding output y at time t
            y_rows.append(outputs[j, :, t])

    # Stack all rows into a single matrix Φ and output vector
    # Each reg_row is (ny, n_params); stacking along dimension 0 creates a
    # (nT * N * ny, n_params) matrix.  Similarly y_rows stacks into
    # (nT * N * ny,) vector when flattened.
    Phi = torch.vstack(reg_rows)  # shape (nT*N*ny, n_params)
    Y_vec = torch.cat([y.reshape(-1)
                      for y in y_rows], dim=0)  # shape (nT*N*ny,)

    # Solve the least squares problem using the pseudoinverse
    # θ = (Φ^T Φ)^−1 Φ^T Y   (normal equations)  or directly θ = pinv(Φ) Y
    # We use pinv for numerical stability
    theta = (torch.linalg.pinv(Phi)) @ Y_vec  # shape (n_params,)

    # Extract B
    if nu > 0:
        B_vec = theta[: n_B]
        B = B_vec.view(sys_dim, nu)
    else:
        B = torch.zeros((sys_dim, 0), dtype=dtype, device=device)

    # Extract D
    D_start = n_B
    D_end = D_start + n_D
    if nu > 0:
        D_vec = theta[D_start: D_end]
        D = D_vec.view(ny, nu)
    else:
        D = torch.zeros((ny, 0), dtype=dtype, device=device)

    # Extract z0 for each trajectory
    z0_vec = theta[D_end:]
    # Reshape to (sys_dim, nT)
    z0 = z0_vec.view(nT, sys_dim).T  # shape (sys_dim, nT)

    return A, B, C, D, z0


def get_ssidgpk(SimData: torch.tensor, nTrain: int, nTest: int, lifting_order: int, delay: int):

    # Data Preparation
    n, N = SimData.shape[1], SimData.shape[2]
    ssid_Y = SimData[:nTrain, :, :]
    ssid_U = torch.zeros(
        (nTrain, 1, N), dtype=ssid_Y.dtype, device=ssid_Y.device)
    ICsetTrain = torch.cat([SimData[j, :, 0].view(n, 1)
                           for j in range(nTrain)], dim=1)
    ICsetTest = torch.cat([SimData[j, :, 0].view(n, 1)
                          for j in range(nTrain, nTrain + nTest)], dim=1)

    # Multi-Trajectory Subspace Identification
    A, B, C, D, z0_lift = SSID(
        ssid_U, ssid_Y, delay=delay, sys_dim=lifting_order)

    # Gaussian Process Regression
    ObsManager = GPObservablesManager()
    for i in range(lifting_order):
        ObsManager.add_observable(
            index=i, d=C.shape[0], ns=z0_lift.shape[1], kernel_types=[
                'Gaussian'],
            combination='sum', noise=1e-4, m=500
        )
    ObsManager.set_random_hyperparameters(scale=[1., 1., None])
    for i in range(lifting_order):
        ObsManager.optimize_hyperparameters(opt_sigma=True)

    # Trajectory Simulation and Model Evaluation
    XhatTrain, XcvTrain, TrainNRMSE = sim_and_eval(
        ObsManager, A, C, ICsetTrain, SimData, traj_offset=0)
    XhatTest,  XcvTest,  TestNRMSE = sim_and_eval(
        ObsManager, A, C, ICsetTest,  SimData, traj_offset=nTrain)

    return {
        "ObsManager": ObsManager,
        "A": A, "B": B, "C": C, "D": D,
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
        }
    }
