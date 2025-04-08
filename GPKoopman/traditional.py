import itertools
import torch
from .autonomous import sim_LTI

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

    # Evaluation on training set
    ZedTrain = torch.empty((nTrain, p, N))    # n+p for state-augmentation
    # ZmeanTrain[:, :n, 0] = ICsetTrain.T    # only for state-augmentation

    XedTrain = torch.empty((nTrain, n, N))
    TrainRMSE_eDMD = torch.empty((nTrain, n))

    for j in range(nTrain):  # Prediction for all training trajectories
        ZedTrain[j, :, 0] = phi_batch(ICsetTrain[:, j].view(n, 1)).view(
            p,)        # n+i for state-augmentation

        ZedTrain[j, :, :], XedTrain[j, :, :] = sim_LTI(
            ZedTrain[j, :, 0], A_edmd, C_edmd, num_steps=N, ts=None, x0cv=None)
        TrainRMSE_eDMD[j, :] = torch.sqrt(torch.mean(
            (XedTrain[j, :, :] - SimData[j, :, :N])**2, 1))

    # Evaluation on test set
    ZedTest = torch.empty((nTest, p, N))
    # ZmeanTest[:, :n, 0] = ICsetTest.T  # only for state-augmentation

    XedTest = torch.empty((nTest, n, N))
    TestRMSE_eDMD = torch.empty((nTest, n))

    for j in range(nTest):  # Prediction for all testing trajectories
        ZedTest[j, :, 0] = phi_batch(ICsetTest[:, j].view(n, 1)).view(
            p,)          # n+i for state-augmentation

        ZedTest[j, :, :], XedTest[j, :, :] = sim_LTI(
            ZedTest[j, :, 0], A_edmd, C_edmd, num_steps=N, ts=None, x0cv=None)
        TestRMSE_eDMD[j, :] = torch.sqrt(torch.mean(
            (XedTest[j, :, :] - SimData[nTest+j, :, :N])**2, 1))

    XedTrain, XedTest = XedTrain.detach(), XedTest.detach()
    TestRMSE_eDMD, TrainRMSE_eDMD = TestRMSE_eDMD.detach(), TrainRMSE_eDMD.detach()

    return A_edmd, C_edmd, XedTrain, XedTest, TrainRMSE_eDMD, TestRMSE_eDMD


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


def eDMD_RBF(SimData, nTrain, nTest, centers, width=None, rbf_type='gaussian', state_aug=False):
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

    # Define the observable lifting function using rbf_observable.
    # It accepts inputs of shape (state_dim, num_points) and returns (m, num_points)
    # phi_batch = lambda X: rbf_observable(X, centers, width, rbf_type, state_aug)

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
