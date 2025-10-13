import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans

# Plotting Functions


def compare_model_predictions(
    time,
    models,
    SimData,
    idx,
    N,
    system_name,
    title_suffix="",
    *,
    split="train",                    # "train" or "test"
    sim_offset=0,                     # e.g., nTrain for test set
    compare_to="SimData",             # "SimData" or "SimData_clean"
    SimData_clean=None,               # required if compare_to="SimData_clean"
    sigma=1.0,                        # number of std-devs for bands (iGPK)
    colors=None                       # optional color map per model
):
    """
    Compare time-series predictions from multiple models to ground truth, with optional uncertainty bands.

    Parameters
    ----------
    time : 1D tensor/array of length N
    models : list of dicts, one per model. Each dict should contain:
        {
          "name": "iGPK",
          # For the chosen 'split' (train/test), provide tensors shaped as below.
          # Xhat_* : (nTraj, nStates, N)
          # Xcvhat_* : (nTraj, nStates, nStates, N)  (optional; if absent → deterministic line)
          "train": {"Xhat": XhatTrain, "Xcvhat": XcvhatTrain (optional)},
          "test" : {"Xhat": XhatTest,  "Xcvhat": XcvhatTest  (optional)}
        }
    SimData : tensor of shape (numTraj, nStates, N_total)
        Ground truth (noisy or clean, depending on your dataset). For test plots,
        use sim_offset=nTrain to index into test trajectories.
    idx : int
        Which trajectory index to plot (within the chosen split).
    N : int
        Number of time steps to plot.
    system_name : str
    title_suffix : str
    split : {"train","test"}
    sim_offset : int
        Added to SimData's first index to align with chosen split.
    compare_to : {"SimData","SimData_clean"}
        Choose which ground truth to overlay. If "SimData_clean", pass SimData_clean.
    SimData_clean : tensor like SimData
    sigma : float
        Width of the uncertainty band in standard deviations for stochastic models (e.g., iGPK).
    colors : dict or list
        Optional mapping from model name → color string, or a list of colors matching `models` order.

    Notes
    -----
    - Any model entry without 'Xcvhat' is treated as deterministic (no bands).
    - Works for any state dimension; creates one subplot per state.
    - If `nStates == 1`, returns a single-axis figure for convenience.
    """

    assert split in ("train", "test"), "split must be 'train' or 'test'"

    # Pick ground truth
    if compare_to == "SimData":
        GT = SimData
    elif compare_to == "SimData_clean":
        if SimData_clean is None:
            raise ValueError(
                "compare_to='SimData_clean' requires SimData_clean")
        GT = SimData_clean
    else:
        raise ValueError("compare_to must be 'SimData' or 'SimData_clean'")

    # Infer state dimension from the first model
    if len(models) == 0:
        raise ValueError("`models` cannot be empty.")
    Xhat0 = models[0][split]["Xhat"]
    n_states = Xhat0.shape[1]

    # Colors
    default_cycle = plt.rcParams['axes.prop_cycle'].by_key().get(
        'color', ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])
    if isinstance(colors, dict):
        def color_for(name, k): return colors.get(
            name, default_cycle[k % len(default_cycle)])
    elif isinstance(colors, (list, tuple)):
        def color_for(name, k): return colors[k % len(colors)]
    else:
        def color_for(name, k): return default_cycle[k % len(default_cycle)]

    # Setup figure/axes
    fig_height = max(4, 1.8 * n_states)
    fig, axes = plt.subplots(
        n_states, 1, figsize=(7.2, fig_height), sharex=True)
    if n_states == 1:
        axes = [axes]

    # Title
    gt_label = "NL (truth: noisy)" if compare_to == "SimData" else "NL (truth: clean)"
    fig.suptitle(f"{system_name}: {title_suffix} [{split.capitalize()}]")

    # Plot per state
    for s in range(n_states):
        ax = axes[s]

        # Ground truth
        gt = GT[sim_offset + idx, s, :N]
        ax.plot(time, gt.cpu().numpy(), linestyle="--", linewidth=1.3,
                color="black", alpha=0.75, label=gt_label)

        # Overlay all models
        for k, model in enumerate(models):
            name = model.get("name", f"Model {k+1}")
            pack = model[split]
            Xhat = pack["Xhat"][idx, s, :N]

            col = color_for(name, k)
            ax.plot(time, Xhat.cpu().numpy(),
                    label=name, linewidth=1.6, color=col)

            # Optional uncertainty band if available
            Xcvhat = pack.get("Xcvhat", None)
            if Xcvhat is not None:
                # take diag element (state s) variance over time
                var_s = Xcvhat[idx, s, s, :N]
                # clamp small negatives due to numerical issues
                std_s = torch.sqrt(torch.clamp(var_s, min=0.0))
                lower = (Xhat - sigma * std_s).cpu().numpy()
                upper = (Xhat + sigma * std_s).cpu().numpy()
                ax.fill_between(time, lower, upper, alpha=0.16, color=col)

        ax.set_ylabel(f"X{s+1}")
        ax.grid(True, linestyle=":", linewidth=0.7)

    axes[-1].set_xlabel("Time [s]")

    # One shared legend
    # Build a clean legend across axes: collect handles/labels from the last axis
    handles, labels = axes[-1].get_legend_handles_labels()
    # Deduplicate while preserving order
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels)
            if not (l in seen or seen.add(l))]
    if uniq:
        fig.legend(*zip(*uniq), loc="upper right", bbox_to_anchor=(0.98, 0.98))

    fig.tight_layout(rect=[0, 0, 0.98, 0.96])
    # plt.show()
    return fig, axes


def plot_time_series_with_bounds(time, Xhat, Xcvhat, SimData, idx, N, system_name, title_suffix, sim_offset=0):
    """
    Plots time series for each state with uncertainty bounds.

    Parameters:
        time        : 1D array of time points.
        Xhat        : Array of model state estimates.
        Xcvhat      : Array of state covariance estimates.
        SimData     : Array of true system states.
        idx         : Index of the trajectory to plot.
        N           : Number of time steps to plot.
        system_name : Name of the system (for labeling).
        title_suffix: Suffix for the plot title.
        sim_offset  : Offset for SimData indexing (e.g., nTrain for test trajectories).
    """
    n_states = Xhat.shape[1]
    fig, axes = plt.subplots(n_states, 1, figsize=(6, 5))

    # If there is only one state, wrap the Axes object in a list for uniform handling.
    if n_states == 1:
        axes = [axes]

    fig.suptitle(f'{system_name}: {title_suffix}')

    for i in range(n_states):
        lower_bound = Xhat[idx, i, :] - (Xcvhat[idx, i, i, :] ** 0.5)
        upper_bound = Xhat[idx, i, :] + (Xcvhat[idx, i, i, :] ** 0.5)
        axes[i].fill_between(time, lower_bound, upper_bound,
                             alpha=0.16, color='blue')
        axes[i].plot(time, Xhat[idx, i, :], label='iGPK', color='blue')
        axes[i].plot(time, SimData[sim_offset + idx, i, :N],
                     label='NL', linestyle='--', color='red')
        axes[i].set_ylabel(f'State X{i+1}')
        axes[i].legend()
        axes[i].grid()
        if i < n_states - 1:
            # Hide x-axis labels for intermediate subplots
            axes[i].set_xticklabels([])

    axes[-1].set_xlabel('Time [s]')
    plt.tight_layout()
    # plt.show()


def plot_phase(Xhat, SimData, ICset, idx, N, system_name, title_suffix, sim_offset=0):
    """
    Plots the phase trajectory comparing the model prediction to the true (nonlinear) system.
    Automatically switches to a 3D plot if the system has 3 state dimensions.

    Parameters:
        Xhat       : Array of model state estimates of shape (trajectories, states, time_steps).
        SimData    : Array of true system states.
        ICset      : Array of initial conditions.
        idx        : Index of the trajectory to plot.
        N          : Number of time steps to plot.
        system_name: Name of the system (for labeling).
        title      : Title for the plot.
        sim_offset : Offset for SimData indexing (e.g., nTrain for test trajectories).
    """
    n_states = Xhat.shape[1]

    if n_states == 3:
        # Create a 3D plot for 3-state systems
        fig = plt.figure(figsize=(6, 4.5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(Xhat[idx, 0, :], Xhat[idx, 1, :],
                Xhat[idx, 2, :], label='iGPK')
        ax.plot(SimData[sim_offset + idx, 0, :],
                SimData[sim_offset + idx, 1, :],
                SimData[sim_offset + idx, 2, :], label='Nonlinear', linestyle='--')
        ax.scatter(ICset[0, idx], ICset[1, idx],
                   ICset[2, idx], label='IC', marker='o')
        ax.set_title(f'{system_name}: {title_suffix}')
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("X3")
        ax.legend()
        # plt.show()
    elif n_states == 2:
        # Default to a 2D phase plot (using the first two state dimensions)
        plt.figure(figsize=(6, 4.5))
        plt.plot(Xhat[idx, 0, :], Xhat[idx, 1, :], label='iGPK')
        plt.plot(SimData[sim_offset + idx, 0, :],
                 SimData[sim_offset + idx, 1, :],
                 label='Nonlinear', linestyle='--')
        plt.plot(ICset[0, idx], ICset[1, idx], label='IC', marker='o')
        plt.title(f'{system_name}: {title_suffix}')
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend()
        plt.grid()
        # plt.show()

    else:
        print('Trajecotry plots are available only for 2 or 3 state systems')


def plot_phase_w_bounds(Xhat, SimData, ICset, idx, N, system_name, title_suffix, sim_offset=0, Xcvhat=None, ellipse_interval=10):
    """
    Plots the phase trajectory comparing the model prediction to the true system.
    Automatically switches to a 3D plot if the system has 3 state dimensions.
    For 2D systems, if Xcvhat is provided, error ellipses (shaded uncertainty regions)
    are added along the trajectory.

    Parameters:
        Xhat          : Array of model state estimates, shape (trajectories, states, time_steps).
        SimData       : Array of true system states.
        ICset         : Array of initial conditions.
        idx           : Index of the trajectory to plot.
        N             : Number of time steps to plot.
        system_name   : Name of the system (for labeling).
        title         : Title for the plot.
        sim_offset    : Offset for SimData indexing (e.g., nTrain for test trajectories).
        Xcvhat        : (Optional) Array of state covariance estimates with shape 
                        (trajectories, states, states, time_steps).
        ellipse_interval: Interval (in time steps) at which to plot error ellipses.
    """
    n_states = Xhat.shape[1]

    if n_states == 3:   # Create a 3D phase plot
        fig = plt.figure(figsize=(6, 4.5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(Xhat[idx, 0, :], Xhat[idx, 1, :],
                Xhat[idx, 2, :], label='iGPK')
        ax.plot(SimData[sim_offset + idx, 0, :N],
                SimData[sim_offset + idx, 1, :N],
                SimData[sim_offset + idx, 2, :N],
                label='Nonlinear', linestyle='--')
        ax.scatter(ICset[0, idx], ICset[1, idx],
                   ICset[2, idx], label='IC', marker='o')
        ax.set_title(f'{system_name}: {title_suffix}')
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("X3")
        ax.legend()
        # plt.show()

    elif n_states == 2:  # 2-D phase plot
        # Create a 2D phase plot
        plt.figure(figsize=(6, 4.5))
        plt.plot(Xhat[idx, 0, :], Xhat[idx, 1, :], label='iGPK')
        plt.plot(SimData[sim_offset + idx, 0, :N],
                 SimData[sim_offset + idx, 1, :N],
                 label='Nonlinear', linestyle='--')
        plt.plot(ICset[0, idx], ICset[1, idx], label='IC', marker='o')

        # If covariance data is provided, overlay error ellipses as uncertainty regions.
        if Xcvhat is not None:  # shaded elipses for 1-sigma bound
            for t in range(0, N, ellipse_interval):
                center = (Xhat[idx, 0, t], Xhat[idx, 1, t])
                # Extract the 2x2 covariance matrix for the first two states
                cov = Xcvhat[idx, 0:2, 0:2, t]
                # Convert to NumPy array if it's a torch tensor
                if hasattr(cov, "cpu"):
                    cov = cov.cpu().numpy()
                # Compute eigenvalues and eigenvectors of the covariance matrix
                vals, vecs = np.linalg.eig(cov)
                order = vals.argsort()[::-1]
                vals = vals[order]
                vecs = vecs[:, order]
                # Compute the angle of the ellipse (in degrees)
                angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
                # For a 1-sigma ellipse, the semi-axis lengths are the square roots of the eigenvalues.
                width, height = 2 * np.sqrt(vals)
                ellipse = Ellipse(xy=center, width=width, height=height, angle=angle,
                                  edgecolor='blue', facecolor='blue', alpha=0.1)
                plt.gca().add_patch(ellipse)

        if Xcvhat is None:  # Plot Title
            plt.title(f'{system_name}: {title_suffix}')
        else:   # Plot Title for 1 sigma bounds
            plt.title(f'{system_name}: {title_suffix} with 1 $\\sigma$ bound')

        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend()
        plt.grid()
        # plt.show()

    else:   # Dimension mismatch
        raise ValueError('Size of Xhat in dimension 1 has to be 2 or 3')


def plot_predicted_sd_error(XcvhatTest, SimData, XhatTest, idx, N, nTrain, trajectory_label):
    """
    Plots the predicted standard deviation (SD) and absolute error for a given test trajectory.

    Parameters:
        XcvhatTest      : Tensor of covariance estimates, shape (trajectories, states, states, time_steps).
        SimData         : Tensor of true system states, shape (num_trajectories, states, time_steps).
        XhatTest        : Tensor of predicted state estimates, shape (trajectories, states, time_steps).
        idx             : Index of the trajectory to plot.
        N               : Number of time steps to plot.
        nTrain          : Offset for test trajectories in SimData.
        trajectory_label: String label for the trajectory (e.g., "Worst Test", "Best Test").
    """
    n = XhatTest.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(6, 5))

    # Protection for 1D systems: ensure axes is always iterable.
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f'Predicted SD & |Error| for {trajectory_label} Trajectory ({idx})')

    for i in range(n):
        # Compute standard deviation from covariance (for a 1-sigma bound)
        sigma = (torch.abs(XcvhatTest[idx, i, i, :N-1]) ** 0.5)
        # Compute absolute error between true and predicted values
        error = torch.abs(
            SimData[nTrain+idx, i, :N-1] - XhatTest[idx, i, :N-1])

        # Optionally, convert tensors to NumPy for plotting (if needed)
        sigma = sigma.detach().cpu().numpy()
        error = error.detach().cpu().numpy()

        axes[i].plot(sigma, label='$\\sigma_X$', color='blue')
        axes[i].plot(error, label='$\\epsilon_X$', color='red', linestyle='--')
        axes[i].set_ylabel(f'X{i+1}')
        axes[i].grid()
        axes[i].legend()
        if i < n - 1:
            # Hide x-axis labels for all but the last subplot
            axes[i].set_xticklabels([])

    axes[-1].set_xlabel('Time Step')
    plt.tight_layout()
    # plt.show()


# Matrix Functions

def check_pd(matrix: torch.Tensor):
    """
    Checks if a given 2D PyTorch tensor is positive definite (PD) or 
    positive semi-definite (PSD) or neither.

    Args:
        matrix (torch.Tensor): A 2D square PyTorch tensor.

    Returns:
        str: "Positive Definite", "Positive Semi-Definite", or "Neither"
    """
    if not (matrix.dim() == 2 and matrix.shape[0] == matrix.shape[1]):
        raise ValueError("Input must be a square matrix")

    try:
        # Attempt Cholesky decomposition for positive definiteness
        torch.linalg.cholesky(matrix)
        return "Positive Definite"
    except RuntimeError:
        # Compute eigenvalues for semi-definiteness check
        eigenvalues = torch.linalg.eigvals(matrix)
        if torch.all(eigenvalues >= 0):
            return "Positive Semi-Definite"
        else:
            return "Neither"


def MatViz3d(matrix: torch.Tensor):
    """
    Plots a 3D surface representation of a 2D PyTorch tensor.

    Args:
        matrix (torch.Tensor): A 2D tensor representing the surface.

    Raises:
        ValueError: If the input is not a 2D tensor.
    """
    if not (matrix.dim() == 2):
        raise ValueError("Input must be a 2D tensor")

    # Convert to NumPy
    matrix_np = matrix.cpu().detach().numpy()

    # Create meshgrid for X, Y indices
    x = np.arange(matrix_np.shape[1])
    y = np.arange(matrix_np.shape[0])
    X, Y = np.meshgrid(x, y)

    # Create figure and 3D axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(X, Y, matrix_np, cmap='viridis', edgecolor='k')

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6)
    cbar.set_label('Element Value')

    # Labels
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    ax.set_zlabel('Value')
    ax.set_title(
        f'3D Surface Plot | Max={np.max(matrix_np)}, Min={np.min(matrix_np)}')

    # plt.show()


def MatViz(matrix: torch.Tensor, plot_type: str = 'surf'):
    """
    Visualize a 2D PyTorch tensor as either a 3D surface or a 2D heatmap.

    Args:
        matrix (torch.Tensor): A 2D tensor of shape (rows, cols).
        plot_type (str): 'surf' for a 3D surface plot, 'heat' for a 2D heatmap.

    Raises:
        ValueError: If the input is not 2D or plot_type is invalid.
    """
    # Validate inputs
    if matrix.dim() != 2:
        raise ValueError("Input must be a 2D tensor")
    if plot_type not in ('surf', 'heat'):
        raise ValueError("plot_type must be either 'surf' or 'heat'")

    # Convert to NumPy
    matrix_np = matrix.cpu().detach().numpy()
    vmin, vmax = matrix_np.min(), matrix_np.max()

    if plot_type == 'heat':
        # 2D heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix_np, cmap='viridis',
                       aspect='auto', vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Element Value')
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Row Index')
        ax.set_title(f'Heatmap | Max={vmax:.3g}, Min={vmin:.3g}')
        # plt.show()

    else:
        # 3D surface plot
        x = np.arange(matrix_np.shape[1])
        y = np.arange(matrix_np.shape[0])
        X, Y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(
            X, Y, matrix_np, cmap='viridis', edgecolor='k', vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6)
        cbar.set_label('Element Value')
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Row Index')
        ax.set_zlabel('Value')
        ax.set_title(f'3D Surface Plot | Max={vmax:.3g}, Min={vmin:.3g}')
        # plt.show()


def plot_eigen(A):  # Eigen value plot of Koopman Matrices
    A = A.detach().cpu()
    eigval = torch.linalg.eigvals(A)

    eigreal, eigimag = eigval.real, eigval.imag
    eigreal, eigimag = eigreal.detach().numpy(), eigimag.detach().numpy()
    eig_mag = np.sqrt(eigreal**2 + eigimag**2)

    theta = np.linspace(0, 2*np.pi, 500)
    unitCirclex, unitCircley = np.cos(theta), np.sin(theta)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    # First subplot: Eigenvalues plot
    axes[0].plot(unitCirclex, unitCircley, color='orange', label='Unit Circle')
    for i in range(np.size(eig_mag)):
        if eig_mag[i] <= 1:
            axes[0].scatter(eigreal, eigimag, color='green',
                            label='Eigenvalues')
        else:
            axes[0].scatter(eigreal, eigimag, color='red', label='Eigenvalues')

    axes[0].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes[0].axvline(0, color='black', linewidth=0.5, linestyle='--')
    axes[0].set_title(f"Eigenvalues of A Matrix with {A.shape[0]} Observables")
    axes[0].set_xlabel("Real Part")
    axes[0].set_ylabel("Imaginary Part")
    axes[0].grid(True)
    axes[0].legend(labels=['Unit Circle', 'Eignevalues'], loc='upper right')

    # Second subplot: Heatmap of matrix A
    im = axes[1].imshow(A.detach().numpy(), cmap='viridis', aspect='auto')
    fig.colorbar(im, ax=axes[1], label="Value")
    axes[1].set_title(f'{A.shape[0]}-D Koopman Matrix')
    axes[1].set_xlabel("Columns")
    axes[1].set_ylabel("Rows")
    plt.tight_layout()
    return fig


# K-Means Clusting helper function

def get_kmeans(data, num_centers=1):
    """
    Compute K-Means clustering for the given data and return cluster centroids.

    Args:
        data (torch.Tensor or np.ndarray): Data of shape (dimension, samples).
        num_centers (int): Number of cluster centers (clusters) to compute.

    Returns:
        torch.Tensor: Cluster centroids of shape (dimension, num_centers).
    """
    # Convert data to NumPy array if necessary
    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = np.array(data)

    # Transpose data to shape (samples, dimension) for scikit-learn
    data_np = data_np.T

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_centers, random_state=0).fit(data_np)

    # Get centroids; shape will be (num_centers, dimension)
    centroids_np = kmeans.cluster_centers_

    # Convert centroids to a torch tensor and transpose to shape (dimension, num_centers)
    centroids = torch.from_numpy(centroids_np.T).float()

    return centroids


# Simulation Tools

def sim_and_eval(ObsManager, A, C, ICset, SimData_ref, traj_offset: int = 0):
    """
    Simulate from Koopman (A, C) for each IC in ICset and compute per-trajectory NRMSE.
    Returns (Xhat, Xcvhat, NRMSE).

    Args:
        ObsManager: trained GPObservablesManager
        A, C: Koopman matrices (CPU or GPU OK; function will move to CPU to simulate)
        ICset: (n, nTraj) tensor of initial conditions for this split (train or test)
        SimData_ref: normalized (or chosen reference) data, shape (num_traj, n, N+1)
        traj_offset: index offset into SimData_ref for this split
    """
    A, C = A.to(device='cpu'), C.to(device='cpu')
    ICset = ICset.to(device='cpu')
    SimData_ref = SimData_ref.to(device='cpu')

    nTraj = ICset.shape[1]
    n = C.shape[0]
    N = SimData_ref.shape[2] - 1
    p = A.shape[0]

    Zmean = torch.empty((nTraj, p, N), dtype=ICset.dtype, device=ICset.device)
    Zcv = torch.empty((nTraj, p, p, N), dtype=ICset.dtype, device=ICset.device)
    Xhat = torch.empty((nTraj, n, N), dtype=ICset.dtype, device=ICset.device)
    Xcv = torch.empty((nTraj, n, n, N), dtype=ICset.dtype, device=ICset.device)
    NRMSE = torch.empty((nTraj, n), dtype=ICset.dtype, device=ICset.device)

    for j in range(nTraj):
        # 1) Predict initial lifted state distribution from IC
        for i in range(p):
            Zmean[j, i, 0] = ObsManager.predict_mean(i, ICset[:, j].view(n, 1))
            Zcv[j, i, i, 0] = ObsManager.predict_covariance(
                i, ICset[:, j].view(n, 1))

        # 2) Propagate with linear model
        Zmean[j], Zcv[j], Xhat[j], Xcv[j] = gpk.sim_LTI(
            Zmean[j, :, 0].view(p, 1), A, C, num_steps=N, ts=None, x0cv=Zcv[j, :, :, 0]
        )

        # 3) NRMSE against reference (per-trajectory range)
        y_true = SimData_ref[traj_offset + j, :, :N]  # (n, N)
        errors = Xhat[j] - y_true
        rmse = torch.sqrt(torch.mean(errors**2, dim=1))                 # (n,)
        y_max = y_true.max(dim=1).values
        y_min = y_true.min(dim=1).values
        y_range = torch.where((y_max - y_min) == 0,
                              torch.ones_like(y_max),
                              (y_max - y_min))
        NRMSE[j] = rmse / y_range

    return Xhat.detach(), Xcv.detach(), NRMSE.detach()
