import os

# Optional: set these before importing numpy / sklearn for stricter reproducibility
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from .autonomous import sim_LTI

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
    sim_offset=0,                     # e.g., nTest for train set
    compare_to="SimData",             # "SimData" or "SimData_clean"
    SimData_clean=None,               # required if compare_to="SimData_clean"
    sigma=1.0,                        # number of std-devs for bands (iGPK)
    colors=None,                      # optional color map per model
    skip_title=False,
    y_labels=None
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
    fig_height = max(5, 1.8 * n_states)
    fig, axes = plt.subplots(
        n_states, 1, figsize=(5, fig_height), sharex=True)
    if n_states == 1:
        axes = [axes]

    # Title
    gt_label = "NL (truth: noisy)" if compare_to == "SimData" else "NL (truth: clean)"
    if skip_title is False:
        fig.suptitle(f"{system_name}: {title_suffix} [{split.capitalize()}]")

    # Plot per state
    for s in range(n_states):
        ax = axes[s]

        # Ground truth
        gt = GT[sim_offset + idx, s, :N].cpu().numpy()
        # choose about 20 evenly spaced marker points
        # n_markers = 20
        # marker_idx = np.linspace(0, N - 1, n_markers, dtype=int)
        # overlay sparse markers for clarity
        # ax.plot(time[marker_idx], gt[marker_idx], marker='o', linestyle='--',
        #         linewidth=1.3, color="black", markersize=4, alpha=0.8, label='Truth')

        # Overlay all models predictions
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
                std_s = torch.sqrt(torch.abs(var_s))
                lower = (Xhat - sigma * std_s).cpu().numpy()
                upper = (Xhat + sigma * std_s).cpu().numpy()
                ax.fill_between(time, lower, upper, alpha=0.16, color=col)
        
        # Ground truth (plotted last for visibility)
        ax.plot(time, gt, linestyle="--", linewidth=1.3, color="black",
                alpha=0.75, label=gt_label)

        if y_labels is None:
            ax.set_ylabel(f"X{s+1}")
        else:
            ax.set_ylabel(f'${y_labels[s]}$')
        ax.grid(True, linestyle=":", linewidth=0.7)

    axes[-1].set_xlabel("Time [s]")

    # One shared legend
    # Build a clean legend across axes: collect handles/labels from the last axis
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),   # just above the plots
        ncol=len(labels),             # all entries in one horizontal row
        frameon=False
    )
    # Deduplicate while preserving order
    # seen = set()
    # uniq = [(h, l) for h, l in zip(handles, labels)
    #         if not (l in seen or seen.add(l))]
    # if uniq:
    #     fig.legend(*zip(*uniq), loc="upper right", bbox_to_anchor=(0.98, 0.98))

    fig.tight_layout(rect=[0, 0, 1., 0.95])
    # plt.show()
    return fig, axes


def plot_NRMSE_metrics(TrainNRMSE_list, TestNRMSE_list, model_names):
    """
    Compare Train/Test NRMSE across multiple models using boxplots (no per-trajectory curves).
    Adds a circular marker to indicate the mean on each box.

    Args:
        TrainNRMSE_list (list of torch.Tensor): Each tensor has shape (nTraj, nStates).
        TestNRMSE_list  (list of torch.Tensor): Each tensor has shape (nTraj, nStates).
        model_names     (list of str): Names of the models, used as plot labels.

    Returns:
        fig (matplotlib.figure.Figure)
    """
    assert len(TrainNRMSE_list) == len(TestNRMSE_list) == len(model_names), \
        "Train/Test lists and model_names must have the same length."

    def _to_traj_means(x: torch.Tensor) -> np.ndarray:
        # x: (nTraj, nStates) -> per-trajectory mean across states -> (nTraj,)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.detach().cpu()
        if x.ndim == 1:
            # already (nTraj,)
            return x.numpy()
        if x.ndim != 2:
            raise ValueError(
                f"Expected tensor of shape (nTraj, nStates) or (nTraj,), got {tuple(x.shape)}")
        return x.mean(dim=1).numpy()

    # Prepare data: list of arrays, each array is (nTraj,)
    train_data = [_to_traj_means(rmse) for rmse in TrainNRMSE_list]
    test_data = [_to_traj_means(rmse) for rmse in TestNRMSE_list]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    def _boxplot_with_mean(ax, data, title):
        bp = ax.boxplot(
            data,
            labels=model_names,
            showfliers=False,
            patch_artist=False,
            widths=0.6,
            showmeans=True,
            meanline=True
        )
        ax.set_title(title)
        ax.set_ylabel("Mean NRMSE (averaged across states)")
        ax.grid(axis="y")
        ax.tick_params(axis="x", labelrotation=20)
        # ax.set_ylim(0, min(2, ax.get_ylim()[1]))
        return bp

    _boxplot_with_mean(axes[0], train_data,
                       "Training NRMSE (Boxplot Across Trajectories)")
    _boxplot_with_mean(axes[1], test_data,
                       "Test NRMSE (Boxplot Across Trajectories)")

    plt.tight_layout()
    return fig


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
        sim_offset  : Offset for SimData indexing (e.g., nTest for train trajectories).
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
        sim_offset : Offset for SimData indexing (e.g., nTest for train trajectories).
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
        sim_offset    : Offset for SimData indexing (e.g., nTest for train trajectories).
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


def plot_predicted_sd_error(XcvhatTest, SimData, XhatTest, idx, N, offset, trajectory_label):
    """
    Plots the predicted standard deviation (SD) and absolute error for a given test trajectory.

    Parameters:
        XcvhatTest      : Tensor of covariance estimates, shape (trajectories, states, states, time_steps).
        SimData         : Tensor of true system states, shape (num_trajectories, states, time_steps).
        XhatTest        : Tensor of predicted state estimates, shape (trajectories, states, time_steps).
        idx             : Index of the trajectory to plot.
        N               : Number of time steps to plot.
        offset          : Offset for trajectories in SimData (eg. nTest for train trajectories).
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
            SimData[offset+idx, i, :N-1] - XhatTest[idx, i, :N-1])

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
        if torch.all(eigenvalues.real >= 0):
            return "Positive Semi-Definite"
        else:
            return "Neither"


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


def plot_eigen(A:torch.Tensor, tol:float=1e-9, legend:bool=False):  # Eigen value plot of Koopman Matrices
    A = A.detach().cpu()
    eigval = torch.linalg.eigvals(A)

    eigreal, eigimag = eigval.real, eigval.imag
    eigreal, eigimag = eigreal.detach().numpy(), eigimag.detach().numpy()
    eig_mag = np.sqrt(eigreal**2 + eigimag**2)

    theta = np.linspace(0, 2*np.pi, 500)
    unitCirclex, unitCircley = np.cos(theta), np.sin(theta)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    # First subplot: Eigenvalues plot
    axes[0].plot(unitCirclex, unitCircley, color='blue', label='Unit Circle',
                 linestyle='-', linewidth=1.0, alpha=0.7, zorder=1)

    # Classify points by magnitude
    is_zero = np.isclose(eig_mag, 0.0, atol=tol)
    is_one = np.isclose(eig_mag, 1.0, atol=tol) & ~is_zero
    less_than_one = (eig_mag < 1.0 - tol) & ~is_zero
    greater_than_one = eig_mag > 1.0 + tol

    if np.any(is_zero):
        axes[0].scatter(eigreal[is_zero], eigimag[is_zero], color='black', label='|eig| = 0', zorder=3)
    if np.any(is_one):
        axes[0].scatter(eigreal[is_one], eigimag[is_one], color='orange', label='|eig| = 1', zorder=3)
    if np.any(less_than_one):
        axes[0].scatter(eigreal[less_than_one], eigimag[less_than_one], color='green', label='|eig| < 1', zorder=3)
    if np.any(greater_than_one):
        axes[0].scatter(eigreal[greater_than_one], eigimag[greater_than_one], color='red', label='|eig| > 1', zorder=3)

    axes[0].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes[0].axvline(0, color='black', linewidth=0.5, linestyle='--')
    axes[0].set_title(f"Eigenvalues of A Matrix with {A.shape[0]} Observables")
    axes[0].set_xlabel("Real Part")
    axes[0].set_ylabel("Imaginary Part")
    axes[0].grid(True)
    if legend:
        axes[0].legend(loc='upper right', fontsize='small')

    # Second subplot: Heatmap of matrix A
    im = axes[1].imshow(A.detach().numpy(), cmap='viridis', aspect='auto')
    fig.colorbar(im, ax=axes[1], label="Value")
    axes[1].set_title(f'{A.shape[0]}-D Koopman Matrix')
    axes[1].set_xlabel("Columns")
    axes[1].set_ylabel("Rows")
    plt.tight_layout()
    return fig


def get_kmeans(data, num_centers=1, seed=0, dtype=torch.float32):
    """
    Deterministic K-Means centroids.

    Args:
        data:
            Tensor/array of shape (dimension, samples).
        num_centers:
            Number of K-Means centers.
        seed:
            Random seed for K-Means initialization.
        dtype:
            Torch dtype of returned centroids.

    Returns:
        centroids:
            Tensor of shape (dimension, num_centers).
    """

    if isinstance(data, torch.Tensor):
        device = data.device
        data_np = data.detach().cpu().numpy()
    else:
        device = torch.device("cpu")
        data_np = np.asarray(data)

    # Shape: (samples, dimension)
    data_np = np.ascontiguousarray(data_np.T, dtype=np.float64)

    kmeans = KMeans(
        n_clusters=num_centers,
        init="k-means++",
        n_init=8,          # do not leave this as sklearn's version-dependent default
        random_state=seed,
        algorithm="lloyd",  # deterministic batch Lloyd iterations
        max_iter=300,
        tol=1e-4,
        copy_x=True,
    )

    kmeans.fit(data_np)

    # Shape: (num_centers, dimension)
    centroids_np = kmeans.cluster_centers_

    # Canonical ordering of centers.
    # K-Means cluster labels are not intrinsically ordered.
    # This makes the returned column order deterministic.
    order = np.lexsort(centroids_np.T[::-1])
    centroids_np = centroids_np[order]

    # Return shape: (dimension, num_centers)
    centroids = torch.as_tensor(centroids_np.T, dtype=dtype, device=device)

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

    Zmean = torch.zeros((nTraj, p, N), dtype=ICset.dtype, device=ICset.device)
    Zcv = torch.zeros((nTraj, p, p, N), dtype=ICset.dtype, device=ICset.device)
    Xhat = torch.zeros((nTraj, n, N), dtype=ICset.dtype, device=ICset.device)
    Xcv = torch.zeros((nTraj, n, n, N), dtype=ICset.dtype, device=ICset.device)
    NRMSE = torch.zeros((nTraj, n), dtype=ICset.dtype, device=ICset.device)

    for j in range(nTraj):
        # 1) Predict initial lifted state distribution from IC
        for i in range(p):
            Zmean[j, i, 0] = ObsManager.predict_mean(i, ICset[:, j].view(n, 1))
            Zcv[j, i, i, 0] = ObsManager.predict_covariance(
                i, ICset[:, j].view(n, 1)).clamp(min=1e-8, max=1e8)
            if torch.isnan(Zcv[j, i, i, 0]).any():
                warnings.warn(f"GPO-{i} produced NaN lifted cov for {j}-th trajectory", RuntimeWarning)
            if Zcv[j, i, i, 0] > 1e20:
                warnings.warn(f"GPO-{i} produced lifted cov > 1e20 for {j}-th trajectory")

        # 2) Propagate with linear model
        Zmean[j], Zcv[j], Xhat[j], Xcv[j] = sim_LTI(
            Zmean[j, :, 0].view(p, 1), A, C, num_steps=N, ts=None, x0cv=Zcv[j, :, :, 0]
        )
        if torch.isnan(Zcv).any():
            warnings.warn(f"sim_LTI produced NaN lifted covariance for {j}-trajectory", UserWarning)
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


# Evaluation Tools

def _nlpd_one(y, mu, S, jitter=1e-8):
    """
    NLPD for a single multivariate Gaussian y~N(mu,S).
    y, mu: (n,)
    S: (n,n) covariance
    Returns scalar (float)
    """
    n = y.numel()
    S = 0.5 * (S + S.T)  # symmetrize
    S = S + jitter * torch.eye(n, dtype=S.dtype)
    try:
        L = torch.linalg.cholesky(S)
        logdet = 2.0 * torch.log(torch.diag(L)).sum()
        diff = (y - mu).view(n, 1)
        sol = torch.cholesky_solve(diff, L)
        quad = float((diff.T @ sol).item())
        return 0.5 * (n * math.log(2.0 * math.pi) + float(logdet) + quad)
    except Exception:
        # Diagonal fallback
        diag = torch.clamp(torch.diagonal(S), min=jitter)
        logdet = torch.log(diag).sum()
        quad = ((y - mu) ** 2 / diag).sum().item()
        return 0.5 * (n * math.log(2.0 * math.pi) + float(logdet) + quad)

def nlpd_per_traj(Xhat:torch.Tensor, Xcv:torch.Tensor, GT:torch.Tensor):
    """
    Time-averaged NLPD per trajectory
    returns (nTraj,) tensor
    """
    nTraj, n, N = Xhat.shape
    traj_vals = torch.zeros(nTraj, dtype=Xhat.dtype)
    for j in range(nTraj):
        acc = 0.0
        for k in range(N):
            acc += _nlpd_one(GT[j, :, k], Xhat[j, :, k],
                            torch.clamp(torch.abs(Xcv[j, :, :, k]), min=1e-6))
        traj_vals[j] = acc / N
    return traj_vals


# Data Pre-Processing Tools

def load_SimData(system_name:str,
                 trainFrac:float, testFrac:float,
                 clip:int | None = None, normalize:bool=False):
    """
    Loads and optionally clips and normalizes trajectory data.
    Number of outputs vary if normalize = True.
    """
    data = torch.load(f"Data/DataAuto_{system_name}.pt", weights_only=True)
    SimData = data["trajectories"]  # Shape: (num_trajectories, state_dim, num_steps)
    ts = data["sample_time"]
    num_trajectories, N = data["num_trajectories"], data["num_steps"]

    if (trainFrac + testFrac) <= 1.0:
        nTest = math.floor(num_trajectories * testFrac)
        nTrain = math.floor(num_trajectories * trainFrac)
    else:
        raise ValueError('Sum of trainFrac and testFrac should be leq 1.')

    if clip is not None:
        SimData = SimData[:, :, :clip+1]
        N = clip
    
    if normalize is True:
        SimData, mu_vec, std_vec = normalize_data(SimData_raw=SimData,
                                    nTest=nTest, nTrain=nTrain, N=N)
        
        return SimData, mu_vec, std_vec, ts, num_trajectories, N, nTrain, nTest
    else:
        return SimData, ts, num_trajectories, N, nTrain, nTest


def normalize_data(SimData_raw:torch.Tensor,
                   nTest:int, nTrain:int, N:int):
    """
    Normalizes entire dataset using training set statistics
    """
    # SimData shape: (num_traj, state_dim, num_steps)
    mu_vec = SimData_raw[nTest:(nTest+nTrain), :, :N].mean(
        dim=(0, 2))                                # (n,)
    std_vec = SimData_raw[nTest:(nTest+nTrain), :, :N].std(
        dim=(0, 2), unbiased=False).clamp_min(1e-8)  # (n,)

    # Apply normalization to ALL trajectories (train+test)
    SimData = (SimData_raw - mu_vec.view(1, -1, 1)) / std_vec.view(1, -1, 1)
    return SimData, mu_vec, std_vec


def add_noise(SimData_norm, noise_type="gaussian", intensity=0.05, seed=1111):
    """
    Add noise to normalized simulation data.

    Args:
        SimData_norm (torch.Tensor): Normalized trajectories,
            shape (num_traj, state_dim, num_steps).
        noise_type (str): 'gaussian' or 'uniform'.
        intensity (float): Noise strength (default 0.05 = 5% of 1 std).
                          Since data is normalized, this is relative to std=1.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        torch.Tensor: Noisy version of SimData_norm.
    """
    if seed is not None:
        torch.manual_seed(seed)

    if noise_type == None:
        noise = 0
    elif noise_type == "gaussian":
        noise = torch.randn_like(SimData_norm) * intensity
    elif noise_type == "uniform":
        # Uniform noise in [-intensity, intensity]
        noise = (torch.rand_like(SimData_norm) * 2 - 1) * intensity
    elif noise_type == "linear_gaussian":
        # Gaussian noise with intensity varying linearly with state value
        var_intensity = intensity * SimData_norm
        noise = torch.randn_like(SimData_norm) * var_intensity
    elif noise_type == "quadratic_gaussian":
        # Gaussian noise with intensity varying as square of state value
        var_intensity = intensity * (SimData_norm ** 2)
        noise = torch.randn_like(SimData_norm) * var_intensity
    elif noise_type == "linear_uniform":
        # Uniform noise with linearly varying intensity
        var_intensity = intensity * SimData_norm
        noise = (torch.rand_like(SimData_norm) * 2 - 1) * var_intensity
    else:
        raise ValueError(
            f"Unsupported noise_type {noise_type}. Choose 'gaussian', 'uniform', 'linear_gaussian', 'quadratic_gaussian', or 'linear_uniform'.")

    return SimData_norm + noise


def find_hp_init(dataset:torch.Tensor, max_pairs:float=5_000_000)->float:
    """
    Finds the median pair-wise distance in the dataset.
    - dataset must be torch.Tensor of shape (trajectories, states, steps).
    """
    X = dataset.permute(0, 2, 1).reshape(-1, dataset.shape[1])
    X = X.detach().cpu().numpy()

    npts = X.shape[0]
    npairs = npts * (npts - 1) // 2

    if npairs <= max_pairs:
        d = []
        for i in range(npts - 1):
            d.append(np.linalg.norm(X[i + 1:] - X[i], axis=1))
        return float(np.median(np.concatenate(d)))

    rng = np.random.default_rng(0)
    i = rng.integers(0, npts, size=max_pairs)
    j = rng.integers(0, npts, size=max_pairs)
    mask = i != j

    return float(np.median(np.linalg.norm(X[i[mask]] - X[j[mask]], axis=1)))
