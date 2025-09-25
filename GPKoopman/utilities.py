import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans

# Plotting Functions


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
