import GPKoopman as gpk
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display, clear_output
import math
import gc


def get_cost(Z, X, Xplus, Xtrain, manager, nT=1, lambda1=1.0, lambda2=1.0, lambda3=1.0):
    """
    Computes the cost function as defined in the Word doc, with different GP Kernel hyperparameters for
    each observable.

    Args:
        Z: Tensor of shape (r**n, p), decision variable, required grad
        X: Tensor of shape (n, nT*N), dataset of N steps from all trajectories
        Xall: Tensor of shape (n, nT*(N+1)), complete training dataset
        Xtrain: Tensor of shape (n, r**n), flattened set of gridpoints for training GPOs
        manager: Object of class GPObservablesManager (manager for all Gaussian Process based Observable functions)
        nT: float, number of trajectories in training dataset
        lambda1: float, Weighting for prediction error minimization term
        lambda2: float, Weightining for Reconstruction Error penalty term
    """

    N = (X.shape[1])//nT    # Number of time steps in each trajectory
    p = Z.shape[1]          # Number of Observables
    l = Z.shape[0]//nT      # Decision Horizon
    n = X.shape[0]          # Dimensionality of original system

    for i in range(p):
        manager.train_observable(i, Xtrain, Z[:, i])

    # For current definition of GPOs
    # Training: Xtrain = dimensions x samples
    # Training: Ytrain = samples x (dimensions=1)
    # Prediction: Xquery = dimensions x num-query = Input
    # Prediction: Yquery = num-query x (dimensions=1) = Output

    # Lifting X and Xplus to higher dimension using trained GPOs
    M = torch.empty((p, N*nT))
    Mplus = torch.empty((p, N*nT))

    # Mall = torch.empty((p,(N+1)*nT))
    for i in range(p):
        M[i, :] = torch.transpose(manager.predict_mean(i, X), dim0=0, dim1=-1)
        Mplus[i, :] = torch.transpose(
            manager.predict_mean(i, Xplus), dim0=0, dim1=-1)

    # Compute C(z) and A(z)
    Mfull = torch.vstack((X, M))
    Mplusfull = torch.vstack((Xplus, Mplus))

    M_pinv = torch.linalg.pinv(Mfull)
    # Cz = X @ M_pinv
    Az = Mplusfull @ M_pinv
    C = torch.zeros((n, n+p))
    for i in range(n):
        C[i, i] = 1.

    # Cost term 1: Multi-Trajectory Prediction Error Minimization
    NormPEM = 0.0
    for j in range(nT):
        TrajPEM = 0.0
        for k in range(N - 1):
            # multi-step at X (with Cz)
            pred_error = X[:, j*N + (k+1)] - \
                C @ (torch.linalg.matrix_power(Az, k+1)) @ Mfull[:, j*N]
            TrajPEM += torch.norm(pred_error)
        NormPEM += TrajPEM

    # Linearity Enforcement
    NormLEP = 0.0
    for j in range(nT):
        TrajLEP = 0.0
        for k in range(l-1):
            Zk = torch.vstack(
                [X[:, j*N + k].view(n, 1), torch.transpose(Z[j*l + k, :], dim0=0, dim1=-1).view(p, 1)])
            Zkplus = torch.vstack(
                [X[:, j*N + k + 1].view(n, 1), torch.transpose(Z[j*l + k + 1, :], dim0=0, dim1=-1).view(p, 1)])
            lin_error = Zkplus - Az @ Zk
            TrajLEP += torch.norm(lin_error)
        NormLEP += TrajLEP

    # Weighted sum of terms
    cost = (lambda1 * NormPEM / (N * nT)) + (lambda2 * NormLEP / (l * nT))
    return cost


def compute_metrics(Xhat, Xcvhat, SimData, nTraj, N, eps=1e-6):
    RMSE = torch.sqrt(torch.mean((Xhat - SimData[:nTraj, :, :N]) ** 2, dim=2))
    return RMSE


def get_iGPOK(system_name, p, l, max_iter=None, clipto=None):
    # Explanation

    # Load train and test datasets based on system_name
    data = torch.load(f'Data/DataAuto_{system_name}.pt', weights_only=True)
    # Shape: (num_trajectories, state_dim, num_steps)
    SimData = data['trajectories']
    SimData = SimData.float()   # dtype = torch.float32 to save memory | reduces accuracy
    ts = data['sample_time']
    num_trajectories = data['num_trajectories']
    nx = SimData.shape[1]
    N = data['num_steps']
    if clipto is not None:
        SimData = SimData[:, :, :clipto+1]
        N = clipto

    # Assuming 100 trajectories, 80 : training, 20 : test
    # Training Set = 80 trajectories
    nTrain = math.floor(num_trajectories * 0.8)
    nTest = math.floor(num_trajectories * 0.2)  # Test Set = 20 trajectories
    ICsetTrain = torch.cat([SimData[j, :, 0].view(nx, 1) for j in range(
        nTrain)], dim=1)    # Random IC set for training
    ICsetTest = torch.cat([SimData[j, :, 0].view(nx, 1) for j in range(
        nTrain, nTrain + nTest)], dim=1)  # Random IC set for testing

    # Prepare Data and Initialize Observables
    Xall = torch.cat([SimData[j, :, :] for j in range(nTrain)],
                     dim=1)      # Concatenated total matrix
    X = torch.cat([SimData[j, :, 0:N] for j in range(nTrain)],
                  dim=1)       # Concatenated Data matrix
    # Time-shifted Data matrix
    Xplus = torch.cat([SimData[j, :, 1:] for j in range(nTrain)], dim=1)

    Xtrain = torch.cat([X[:, j*N:j*N+l] for j in range(nTrain)], dim=1)
    Z = torch.rand(l*nTrain, p, requires_grad=True)
    # initialize Observable Manager class object
    ObsManager = gpk.GPObservablesManager()
    for i in range(p):
        ObsManager.add_observable(
            index=i, d=nx, ns=l*nTrain, kernel_types=['Gaussian'], noise=1e-6)

    ObsManager.set_random_hyperparameters(
        scale=1.0)    # randomize hyperparameters

    # Initialize Optimizer
    if max_iter is None:
        max_iter = 300

    lr, err_thresh, patience, min_delta = 0.001, 0.01, 30, 5e-3
    optimizer = torch.optim.Adam([Z, ObsManager.get_all_params()], lr=lr)
    cost_history, iter, count_insig = [], 0, 0

    # Optimization Routine
    while iter < max_iter:
        optimizer.zero_grad()  # Clear gradients
        cost = get_cost(Z, X, Xplus, Xtrain, ObsManager,
                        nT=nTrain, lambda1=10.)   # compute cost
        cost_history.append(cost.item())    # add to cost history
        cost.backward(retain_graph=True)    # backpropagate
        optimizer.step()    # gradient descent step

        # Stopping conditions
        if cost.item() < err_thresh:
            print("Stopping: Error threshold reached.")
            break

        # Check for significant improvement
        if iter > patience:
            error_change = cost_history[-patience] - cost_history[-1]
            if error_change < min_delta or error_change < 0:
                count_insignificant += 1
            else:
                count_insignificant = 0

            if count_insignificant >= patience:
                print(
                    "Stopping: No significant improvement over consecutive iterations.")
                break

        # Increment iteration
        iter += 1

    plt.figure(figsize=(8, 6))
    plt.plot(cost_history, label="Cost")
    plt.title("Cost History")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.legend()
    plt.grid()
    plt.savefig(
        f'Plots/{system_name}/CostHistory_{system_name}_p{p}_l{l}.png', dpi=300, bbox_inches='tight')

    # Post Process Optimization Results
    optimal_Z = Z.detach()
    for i in range(p):
        # train GP Observables with Optimal Z outputs
        ObsManager.train_observable(i, Xtrain, optimal_Z[:, i])

    # Optimize Kernel hyperparameters for Optimal training data
    ObsManager.optimize_hyperparameters()
    A, C = gpk.getKoopman(
        ObsManager, [i for i in range(p)], Xall, nTrain, stateAug=True)

    # Model Evaluation
    # Evaluation on training set
    ZmeanTrain, ZcvTrain = torch.empty(
        (nTrain, nx+p, N)), torch.empty((nTrain, nx+p, nx+p, N))
    ZmeanTrain[:, :nx, 0] = ICsetTrain.T

    XhatTrain, XcvhatTrain = torch.empty(
        (nTrain, nx, N)), torch.empty((nTrain, nx, nx, N))
    TrainRMSE = torch.empty((nTrain, nx))

    for j in range(nTrain):  # GPK Predict for all Training trajectories
        for i in range(p):  # GP predict IC and IC-cv
            ZmeanTrain[j, nx+i,
                       0] = ObsManager.predict_mean(i, ICsetTrain[:, j].view(nx, 1))
            ZcvTrain[j, nx+i, nx+i,
                     0] = ObsManager.predict_covariance(i, ICsetTrain[:, j].view(nx, 1))

        ZmeanTrain[j, :, :], ZcvTrain[j, :, :, :], XhatTrain[j, :, :], XcvhatTrain[j, :, :, :] = gpk.sim_LTI(
            ZmeanTrain[j, :, 0], A, C, num_steps=N, ts=None, x0cv=ZcvTrain[j, :, :, 0])
        TrainRMSE[j, :] = torch.sqrt(torch.mean(
            (XhatTrain[j, :, :] - SimData[j, :, :N])**2, 1))

    # Evaluation on test set
    ZmeanTest = torch.empty((nTest, nx+p, N))
    ZcvTest = torch.empty((nTest, nx+p, nx+p, N))
    ZmeanTest[:, :nx, 0] = ICsetTest.T

    XhatTest = torch.empty((nTest, nx, N))
    XcvhatTest = torch.empty((nTest, nx, nx, N))
    TestRMSE = torch.empty((nTest, nx))

    for j in range(nTest):  # GPK Predict for all Test trajectories
        for i in range(p):  # GP predict IC and IC-cv
            ZmeanTest[j, nx+i,
                      0] = ObsManager.predict_mean(i, ICsetTest[:, j].view(nx, 1))
            ZcvTest[j, nx+i, nx+i,
                    0] = ObsManager.predict_covariance(i, ICsetTest[:, j].view(nx, 1))

        ZmeanTest[j, :, :], ZcvTest[j, :, :, :], XhatTest[j, :, :], XcvhatTest[j, :, :, :] = gpk.sim_LTI(
            ZmeanTest[j, :, 0], A, C, num_steps=N, ts=None, x0cv=ZcvTest[j, :, :, 0])
        TestRMSE[j, :] = torch.sqrt(torch.mean(
            (XhatTest[j, :, :] - SimData[nTest+j, :, :N])**2, 1))

    time = torch.arange(0., ts * N, ts)
    idx = 20 - 1
    XhatTrain, XhatTest = XhatTrain.detach(), XhatTest.detach()
    XcvhatTrain, XcvhatTest = XcvhatTrain.detach(), XcvhatTest.detach()

    # Training Trajectory Phase Plot
    plt.figure(figsize=(8, 6))
    plt.plot(XhatTrain[idx, 0, :], XhatTrain[idx, 1, :], label='iGPK')
    plt.plot(SimData[idx, 0, :N], SimData[idx, 1, :N],
             label='Nonlinear', linestyle='--')
    plt.plot(ICsetTrain[0, idx], ICsetTrain[1, idx], label='IC', marker='o')
    plt.title(f"Validation on Training Trajectory for {system_name}")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid()
    plt.savefig(
        f'Plots/{system_name}/PhasePlotTrain_{system_name}_p{p}_l{l}.png', dpi=300, bbox_inches='tight')

    # Training Trajectory Time Plot with uncertainty bound
    plt.figure(figsize=(8, 6))
    plt.fill_between(time, XhatTrain[idx, 0, :] - 3 * XcvhatTrain[idx, 0, 0, :] ** 0.5,
                     XhatTrain[idx, 0, :] + 3 * XcvhatTrain[idx, 0, 0, :] ** 0.5, alpha=0.3, color='blue')
    plt.fill_between(time, XhatTrain[idx, 1, :] - 3 * XcvhatTrain[idx, 1, 1, :] ** 0.5,
                     XhatTrain[idx, 1, :] + 3 * XcvhatTrain[idx, 1, 1, :] ** 0.5, alpha=0.3, color='red')
    plt.plot(time, XhatTrain[idx, 0, :], label='X1 - iGPK', color='blue')
    plt.plot(time, XhatTrain[idx, 1, :], label='X2 - iGPK', color='red')
    plt.plot(time, SimData[idx, 0, :N], label='X1 - NL', linestyle='--')
    plt.plot(time, SimData[idx, 1, :N], label='X2 - NL', linestyle='--')
    plt.xlabel('Time [s]')
    plt.ylabel('State')
    plt.title(f'{system_name}: States in Training with 3$\\sigma$ bound')
    plt.legend()
    plt.grid()
    plt.savefig(
        f'Plots/{system_name}/TimePlotTrain_{system_name}_p{p}_l{l}.png', dpi=300, bbox_inches='tight')

    # Test Trajectory Plot
    plt.figure(figsize=(8, 6))
    plt.plot(XhatTest[idx, 0, :], XhatTest[idx, 1, :], label='iGPK')
    plt.plot(SimData[nTrain+idx, 0, :N], SimData[nTrain +
             idx, 1, :N], label='Nonlinear', linestyle='--')
    plt.plot(ICsetTest[0, idx], ICsetTest[1, idx], label='IC', marker='o')
    plt.title(f"Validation on Test Trajectory for {system_name}")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid()
    plt.savefig(
        f'Plots/{system_name}/PhasePlotTest_{system_name}_p{p}_l{l}.png', dpi=300, bbox_inches='tight')

    # Test Trajectory Time Plot with uncertainty bound
    plt.figure(figsize=(8, 6))
    plt.fill_between(time, XhatTest[idx, 0, :] - 3 * XcvhatTest[idx, 0, 0, :] ** 0.5,
                     XhatTest[idx, 0, :] + 3 * XcvhatTest[idx, 0, 0, :] ** 0.5, alpha=0.3, color='blue')
    plt.fill_between(time, XhatTest[idx, 1, :] - 3 * XcvhatTest[idx, 1, 1, :] ** 0.5,
                     XhatTest[idx, 1, :] + 3 * XcvhatTest[idx, 1, 1, :] ** 0.5, alpha=0.3, color='red')
    plt.plot(time, XhatTest[idx, 0, :], label='X1 - iGPK', color='blue')
    plt.plot(time, XhatTest[idx, 1, :], label='X2 - iGPK', color='red')
    plt.plot(time, SimData[nTrain+idx, 0, :N], label='X1 - NL', linestyle='--')
    plt.plot(time, SimData[nTrain+idx, 1, :N], label='X2 - NL', linestyle='--')
    plt.xlabel('Time [s]')
    plt.ylabel('State')
    plt.title(f'{system_name}: States in Test with 3$\\sigma$ bound')
    plt.legend()
    plt.grid()
    plt.savefig(
        f'Plots/{system_name}/TimePlotTest_{system_name}_p{p}_l{l}.png', dpi=300, bbox_inches='tight')

    # RMSE Plot
    # Compute metrics for training set
    TrainRMSE = compute_metrics(
        XhatTrain, XcvhatTrain, SimData[:nTrain, :, :], nTrain, N)
    # Compute metrics for test set
    TestRMSE = compute_metrics(
        XhatTest, XcvhatTest, SimData[nTrain:(nTrain+nTest), :, :], nTest, N)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Training set plot
    axes[0].plot(range(nTrain), TrainRMSE.mean(dim=1).numpy(),
                 marker='o', linestyle='-', label='RMSE')
    axes[0].set_title('Training Metrics')
    axes[0].set_xlabel("Trajectory Index")
    axes[0].set_ylabel("Metric Value")
    axes[0].legend()
    axes[0].grid()

    # Test set plot
    axes[1].plot(range(nTest), TestRMSE.mean(dim=1).numpy(),
                 marker='o', linestyle='-', label='RMSE')
    axes[1].set_title('Test Metrics')
    axes[1].set_xlabel("Trajectory Index")
    axes[1].set_ylabel("Metric Value")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.savefig(
        f'Plots/{system_name}/ErrorPlot_{system_name}_p{p}_l{l}.png', dpi=300, bbox_inches='tight')

    # A Matrix Eigenvalues and Heatmap
    eigval = torch.linalg.eigvals(A)
    eigreal, eigimag = eigval.real, eigval.imag
    eigreal, eigimag = eigreal.detach().numpy(), eigimag.detach().numpy()
    theta = np.linspace(0, 2*np.pi, 500)
    unitCirclex, unitCircley = np.cos(theta), np.sin(theta)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # First subplot: Eigenvalues plot
    axes[0].plot(unitCirclex, unitCircley, color='red', label='Unit Circle')
    axes[0].scatter(eigreal, eigimag, color='blue', label='Eigenvalues')
    axes[0].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes[0].axvline(0, color='black', linewidth=0.5, linestyle='--')
    axes[0].set_title(f"Eigenvalues of A Matrix with {p} Observables")
    axes[0].set_xlabel("Real Part")
    axes[0].set_ylabel("Imaginary Part")
    axes[0].grid(True)
    axes[0].legend(loc='upper right')

    # Second subplot: Heatmap of matrix A
    im = axes[1].imshow(A.detach().numpy(), cmap='viridis', aspect='auto')
    fig.colorbar(im, ax=axes[1], label="Value")
    axes[1].set_title(f'{A.shape[0]}-D Koopman Matrix')
    axes[1].set_xlabel("Columns")
    axes[1].set_ylabel("Rows")

    plt.savefig(f'Plots/{system_name}/MatrixPlot_{system_name}_p{
                p}_l{l}.png', dpi=300, bbox_inches='tight')

    plt.close('all')    # Close all matplotlib figures to conserve memory

    # Save model
    GPKmodel = {
        "ObsManager": ObsManager,
        "Observables": ObsManager.observables,
        "A matrix": A,
        "C matrix": C,
        "Optimal Z": optimal_Z,
        "X Train": Xtrain,
        "Train RMSE": TrainRMSE,
        "Test RMSE": TestRMSE
    }

    torch.save(GPKmodel, f"Models/GPKModelAuto_{system_name}_p{p}_l{l}.pt")
    print(f'Model saved to Models/GPKModelAuto_{system_name}_p{p}_l{l}.pt')

    # Garbage collection inside the function
    gc.collect()                # clear unreferenced objects from memory
    torch.cuda.empty_cache()    # clear PyTorch GPU memory

    pass

# Allowed system names -
# "Unforced Duffing"
# "van der Pol"
# "Simple Pendulum"
# "Lorenz"
# "Lotka Volterra"


if __name__ == "__main__":

    for i in range(1, 4):   # Iterating through decision horizon
        for j in range(1, 5):   # Iterating through decision horizon
            clip = 150
            # Decision Horizon steps of 5, 10, 15 % of N
            l = math.floor(0.05*clip*i)
            p = 5*j
            print(
                f'Starting iGPK model building for all systems with p={p}, l={l}')
            get_iGPOK("Simple Pendulum", p=p, l=l, clipto=150)
            get_iGPOK("Unforced Duffing", p=p, l=l, clipto=150)
            get_iGPOK("van der Pol", p=p, l=l, clipto=150)
            get_iGPOK("Lotka Volterra", p=p, l=l, clipto=150)
            # get_iGPOK("Lorenz", p=5*i, l=10*(j+1))
            print(
                f'Finished iGPK model building for all systems with current parameters.')
