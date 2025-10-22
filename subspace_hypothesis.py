import GPKoopman as gpk
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from get_iGPK_fcn import get_iGPK
import os
from datetime import datetime


system_name = 'Simple Pendulum'
train_frac, test_frac = 0.8, 0.2
clip = None
lifted_order = 10

# 1) Load + normalize
SimData_raw, ts, num_traj, N, nTrain, nTest = gpk.load_SimData(
    # :contentReference[oaicite:9]{index=9}
    system_name, train_frac, test_frac, clip=clip)
SimData_clean, mu_vec, std_vec = gpk.normalize_data(
    SimData_raw, nTrain, N)  # :contentReference[oaicite:10]{index=10}

# 2) Noise
# SimData = gpk.add_noise(SimData_clean, noise_type=noise_type,
#                     intensity=intensity, seed=seed)
SimData = SimData_clean
print('Data Loading completed!')
# 3) Do Polynomial and RBF eDMD
t0 = time.perf_counter()
A_eDMD_poly, C_poly, _, _, _, _ = gpk.eDMD_poly(
    SimData, nTrain, nTest, poly_deg=3)
t_poly = time.perf_counter() - t0
print(f'Polynomial eDMD finished in {t_poly:.2f} seconds.')
t0 = time.perf_counter()
A_rbf, C_rbf, _, _, _, _ = gpk.eDMD_RBF_kmeans(
    SimData, nTrain, nTest, num_centers=lifted_order, width=0.2, rbf_type='thin_plate', state_aug=True)
t_rbf = time.perf_counter() - t0
print(f'K-Means RBF eDMD finished in {t_rbf:.2f} seconds.')

# 4) Do SSID
t0 = time.perf_counter()
results_ssid = gpk.get_ssidgpk(
    SimData=SimData,
    nTrain=nTrain, nTest=nTest,
    lifting_order=lifted_order,
    delay=N - 1)
t_ssid = time.perf_counter() - t0
print(f'SSID-GPK finished in {t_ssid:.2f} seconds.')

# unpack SSID-GPK results
A_ssid, C_ssid = results_ssid["A"], results_ssid["C"]
SS_ObsManager = results_ssid["ObsManager"]

# 5) Do eDMD with SSID-GPK Observables
t0 = time.perf_counter()
X = torch.cat([SimData[j, :, 0:N]
               for j in range(nTrain)], dim=1)     # n x (nTrain*N)
Xplus = torch.cat([SimData[j, :, 1:]
                   for j in range(nTrain)], dim=1)     # n x (nTrain*N)

M, Mplus = torch.empty((lifted_order, N*nTrain), dtype=X.dtype
                       ), torch.empty((lifted_order, N*nTrain), dtype=X.dtype)
for i in range(lifted_order):
    M[i, :] = SS_ObsManager.predict_mean(i, X)
    Mplus[i, :] = SS_ObsManager.predict_mean(i, Xplus)

Mpinv = torch.linalg.pinv(M)
A_ssid_eDMD = Mplus @ Mpinv
C_ssid_eDMD = X @ Mpinv
t_ssid_eDMD = time.perf_counter() - t0
print(
    f'eDMD with GP Observables from SSID-GPK finished in {t_ssid_eDMD:.2f} seconds.')

# 6) Plotting
gpk.plot_eigen(A_eDMD_poly)

gpk.plot_eigen(A_ssid)

gpk.plot_eigen(A_ssid_eDMD)

gpk.MatViz(C_poly, 'heat')
gpk.MatViz(C_ssid, 'heat')
gpk.MatViz(C_ssid_eDMD, 'heat')

eig_A_ssid = torch.linalg.eigvals(A_ssid)
eig_A_ssid_edmd = torch.linalg.eigvals(A_ssid_eDMD)
eig_A_edmd = torch.linalg.eigvals(A_eDMD_poly)

print(f'\n==========================\n')
print(f'\nEigenvalues from eDMD are:\n{eig_A_edmd}')
print(f'\n==========================\n')
print(f'\nEigenvalues from SSID are:\n{eig_A_ssid}')
print(f'\n==========================\n')
print(
    f'\nEigenvalues from eDMD with SSID-GPK-Observables are:\n{eig_A_ssid_edmd}')
print(f'\n==========================\n')

plt.show()
