## --- IMPORTS --- ###
import GPKoopman as gpk
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from get_iGPK_fcn import get_iGPK
import os
from datetime import datetime

## --- HELPER FUNCTIONS --- ##


def _save(fig, outdir, fname_stub: str):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{fname_stub}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"saved: {path}")


def run_models_for_noise(
    system_name: str,
    train_frac: float,
    test_frac: float,
    clip: int | None,
    noise_type: str,
    intensity: float,
    seed: int,
    # modeling knobs
    lifted_order: int = 10,
    iters_list=(250, 50, 50, 100),
    learn_rate: float = 0.04,
    opt_weights=(10.0, 1.0, 10.0),
    routine: str = "Z_only",
    train_method: str = "Horizon",
    device: str = "cuda:0",
    # saving
    outdir: str = "Figures"
):
    # 1) Load + normalize
    SimData_raw, ts, num_traj, N, nTrain, nTest = gpk.load_SimData(
        system_name, train_frac, test_frac, clip=clip)
    # SimData_clean, mu_vec, std_vec = gpk.normalize_data(
    #     SimData_raw, nTrain, N)
    SimData_clean = SimData_raw
    # 2) Noise
    SimData = gpk.add_noise(SimData_clean, noise_type=noise_type,
                            intensity=intensity, seed=seed)

    # 3) iGPK
    t0 = time.perf_counter()
    results = get_iGPK(
        SimData=SimData,
        nTrain=nTrain, nTest=nTest,
        lifting_order=lifted_order,
        iters_list=list(iters_list),
        learn_rate=learn_rate,
        opt_weights=list(opt_weights),
        routine=routine,
        train_method=train_method,
        device=device
    )
    t_iGPK = time.perf_counter() - t0

    # unpack iGPK
    A_igpk, C_igpk = results["A"], results["C"]
    # ICsetTrain, ICsetTest = results["ICsetTrain"], results["ICsetTest"]
    XhatTrain, XcvhatTrain, TrainNRMSE = results["Train"][
        "Xhat"], results["Train"]["Xcv"], results["Train"]["NRMSE"]
    XhatTest,  XcvhatTest,  TestNRMSE = results["Test"][
        "Xhat"],  results["Test"]["Xcv"],  results["Test"]["NRMSE"]

    # 4) eDMDs
    t0 = time.perf_counter()
    A_poly, C_poly, XhatTrain_poly, XhatTest_poly, TrainNRMSE_poly, TestNRMSE_poly = gpk.eDMD_poly(
        SimData, nTrain, nTest, poly_deg=3)
    t_poly = time.perf_counter() - t0

    t0 = time.perf_counter()
    A_rbf, C_rbf, XhatTrain_rbf, XhatTest_rbf, TrainNRMSE_rbf, TestNRMSE_rbf = gpk.eDMD_RBF_kmeans(
        SimData, nTrain, nTest, num_centers=lifted_order, width=0.2, rbf_type='thin_plate', state_aug=True)
    t_rbf = time.perf_counter() - t0

    # 5) SSID-GPK
    t0 = time.perf_counter()
    results_ssid = gpk.get_ssidgpk(
        SimData=SimData,
        nTrain=nTrain, nTest=nTest,
        lifting_order=lifted_order,
        delay=N - 1)
    t_ssid = time.perf_counter() - t0

    # unpack SSID-GPK results
    A_ssid, C_ssid = results_ssid["A"], results_ssid["C"]
    # ICsetTrain_ssid, ICsetTest_ssid = results["ICsetTrain"], results["ICsetTest"]
    XhatTrain_ssid, XcvhatTrain_ssid, TrainNRMSE_ssid = results_ssid["Train"][
        "Xhat"], results_ssid["Train"]["Xcv"], results_ssid["Train"]["NRMSE"]
    XhatTest_ssid,  XcvhatTest_ssid,  TestNRMSE_ssid = results_ssid["Test"][
        "Xhat"],  results_ssid["Test"]["Xcv"],  results_ssid["Test"]["NRMSE"]

    # 6) indices + timebase
    idx_trainMIN = torch.argmin(TrainNRMSE.mean(dim=1))
    idx_testMIN = torch.argmin(TestNRMSE.mean(dim=1))
    idx_testMAX = torch.argmax(TestNRMSE.mean(dim=1))
    time_arr = torch.arange(0., ts * (SimData.shape[2] - 1), ts)

    # 7) pack models for overlay plot
    models = [
        {"name": "iGPK", "train": {"Xhat": XhatTrain, "Xcvhat": XcvhatTrain},
            "test": {"Xhat": XhatTest, "Xcvhat": XcvhatTest}},
        {"name": "Poly-eDMD", "train": {"Xhat": XhatTrain_poly},
            "test": {"Xhat": XhatTest_poly}},
        {"name": "RBF-eDMD",  "train": {"Xhat": XhatTrain_rbf},
            "test": {"Xhat": XhatTest_rbf}},
        {"name": "SSID-GPK", "train": {"Xhat": XhatTrain_ssid, "Xcvhat": XcvhatTrain_ssid},
            "test": {"Xhat": XhatTest_ssid, "Xcvhat": XcvhatTest_ssid}}
    ]

    models_nocv = [
        {"name": "iGPK", "train": {"Xhat": XhatTrain},
            "test": {"Xhat": XhatTest}},
        {"name": "Poly-eDMD", "train": {"Xhat": XhatTrain_poly},
            "test": {"Xhat": XhatTest_poly}},
        {"name": "RBF-eDMD",  "train": {"Xhat": XhatTrain_rbf},
            "test": {"Xhat": XhatTest_rbf}},
        {"name": "SSID-GPK", "train": {"Xhat": XhatTrain_ssid},
            "test": {"Xhat": XhatTest_ssid}}
    ]

    # 8) make & save all figures
    stamp = datetime.now().strftime("%Y%m%d")
    tag = f"{system_name.replace(' ', '_')}_noise-{noise_type}_int-{intensity:.3f}_seed-{seed}_{stamp}"

    # a) 3 trajectory overlays
    for (which, idx, split, sim_offset, suffix) in [
        ("best-train", idx_trainMIN, "train", 0,         "Best_Train"),
        ("best-test",  idx_testMIN,  "test",  nTrain,    "Best_Test"),
        ("worst-test", idx_testMAX,  "test",  nTrain,    "Worst_Test"),
    ]:
        fig, _ = gpk.compare_model_predictions(
            time=time_arr, models=models, SimData=SimData, idx=idx, N=(
                SimData.shape[2]-1),
            system_name=system_name, title_suffix=suffix, split=split, sim_offset=sim_offset,
            compare_to="SimData_clean", SimData_clean=SimData_clean, sigma=1.0
        )
        _save(fig, outdir, f"{tag}_timeseries_{which}")

        fig, _ = gpk.compare_model_predictions(
            time=time_arr, models=models_nocv, SimData=SimData, idx=idx, N=(
                SimData.shape[2]-1),
            system_name=system_name, title_suffix=suffix, split=split, sim_offset=sim_offset,
            compare_to="SimData_clean", SimData_clean=SimData_clean, sigma=1.0
        )
        _save(fig, outdir, f"{tag}_timeseries_NoCV_{which}")

    # b) Eigen (iGPK)
    fig_eig = gpk.plot_eigen(A_igpk)
    _save(fig_eig, outdir, f"{tag}_eig_igpk")

    # c) NRMSE comparison
    fig_nrmse = gpk.plot_NRMSE_metrics(
        [TrainNRMSE, TrainNRMSE_poly, TrainNRMSE_rbf, TrainNRMSE_ssid],
        [TestNRMSE,  TestNRMSE_poly,  TestNRMSE_rbf, TestNRMSE_ssid],
        ["iGPK", "Poly-eDMD", "RBF-eDMD", "SSID-GPK"]
    )
    _save(fig_nrmse, outdir, f"{tag}_NRMSE_compare")

    # 9) small return bundle (optional)
    return {
        "timings": {"iGPK": t_iGPK, "Poly-eDMD": t_poly, "RBF-eDMD": t_rbf, "SSID-GPK": t_ssid},
        "orders":  {"iGPK": C_igpk.shape[1], "Poly-eDMD": C_poly.shape[1], "RBF-eDMD": C_rbf.shape[1], "SSID-GPK": C_ssid.shape[1]},
        "splits":  {"nTrain": nTrain, "nTest": nTest},
        "tag": tag,
        "outdir": outdir
    }


if __name__ == "__main__":
    # Optional: a single manual call for quick test
    run_models_for_noise(
        system_name="Simple Pendulum",
        train_frac=0.3, test_frac=0.2, clip=150,
        noise_type="linear_gaussian", intensity=0.0, seed=100,
        outdir="Figures_Trial"
    )


# ===== END ===== #
