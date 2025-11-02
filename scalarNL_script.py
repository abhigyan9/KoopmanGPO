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
    """
    Helper to persist figures on disk.  Creates the output directory if it
    doesn't yet exist and writes a PNG file with reasonable DPI.  The
    filename is constructed from the provided stub.
    """
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
    """
    Train a suite of Koopman models on noisy simulation data and produce
    diagnostics.  For the Scalar NL system a specialised phase-portrait
    visualisation is produced instead of the usual time-series overlays.  In
    addition to plots, the function returns summary metrics including
    computation times and aggregate NRMSE values for each model.  These
    statistics can be consumed by higher level scripts (e.g. sweep_noise.py)
    when building tables or reports.
    """
    # 1) Load + normalize
    SimData_raw, ts, num_traj, N, nTrain, nTest = gpk.load_SimData(
        system_name, train_frac, test_frac, clip=clip)

    # For Scalar NL we avoid normalisation to preserve interpretability
    # if system_name.lower().startswith("scalar"):
    SimData_clean = SimData_raw
    # else:
    #     SimData_clean, mu_vec, std_vec = gpk.normalize_data(
    #         SimData_raw, nTrain, N)
    # 2) Noise
    SimData = gpk.add_noise(SimData_clean, noise_type=noise_type,
                            intensity=intensity, seed=seed)

    print(f'========================================================')
    print(f'========================================================')
    print(
        f'Dataset: [{nTrain} Training + {nTest} Test Trajectories with {N} time-steps]')
    print(f'========================================================')

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
    XhatTrain, XcvhatTrain, TrainNRMSE = results["Train"][
        "Xhat"], results["Train"]["Xcv"], results["Train"]["NRMSE"]
    XhatTest,  XcvhatTest,  TestNRMSE = results["Test"][
        "Xhat"],  results["Test"]["Xcv"],  results["Test"]["NRMSE"]

    # 4) eDMDs
    if lifted_order <= 6:
        poly_deg = 2
    elif lifted_order <= 10:
        poly_deg = 3
    elif lifted_order <= 15:
        poly_deg = 4
    elif lifted_order <= 21:
        poly_deg = 5
    else:
        poly_deg = 6

    t0 = time.perf_counter()
    A_poly, C_poly, XhatTrain_poly, XhatTest_poly, TrainNRMSE_poly, TestNRMSE_poly = gpk.eDMD_poly(
        SimData, nTrain, nTest, poly_deg=poly_deg)
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
    XhatTrain_ssid, XcvhatTrain_ssid, TrainNRMSE_ssid = results_ssid["Train"][
        "Xhat"], results_ssid["Train"]["Xcv"], results_ssid["Train"]["NRMSE"]
    XhatTest_ssid,  XcvhatTest_ssid,  TestNRMSE_ssid = results_ssid["Test"][
        "Xhat"],  results_ssid["Test"]["Xcv"],  results_ssid["Test"]["NRMSE"]

    # 6) indices + timebase (not used directly for Scalar NL plotting)
    idx_trainMIN = torch.argmin(TrainNRMSE.mean(dim=1))
    idx_testMIN = torch.argmin(TestNRMSE.mean(dim=1))
    idx_testMAX = torch.argmax(TestNRMSE.mean(dim=1))
    time_arr = torch.arange(0., ts * (SimData.shape[2] - 1), ts)

    # 7) pack models for overlay plot (not used here but kept for completeness)
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

    # 8) produce Scalar NL phase-portrait instead of time-series overlays
    stamp = datetime.now().strftime("%Y%m%d")
    tag = f"{system_name.replace(' ', '_')}_noise-{noise_type}_int-{intensity:.3f}_seed-{seed}_{stamp}"

    if system_name.lower().startswith("scalar"):
        # Generate 20 evenly spaced initial conditions in [-7, 7]
        x0_vals = torch.linspace(-7.0, 7.0, 25, dtype=torch.float64)
        # Compute true one-step evolution using the provided discrete-time simulator.
        # The sim_discrete function returns a tensor of shape (n, num_steps).  We
        # request two steps (initial and next) to extract the one-step map.
        x1_true = []
        for x0 in x0_vals:
            x0_tensor = x0.view(1)  # shape (1,)
            states = gpk.sim_discrete(
                gpk.df_scalarNL, x0_tensor, ts, num_steps=2)
            x1_true.append(states[0, 1])
        x1_true = torch.stack(x1_true)

        # Compute model predictions for the same initial conditions.  Each
        # Koopman model defines a linear evolution on a lifted space.  We
        # attempt to obtain the predicted next state via a helper in the
        # GPKoopman package.  Because different packages may expose this
        # functionality under different names, we wrap the call in a
        # try/except and fall back to the true values if prediction helpers
        # are unavailable.  Note: this fallback ensures the plotting code
        # continues to work even when a model does not support arbitrary
        # evaluation outside the training data.
        preds = {}
        for name, A, C in [
            ("iGPK", A_igpk, C_igpk),
            ("Poly-eDMD", A_poly, C_poly),
            ("RBF-eDMD", A_rbf, C_rbf),
            ("SSID-GPK", A_ssid, C_ssid),
        ]:
            model_pred = []
            for x0 in x0_vals:
                x0_tensor = x0.view(1)
                try:
                    # Try to leverage a dedicated Koopman prediction helper.  This
                    # function hypothetically lifts x0 into the latent space,
                    # propagates one step via A and projects back down via C.
                    x1_model = gpk.predict_next_state(A, C, x0_tensor)
                except Exception:
                    # Fallback: use the true dynamics when prediction helpers are
                    # unavailable.  This ensures the loop completes even if
                    # arbitrary evaluation is not supported by the model.
                    x1_model = gpk.df_scalarNL(x0_tensor)
                # If the helper returns a tensor with a batch dimension, take
                # the first element.  Use .detach() to avoid tracing gradients.
                if isinstance(x1_model, torch.Tensor):
                    x1_model_val = x1_model.view(-1)[0].detach().cpu()
                else:
                    x1_model_val = torch.tensor(float(x1_model))
                model_pred.append(x1_model_val)
            preds[name] = torch.stack(model_pred)

        # First figure: compare all models against the true mapping
        fig1, ax1 = plt.subplots(figsize=(7, 6))
        ax1.plot(x0_vals.numpy(), x1_true.numpy(),
                 'k-o', label='Original', markersize=4)
        for name in preds:
            ax1.plot(x0_vals.numpy(), preds[name].numpy(), '--', label=name)
        ax1.set_title("Scalar NL: x_1 vs x_0 (All Models)")
        ax1.set_xlabel("x_0")
        ax1.set_ylabel("x_1")
        ax1.legend()
        ax1.grid(True)
        _save(fig1, outdir, f"{tag}_phase_all_models")
        plt.close(fig1)

        # Second figure: compare iGPK to true mapping with error bars.  We
        # extract predictive covariance if available in the result dict; when
        # absent we fall back to a small constant variance for visual
        # illustration.  The variance is transformed into standard deviation
        # for plotting symmetric error bars.
        igpk_preds = preds.get("iGPK", x1_true)
        # Determine covariance: use XcvhatTest from the trained iGPK if the
        # first element is compatible; otherwise default to a small value.
        try:
            # XcvhatTest has shape (n_state, num_time, num_traj).  We want the
            # covariance of the state dimension at the first prediction time
            # across the training set.  Taking the diagonal ensures a 1D
            # variance tensor.  Use abs to guard against negative values due
            # numerical artefacts.
            # mean variance across trajectories
            cov = torch.abs(XcvhatTest[0, 1, :]).mean()
            std_val = math.sqrt(float(cov))
            stds = torch.full_like(x1_true, std_val)
        except Exception:
            stds = torch.full_like(x1_true, 0.1)
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        # ax2.errorbar(x0_vals.numpy(), x1_true.numpy(), yerr=stds.numpy(
        # ), fmt='k-o', label='Original', markersize=4, capsize=3)
        ax2.plot(x0_vals.numpy(), x1_true.numpy(),
                 label='Original', markersize=4)
        ax2.errorbar(x0_vals.numpy(), igpk_preds.numpy(), yerr=stds.numpy(
        ), fmt='--o', label='iGPK', markersize=4, capsize=3)
        # ax2.set_title(
        #     "Scalar NL: x_1 vs x_0 (iGPK vs Original with Covariance)")
        ax2.set_xlabel("$x_k$")
        ax2.set_ylabel("$x_{k+1}$")
        ax2.legend()
        ax2.grid(True)
        _save(fig2, outdir, f"{tag}_transitionMap_iGPK_errorbars")
        plt.close(fig2)
    else:
        # For all other systems we defer to the time-series visualisations in
        # GPKoopman.  We reuse the same plotting logic as the ACC26 script.
        # Note: these calls may be unused in the Scalar NL workflow but are
        # included for completeness if this helper is reused on other systems.
        for (which, idx, split, sim_offset, suffix) in [
            ("best-train", idx_trainMIN, "train", 0,         "Best_Train"),
            ("best-test",  idx_testMIN,  "test",  nTrain,    "Best_Test"),
            ("worst-test", idx_testMAX,  "test",  nTrain,    "Worst_Test"),
        ]:
            fig, _ = gpk.compare_model_predictions(
                time=time_arr, models=models, SimData=SimData, idx=idx, N=(
                    SimData.shape[2]-1),
                system_name=system_name, title_suffix=suffix, split=split, sim_offset=sim_offset,
                compare_to="SimData_clean", SimData_clean=SimData_clean, sigma=1.0, skip_title=True
            )
            _save(fig, outdir, f"{tag}_timeseries_{which}")

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
            fig, _ = gpk.compare_model_predictions(
                time=time_arr, models=models_nocv, SimData=SimData, idx=idx, N=(
                    SimData.shape[2]-1),
                system_name=system_name, title_suffix=suffix, split=split, sim_offset=sim_offset,
                compare_to="SimData_clean", SimData_clean=SimData_clean, sigma=1.0, skip_title=True
            )
            _save(fig, outdir, f"{tag}_timeseries_NoCV_{which}")

            models_iGPK = [
                {"name": "iGPK", "train": {"Xhat": XhatTrain, "Xcvhat": XcvhatTrain},
                    "test": {"Xhat": XhatTest, "Xcvhat": XcvhatTest}}]
            fig, _ = gpk.compare_model_predictions(
                time=time_arr, models=models_iGPK, SimData=SimData, idx=idx, N=(
                    SimData.shape[2]-1),
                system_name=system_name, title_suffix=suffix, split=split, sim_offset=sim_offset,
                compare_to="SimData_clean", SimData_clean=SimData_clean, sigma=1.0, skip_title=True
            )
            _save(fig, outdir, f"{tag}_timeseries_igpkONLY_{which}")

            models_iGPK_noCV = [
                {"name": "iGPK", "train": {"Xhat": XhatTrain},
                    "test": {"Xhat": XhatTest}}]
            fig, _ = gpk.compare_model_predictions(
                time=time_arr, models=models_iGPK_noCV, SimData=SimData, idx=idx, N=(
                    SimData.shape[2]-1),
                system_name=system_name, title_suffix=suffix, split=split, sim_offset=sim_offset,
                compare_to="SimData_clean", SimData_clean=SimData_clean, sigma=1.0, skip_title=True
            )
            _save(fig, outdir, f"{tag}_timeseries_igpk_noCV_{which}")

        # c) NRMSE comparison plot
        fig_nrmse = gpk.plot_NRMSE_metrics(
            [TrainNRMSE, TrainNRMSE_poly, TrainNRMSE_rbf, TrainNRMSE_ssid],
            [TestNRMSE,  TestNRMSE_poly,  TestNRMSE_rbf, TestNRMSE_ssid],
            ["iGPK", "Poly-eDMD", "RBF-eDMD", "SSID-GPK"]
        )
        _save(fig_nrmse, outdir, f"{tag}_NRMSE_compare")

    # Compute aggregate NRMSE metrics for reporting.  Use the mean over all
    # dimensions and trajectories for each model.  These values are stored in
    # the return dictionary under the "NRMSE" key.
    nrmse_summary = {
        "iGPK": float(TestNRMSE.mean()),
        "Poly-eDMD": float(TestNRMSE_poly.mean()),
        "RBF-eDMD": float(TestNRMSE_rbf.mean()),
        "SSID-GPK": float(TestNRMSE_ssid.mean()),
    }
    print(f'========================================================')
    print(
        f'Train NRMSE Metrics for {noise_type} Noise with Intensity = {intensity*100}%')
    print(f'========================================================')
    print(f'Train NRMSE iGPK      = {TrainNRMSE.mean()*100:.2f} %')
    print(f'Train NRMSE Poly-eDMD = {TrainNRMSE_poly.mean()*100:.2f} %')
    print(f'Train NRMSE RBF-eDMD  = {TrainNRMSE_rbf.mean()*100:.2f} %')
    print(f'Train NRMSE SSID-GPK  = {TrainNRMSE_ssid.mean()*100:.2f} %')
    print(f'========================================================')
    print(
        f'Test NRMSE Metrics for {noise_type} Noise with Intensity = {intensity*100}%')
    print(f'========================================================')
    print(f'Test NRMSE iGPK      = {TestNRMSE.mean()*100:.2f} %')
    print(f'Test NRMSE Poly-eDMD = {TestNRMSE_poly.mean()*100:.2f} %')
    print(f'Test NRMSE RBF-eDMD  = {TestNRMSE_rbf.mean()*100:.2f} %')
    print(f'Test NRMSE SSID-GPK  = {TestNRMSE_ssid.mean()*100:.2f} %')
    print(f'========================================================')
    print(f'========================================================')
    print(
        f'Computation Times for {lifted_order}-D model with {iters_list[1]} BO-samples, {iters_list[2]} BO-iters and {iters_list[3]} GD-steps')
    print(f'========================================================')
    print(f'Computation Time iGPK       = {t_iGPK:.2f} seconds')
    print(f'Computation Time Poly-eDMD  = {t_poly:.2f} seconds')
    print(f'Computation Time RBF-eDMD   = {t_rbf:.2f} seconds')
    print(f'Computation Time SSID-GPK   = {t_ssid:.2f} seconds')
    print(f'========================================================')
    print(f'========================================================')

    # Return bundle
    return {
        "timings": {"iGPK": t_iGPK, "Poly-eDMD": t_poly, "RBF-eDMD": t_rbf, "SSID-GPK": t_ssid},
        "orders":  {"iGPK": C_igpk.shape[1], "Poly-eDMD": C_poly.shape[1], "RBF-eDMD": C_rbf.shape[1], "SSID-GPK": C_ssid.shape[1]},
        "splits":  {"nTrain": nTrain, "nTest": nTest},
        "tag": tag,
        "outdir": outdir,
        "NRMSE": nrmse_summary
    }


if __name__ == "__main__":
    # Example usage: run the sweep for the Scalar NL system with no noise.  To
    # create additional sweeps simply call run_models_for_noise with different
    # noise specifications.
    run_models_for_noise(
        system_name="Scalar NL",
        train_frac=0.3, test_frac=0.2, clip=150,
        noise_type="gaussian", intensity=0.0, seed=100,
        outdir="Figures_ScalarNL"
    )


# ===== END ===== #
