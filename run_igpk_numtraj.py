#!/usr/bin/env python3
"""Run one iGPK experiment for a specified number of training trajectories.

This script is designed to be launched repeatedly from a shell script so that
Each run starts in a fresh Python process and releases RAM / GPU memory when it
exits.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import torch

import GPKoopman as gpk
from get_iGPK_new import find_hp_init, get_iGPK
from TrajDataGen_refactored import generate_dataset


def _nlpd_one(y: torch.Tensor, mu: torch.Tensor, S: torch.Tensor, jitter: float = 1e-8) -> float:
    """
    NLPD for a single multivariate Gaussian y~N(mu,S).
    y, mu: (n,)
    S: (n,n) covariance
    Returns scalar (float)
    """
    n = y.numel()
    S = 0.5 * (S + S.T)
    S = S + jitter * torch.eye(n, dtype=S.dtype, device=S.device)
    try:
        L = torch.linalg.cholesky(S)
        logdet = 2.0 * torch.log(torch.diag(L)).sum()
        diff = (y - mu).view(n, 1)
        sol = torch.cholesky_solve(diff, L)
        quad = float((diff.T @ sol).item())
        return 0.5 * (n * math.log(2.0 * math.pi) + float(logdet) + quad)
    except Exception:
        diag = torch.clamp(torch.diagonal(S), min=jitter)
        logdet = torch.log(diag).sum()
        quad = ((y - mu) ** 2 / diag).sum().item()
        return 0.5 * (n * math.log(2.0 * math.pi) + float(logdet) + quad)


def _nlpd_per_traj(Xhat: torch.Tensor, Xcv: torch.Tensor, GT: torch.Tensor) -> torch.Tensor:
    """
    Average NLPD per trajectory across time-steps.
    returns (nTraj,) tensor
    """
    nTraj, _, N = Xhat.shape
    traj_vals = torch.empty(nTraj, dtype=Xhat.dtype)
    for j in range(nTraj):
        acc = 0.0
        for k in range(N):
            acc += _nlpd_one(
                GT[j, :, k],
                Xhat[j, :, k],
                torch.clamp(torch.abs(Xcv[j, :, :, k]), min=1e-6),
            )
        traj_vals[j] = acc / N
    return traj_vals


def _trajwise_mean(metric: torch.Tensor) -> torch.Tensor:
    metric = torch.as_tensor(metric).detach().cpu()
    if metric.ndim == 1:
        return metric
    if metric.ndim != 2:
        raise ValueError(
            f"Expected 1D or 2D metric tensor, got shape {tuple(metric.shape)}")
    return metric.mean(dim=1)


def _summary_stats(metric: torch.Tensor) -> dict[str, float]:
    x = torch.as_tensor(metric).detach().cpu().reshape(-1)
    return {
        "mean": float(x.mean().item()),
        "median": float(x.median().item()),
    }


def _resolve_device(requested: str) -> str:
    requested = requested.strip()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(
            "[warning] CUDA requested but not available. Falling back to CPU.", flush=True)
        return "cpu"
    return requested


def _write_text_summary(path: Path, payload: dict) -> None:
    lines = [
        f"System: {payload['system']}",
        f"Total Generated Trajectories: {payload['num_total_trajectories']}",
        f"Number of Training Trajectories: {payload['num_train_trajectories']}",
        f"Number of Test Trajectories: {payload['num_test_trajectories']}",
        f"Prediction Steps Used for NLPD: {payload['nlpd_num_time_steps_used']}",
        f"NLPD Time Slice: [{payload['nlpd_first_kept_step']}, {payload['nlpd_last_kept_step']}]",
        f"Computation Time [s]: {payload['computation_time_seconds']:.6f}",
        f"Final Train Cost: {payload['final_train_cost']:.10e}",
        "",
        "Training-set %-NRMSE (trajectory-wise mean across states):",
        f"  Mean   : {100*payload['train_nrmse']['mean']:.2f}",
        f"  Median : {100*payload['train_nrmse']['median']:.2f}",
        "",
        "Test-set %-NRMSE (trajectory-wise mean across states):",
        f"  Mean   : {100*payload['test_nrmse']['mean']:.2f}",
        f"  Median : {100*payload['test_nrmse']['median']:.2f}",
        "",
        "Training-set NLPD (trajectory-wise):",
        f"  Mean   : {payload['train_nlpd']['mean']:.3f}",
        f"  Median : {payload['train_nlpd']['median']:.3f}",
        "",
        "Test-set NLPD (trajectory-wise):",
        f"  Mean   : {payload['test_nlpd']['mean']:.3f}",
        f"  Median : {payload['test_nlpd']['median']:.3f}",
        "",
        "Configuration:",
        json.dumps(payload["config"], indent=2),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one iGPK experiment for a chosen number of training trajectories."
    )
    parser.add_argument("--num-train", type=int, required=True,
                        help="Number of training trajectories.")
    parser.add_argument(
        "--num-test",
        type=int,
        default=20,
        help="Number of held-out test trajectories generated for every run.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="inhibited_predator_prey",
        help="System name passed to TrajDataGen_refactored.generate_dataset(...).",
    )
    parser.add_argument("--lifting-order", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--learn-rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--train-method", type=str, default="Horizon")
    parser.add_argument("--routine", type=str, default="Z_only")
    parser.add_argument("--hp1-scale", type=float, default=4.0)
    parser.add_argument(
        "--opt-weights",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 0.0],
        metavar=("L1", "L2", "L3"),
    )
    parser.add_argument("--seed-z", type=int, default=1234)
    parser.add_argument("--seed-hp", type=int, default=1234)
    parser.add_argument(
        "--outdir",
        type=str,
        default="Figures/numtraj_sweep",
        help="Directory where this run writes its text and JSON summaries.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional filename tag. Default: numtrain_XXXX",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    num_train = int(args.num_train)
    num_test = int(args.num_test)
    if num_train <= 0:
        raise ValueError("--num-train must be positive.")
    if num_test <= 0:
        raise ValueError("--num-test must be positive.")

    total_traj = num_train + num_test
    tag = args.tag or f"numtrain_{num_train:04d}"
    txt_path = outdir / f"{tag}.txt"
    json_path = outdir / f"{tag}.json"

    device = _resolve_device(args.device)

    try:
        print(f"[run] generating dataset with total trajectories = {total_traj} "
              f"({num_train} train + {num_test} test)", flush=True)
        new_data = generate_dataset(args.system, num_trajectories=total_traj)
        SimData_raw = new_data["trajectories"]
        if not torch.is_tensor(SimData_raw):
            SimData_raw = torch.as_tensor(SimData_raw)

        if SimData_raw.shape[0] < total_traj:
            raise RuntimeError(
                f"Generator returned only {SimData_raw.shape[0]} trajectories, but {total_traj} were requested."
            )

        SimData_raw = SimData_raw[:total_traj].detach().cpu()
        N = SimData_raw.shape[2] - 1
        if N <= 4:
            raise ValueError(
                f"Need at least 5 prediction steps to ignore the first and last 2, but got N={N}."
            )

        print(f"[run] normalizing data using training split statistics", flush=True)
        SimData, _, _ = gpk.normalize_data(SimData_raw, num_train, N)

        print(f"[run] estimating kernel lengthscale heuristic", flush=True)
        hp_init = find_hp_init(SimData, num_train)

        print(f"[run] starting iGPK", flush=True)
        t0 = time.perf_counter()
        results = get_iGPK(
            SimData=SimData,
            nTrain=num_train,
            nTest=num_test,
            lifting_order=args.lifting_order,
            max_iter=args.max_iter,
            learn_rate=args.learn_rate,
            opt_weights=list(args.opt_weights),
            routine=args.routine,
            train_method=args.train_method,
            hp_scale=[args.hp1_scale, hp_init, None],
            device=device,
            seed_z=args.seed_z,
            seed_hp=args.seed_hp,
        )
        elapsed = time.perf_counter() - t0

        TrainNRMSE = results["Train"]["NRMSE"].detach().cpu()
        TestNRMSE = results["Test"]["NRMSE"].detach().cpu()
        XhatTrain = results["Train"]["Xhat"].detach().cpu()
        XcvTrain = results["Train"]["Xcv"].detach().cpu()
        XhatTest = results["Test"]["Xhat"].detach().cpu()
        XcvTest = results["Test"]["Xcv"].detach().cpu()
        final_train_cost = float(torch.as_tensor(
            results["final_train_cost"]).item())

        valid_slice = slice(2, N - 2)
        GT_train = SimData[:num_train, :, valid_slice].detach().cpu()
        GT_test = SimData[num_train:num_train +
                          num_test, :, valid_slice].detach().cpu()

        print("[run] computing NLPD summaries", flush=True)
        train_nlpd = _nlpd_per_traj(
            XhatTrain[:, :, valid_slice],
            XcvTrain[:, :, :, valid_slice],
            GT_train,
        )
        test_nlpd = _nlpd_per_traj(
            XhatTest[:, :, valid_slice],
            XcvTest[:, :, :, valid_slice],
            GT_test,
        )

        train_nrmse_traj = _trajwise_mean(TrainNRMSE)
        test_nrmse_traj = _trajwise_mean(TestNRMSE)

        payload = {
            "system": args.system,
            "num_total_trajectories": total_traj,
            "num_train_trajectories": num_train,
            "num_test_trajectories": num_test,
            "nlpd_first_kept_step": 2,
            "nlpd_last_kept_step": N - 3,
            "nlpd_num_time_steps_used": N - 4,
            "computation_time_seconds": elapsed,
            "final_train_cost": final_train_cost,
            "train_nrmse": _summary_stats(train_nrmse_traj),
            "test_nrmse": _summary_stats(test_nrmse_traj),
            "train_nlpd": _summary_stats(train_nlpd),
            "test_nlpd": _summary_stats(test_nlpd),
            "files": {
                "text_summary": str(txt_path),
                "json_summary": str(json_path),
            },
            "config": {
                "lifting_order": args.lifting_order,
                "max_iter": args.max_iter,
                "learn_rate": args.learn_rate,
                "device": device,
                "train_method": args.train_method,
                "routine": args.routine,
                "hp1_scale": args.hp1_scale,
                "hp2_init": hp_init,
                "opt_weights": list(args.opt_weights),
                "seed_z": args.seed_z,
                "seed_hp": args.seed_hp,
            },
        }

        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_text_summary(txt_path, payload)

        print(f"[done] wrote {txt_path}")
        print(f"[done] wrote {json_path}")
        return 0

    except Exception as exc:
        err_path = outdir / f"{tag}_FAILED.txt"
        err_payload = {
            "system": args.system,
            "num_train_trajectories": num_train,
            "num_test_trajectories": num_test,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        err_path.write_text(json.dumps(
            err_payload, indent=2), encoding="utf-8")
        print(
            f"[error] run failed. Details written to {err_path}", file=sys.stderr)
        print(err_payload["traceback"], file=sys.stderr)
        return 1
    finally:
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
