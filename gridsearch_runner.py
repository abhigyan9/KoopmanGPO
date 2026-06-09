import argparse
import gc
import json
import math
import os
import traceback
from datetime import datetime
from pathlib import Path

import torch

import GPKoopman as gpk
from get_iGPK_fcn import get_iGPK

import warnings
warnings.filterwarnings("ignore")


# -----------------------------
# Default experiment configuration
# -----------------------------
SYSTEM_NAME = "Inhibited Predator-Prey"
TRAIN_FRAC = 0.80
TEST_FRAC = 1.0 - TRAIN_FRAC
CLIP = 50

NORMALIZE_DATA = True
NOISE_TYPE = "gaussian"
NOISE_INTENSITY = 0.0
NOISE_SEED = 100

LIFTING_ORDER = 10
MAX_ITER = 50000
DEVICE = "cuda:0"

FULL_COST_EVAL_EVERY = 50
OPT_WEIGHTS = [1.0, 1.0, 0.0]
ROUTINE = "multi-perturb"  # OR "multi-perturb"
TRAIN_METHOD = "Zero-Mean"

SEED_Z = 1234
SEED_HP = 1234


def to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().mean())
    return float(x)


def parse_traj_batch_size(value):
    if value is None:
        return None

    text = str(value).strip().lower()
    if text in {"", "none", "null", "full"}:
        return None

    return int(text)


def batch_label(batch_size, n_train):
    if batch_size is None or batch_size >= n_train:
        return "full"
    return str(int(batch_size))


def actual_batch_size(batch_size, n_train):
    if batch_size is None or batch_size >= n_train:
        return n_train
    return int(batch_size)


def safe_token(value):
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return (
        text.replace("-", "m")
        .replace("+", "")
        .replace(".", "p")
        .replace(" ", "_")
    )


def result_filename(args):
    b = "full" if args.traj_batch_size is None else safe_token(args.traj_batch_size)
    return (
        f"grid_lr-{safe_token(args.learn_rate)}"
        f"_mom-{safe_token(args.momentum)}"
        f"_tol-{safe_token(args.stop_tol)}"
        f"_batch-{b}.json"
    )


def default_outdir(system_name, datestamp):
    return Path("Figures") / "GridSearch" / f"{system_name}_{datestamp}"


def atomic_write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(clean_json_value(payload), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")

    os.replace(tmp_path, path)


def clean_json_value(value):
    if isinstance(value, dict):
        return {str(k): clean_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_dataset(args):
    sim_data_raw, ts, num_traj, n_steps, n_train, n_test = gpk.load_SimData(
        args.system_name,
        args.train_frac,
        1.0 - args.train_frac,
        clip=CLIP,
    )

    if NORMALIZE_DATA:
        sim_data_clean, mu_vec, std_vec = gpk.normalize_data(
            sim_data_raw.to(dtype=torch.float32),
            n_test,
            n_train,
            n_steps,
        )
    else:
        sim_data_clean = sim_data_raw

    hp2_scale = gpk.find_hp_init(sim_data_clean[n_test:n_test + n_train, :, :-1])
    hp_scale = [None, hp2_scale, None]

    sim_data = gpk.add_noise(
        sim_data_clean,
        noise_type=NOISE_TYPE,
        intensity=NOISE_INTENSITY,
        seed=NOISE_SEED,
    )

    dataset = {}
    nx = sim_data.shape[1]
    n_steps = sim_data.shape[2] - 1
    ns_gpo = 3 * n_train

    dataset["SimData"] = sim_data
    dataset["X"] = torch.cat(
        [sim_data[n_test + j, :, 0:n_steps] for j in range(n_train)],
        dim=1,
    )
    dataset["Xplus"] = torch.cat(
        [sim_data[n_test + j, :, 1:] for j in range(n_train)],
        dim=1,
    )
    dataset["ICsetTrain"] = torch.cat(
        [sim_data[n_test + j, :, 0].view(nx, 1) for j in range(n_train)],
        dim=1,
    )
    dataset["ICsetTest"] = torch.cat(
        [sim_data[j, :, 0].view(nx, 1) for j in range(n_test)],
        dim=1,
    )
    dataset["Xtrain"] = gpk.get_kmeans(dataset["X"], num_centers=ns_gpo)
    dataset["dims"] = (nx, n_steps, ns_gpo)

    metadata = {
        "system_name": args.system_name,
        "train_frac": args.train_frac,
        "test_frac": 1.0 - args.train_frac,
        "num_traj": int(num_traj),
        "n_steps": int(n_steps),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "ts": to_float(ts),
        "hp2_scale": to_float(hp2_scale),
    }

    return dataset, sim_data_clean, hp_scale, metadata


def run_single_combo(args, out_path):
    dataset, sim_data_clean, hp_scale, metadata = build_dataset(args)

    n_train = metadata["n_train"]
    n_test = metadata["n_test"]
    n_steps = metadata["n_steps"]
    b_actual = actual_batch_size(args.traj_batch_size, n_train)
    b_label = batch_label(args.traj_batch_size, n_train)

    print("--------------------------------------------")
    print(
        f"Grid run | lr={args.learn_rate:.2e}, "
        f"momentum={args.momentum:.2f}, "
        f"traj_batch_size={b_label}, stop_tol={args.stop_tol:.1e}"
    )
    print(f"Output JSON: {out_path}")

    results = get_iGPK(
        Data=dataset,
        nTrain=n_train,
        nTest=n_test,
        lifting_order=args.lifting_order,
        max_iter=args.max_iter,
        sgd_lr=args.learn_rate,
        sgd_m=args.momentum,
        stop_tol=args.stop_tol,
        opt_weights=OPT_WEIGHTS,
        routine=ROUTINE,
        train_method=TRAIN_METHOD,
        hp_scale=hp_scale,
        device=args.device,
        seed_z=args.seed_z,
        seed_hp=args.seed_hp,
        traj_batch_size=args.traj_batch_size,
        full_cost_eval_every=args.full_cost_eval_every,
    )

    if "full_cost" in results["history"] and len(results["history"]["full_cost"]) > 0:
        final_full_cost = results["history"]["full_cost"][-1]
    else:
        final_full_cost = results["history"]["cost"][-1]

    train_nlpd = gpk.nlpd_per_traj(
        results["Train"]["Xhat"][:, :, :n_steps - 1],
        results["Train"]["Xcv"][:, :, :, :n_steps - 1],
        sim_data_clean[n_test:n_test + n_train, :, :n_steps - 1],
    )

    test_nlpd = gpk.nlpd_per_traj(
        results["Test"]["Xhat"][:, :, :n_steps - 1],
        results["Test"]["Xcv"][:, :, :, :n_steps - 1],
        sim_data_clean[:n_test, :, :n_steps - 1],
    )

    row = {
        "status": "ok",
        "result_file": str(out_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **metadata,
        "lifting_order": int(args.lifting_order),
        "max_iter": int(args.max_iter),
        "device": str(args.device),
        "seed_z": int(args.seed_z),
        "seed_hp": int(args.seed_hp),
        "noise_type": NOISE_TYPE,
        "noise_intensity": float(NOISE_INTENSITY),
        "noise_seed": int(NOISE_SEED),
        "learn_rate": float(args.learn_rate),
        "momentum": float(args.momentum),
        "stop_tol": float(args.stop_tol),
        "traj_batch_size": int(b_actual),
        "traj_batch_label": b_label,
        "full_cost_eval_every": int(args.full_cost_eval_every),
        "final_cost": to_float(final_full_cost),
        "final_full_cost": to_float(final_full_cost),
        "best_full_cost": to_float(
            results["history"].get("best_full_cost", final_full_cost)
        ),
        "post_mle_cost": to_float(results["history"]["post_mle_cost"]),
        "train_nrmse": 100.0 * to_float(results["Train"]["NRMSE"].mean()),
        "test_nrmse": 100.0 * to_float(results["Test"]["NRMSE"].mean()),
        "train_nlpd": to_float(train_nlpd.mean()),
        "test_nlpd": to_float(test_nlpd.mean()),
        "iters": int(results["history"]["iters"]),
        "total_iterations": int(results["history"]["iters"]),
        "best_iter": int(results["history"].get("best_iter", results["history"]["iters"])),
        "opt_time": to_float(results["history"]["opt_time"]),
        "total_compute_time": to_float(results["history"]["opt_time"]),
    }

    print(
        "Finished | "
        f"iters={row['iters']} | "
        f"time={row['opt_time']:.2f}s | "
        f"final={row['final_full_cost']:.3e} | "
        f"post={row['post_mle_cost']:.3e} | "
        f"test NRMSE={row['test_nrmse']:.2f}% | "
        f"test NLPD={row['test_nlpd']:.3f}"
    )

    del results
    del train_nlpd
    del test_nlpd
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one iGPK optimization-parameter grid-search point."
    )
    parser.add_argument("--learn-rate", type=float, required=True)
    parser.add_argument("--momentum", type=float, required=True)
    parser.add_argument("--stop-tol", type=float, required=True)
    parser.add_argument("--traj-batch-size", type=parse_traj_batch_size, required=True)

    parser.add_argument("--system-name", default=SYSTEM_NAME)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--lifting-order", type=int, default=LIFTING_ORDER)
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed-z", type=int, default=SEED_Z)
    parser.add_argument("--seed-hp", type=int, default=SEED_HP)
    parser.add_argument("--full-cost-eval-every", type=int, default=FULL_COST_EVAL_EVERY)
    parser.add_argument("--datestamp", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--outdir", default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir) if args.outdir else default_outdir(
        args.system_name,
        args.datestamp,
    )
    out_path = outdir / result_filename(args)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        payload = run_single_combo(args, out_path)
    except Exception as exc:
        payload = {
            "status": "failed",
            "result_file": str(out_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "system_name": args.system_name,
            "learn_rate": float(args.learn_rate),
            "momentum": float(args.momentum),
            "stop_tol": float(args.stop_tol),
            "traj_batch_size": (
                None if args.traj_batch_size is None else int(args.traj_batch_size)
            ),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(payload["traceback"])

    atomic_write_json(payload, out_path)
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
