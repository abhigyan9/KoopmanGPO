# sweep_noise.py
import itertools
from scalarNL_script import run_models_for_noise
import torch
import numpy as np
import argparse
import GPKoopman as gpk
from datetime import datetime
import os

# --------- #
## HELPERS ##
# --------- #


def find_hp_init(SimData: torch.tensor, nTrain: int) -> float:
    def _stack_snapshot_pairs(batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        batch: (nB, n, T+1)
        Returns:
            X: (nB*T, n), Y: (nB*T, n)
        """
        n = batch.shape[1]
        X = batch[:, :, :-1].permute(0, 2, 1).reshape(-1,
                                                      n).detach().cpu().numpy()
        Y = batch[:, :,  1:].permute(
            0, 2, 1).reshape(-1, n).detach().cpu().numpy()
        return X, Y
    # ---------- build stacked (X,Y) ----------
    train_batch = SimData[:nTrain]
    Xtr, _ = _stack_snapshot_pairs(train_batch)
    Npts = Xtr.shape[0]

    max_pairs_to_store = 5_000_000  # ~5 million floats ~ 40MB
    num_pairs = Npts * (Npts - 1) // 2

    if num_pairs <= max_pairs_to_store:
        # Store all distances (exact median).
        dists = np.empty(num_pairs, dtype=np.float32)
        k = 0
        for i in range(Npts - 1):
            diff = Xtr[i + 1:] - Xtr[i]                 # (Npts-i-1, n)
            di = np.sqrt(np.sum(diff * diff, axis=1))    # (Npts-i-1,)
            dists[k: k + di.size] = di
            k += di.size
        hp_init = float(np.median(dists))
        return hp_init
    else:  # fallback for huge datasets
        rng = np.random.default_rng(0)
        # sample up to the cap
        sample_pairs = min(max_pairs_to_store, num_pairs)
        idx_i = rng.integers(0, Npts, size=sample_pairs, endpoint=False)
        idx_j = rng.integers(0, Npts, size=sample_pairs, endpoint=False)

        # Ensure i != j (resample conflicts)
        mask = idx_i == idx_j
        while np.any(mask):
            idx_j[mask] = rng.integers(
                0, Npts, size=int(mask.sum()), endpoint=False)
            mask = idx_i == idx_j

        diffs = Xtr[idx_i] - Xtr[idx_j]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        hp_init = float(np.median(dists))
        return hp_init


# -------------------------------------------------
# Parse command-line arguments
# -------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run Koopman experiments with configurable noise.")

parser.add_argument("--system", type=str, default="Lorenz",
                    help="System name (e.g., Lorenz, Cart_data, etc.)")

parser.add_argument("--noise_types", nargs="+", default=["gaussian"],
                    help="Noise types (space separated)")

parser.add_argument("--intensities", nargs="+", type=float,
                    default=[0.0],
                    help="Noise intensities (space separated)")

parser.add_argument("--lifted_order", type=int,
                    default=5,
                    help="Lifted System Order")

parser.add_argument("--poly_deg", type=int,
                    default=3,
                    help="Order of Polynomials for poly-eDMD")

parser.add_argument("--max_iter", type=int,
                    default=2000,
                    help="Maximum Iterations for iGPK")

parser.add_argument("--learn_rate", type=float,
                    default=0.001,
                    help="Learning Rate for iGPK")

parser.add_argument("--train_frac", type=float,
                    default=0.60,
                    help="Fraction of Trajectories to use for Training [must be less than 1]")

args = parser.parse_args()

# -------------------------------------------------
# Configuration from CLI
# -------------------------------------------------
SYSTEM_NAME = args.system
NOISE_TYPES = args.noise_types
INTENSITIES = args.intensities
LIFTED_ORDER = int(args.lifted_order)
POLY_DEG = int(args.poly_deg)
MAX_ITER = int(args.max_iter)
LEARN_RATE = float(args.learn_rate)
TRAIN_FRAC = float(args.train_frac)

TEST_FRAC = 1 - TRAIN_FRAC
CLIP = None
NORMALIZE_DATA = True

SEEDS = [100]
stamp = datetime.now().strftime("%Y%m%d")
OUTDIR = "Figures/Journal/" + SYSTEM_NAME + f'_{LIFTED_ORDER}D-' + stamp
os.makedirs(OUTDIR, exist_ok=True)
# Find Scale of Hyperparameter Initialization
SimData_raw, _, _, N, nTrain, _ = gpk.load_SimData(
    SYSTEM_NAME, TRAIN_FRAC, TEST_FRAC, clip=CLIP)

if NORMALIZE_DATA:
    SimData, _, _ = gpk.normalize_data(
        SimData_raw, nTrain, N)
else:
    SimData_clean = SimData_raw

hp_scale = find_hp_init(SimData, nTrain)

# -------------------------------------------------
# Run experiments
# -------------------------------------------------
for noise_type, intensity, seed in itertools.product(NOISE_TYPES, INTENSITIES, SEEDS):

    if intensity == 0.0 and noise_type != "gaussian":
        continue  # skip duplicates at zero noise

    print(f"\n=== {noise_type} | intensity={intensity:.3f} | seed={seed} ===")

    run_models_for_noise(
        system_name=SYSTEM_NAME,
        train_frac=TRAIN_FRAC,
        test_frac=TEST_FRAC,
        clip=CLIP,
        noise_type=noise_type,
        intensity=intensity,
        seed=seed,
        outdir=OUTDIR,
        normalizeData=NORMALIZE_DATA,
        lifted_order=LIFTED_ORDER,
        poly_deg=POLY_DEG,
        max_iter=MAX_ITER,
        learn_rate=LEARN_RATE,
        kernel_hp_scale=[1.0, hp_scale, None],
    )
