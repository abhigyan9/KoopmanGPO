# sweep_noise.py
import itertools
from acc26_script import run_models_for_noise

SYSTEM_NAME = "Simple Pendulum"   # change as needed
TRAIN_FRAC = 0.40
TEST_FRAC = 0.20
CLIP = 100                 # or None

NOISE_TYPES = [
    "gaussian",
    "uniform",
    "linear_gaussian",
    "quadratic_gaussian",
    "linear_uniform",
]

INTENSITIES = [0.00, 0.02, 0.05, 0.10, 0.20]  # normalized scale
SEEDS = [100, 101]                      # repeatability / variability

OUTDIR = "Figures"

for noise_type, intensity, seed in itertools.product(NOISE_TYPES, INTENSITIES, SEEDS):
    print(f"\n=== {noise_type} | intensity={intensity:.3f} | seed={seed} ===")
    run_models_for_noise(
        system_name=SYSTEM_NAME,
        train_frac=TRAIN_FRAC,
        test_frac=TEST_FRAC,
        clip=CLIP,
        noise_type=noise_type,
        intensity=intensity,
        seed=seed,
        # (tweak model knobs if desired)
        lifted_order=10,
        iters_list=(2000, 50, 50, 100),
        learn_rate=0.04,
        opt_weights=(10.0, 1.0, 10.0),
        routine="Z_only",
        train_method="Horizon",
        device="cuda:0",
        outdir=OUTDIR
    )
