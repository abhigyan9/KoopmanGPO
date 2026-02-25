# sweep_noise.py
import itertools
from scalarNL_script import run_models_for_noise

SYSTEM_NAME = "Sloshing"   # change as needed
TRAIN_FRAC = 0.80
TEST_FRAC = 1 - TRAIN_FRAC
CLIP = None                 # or None

NOISE_TYPES = ["gaussian"]

INTENSITIES = [0]  # normalized scale
SEEDS = [100]                      # repeatability / variability

OUTDIR = "Figures/Trial_" + SYSTEM_NAME

for noise_type, intensity, seed in itertools.product(NOISE_TYPES, INTENSITIES, SEEDS):
    if intensity == 0.0 and noise_type != "gaussian":
        continue   # skip duplicates at zero noise

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
        lifted_order=15,
        iters_list=(0, 32, 16, 600),
        learn_rate=0.04,
        opt_weights=(1.0, 1.0, 1.0),
        routine="Z_only",
        train_method="Horizon",
        device="cuda:0",
        outdir=OUTDIR
    )
