# sweep_noise.py
import itertools
from acc26_script import run_models_for_noise

SYSTEM_NAME = "Reverse van der Pol"   # change as needed
TRAIN_FRAC = 0.40
TEST_FRAC = 0.20
CLIP = None                 # or None

NOISE_TYPES = [
    "gaussian"
]

INTENSITIES = [0., 0.02, 0.05]  # normalized scale
SEEDS = [100]                      # repeatability / variability

OUTDIR = "Figures_Noise-Sweep_" + SYSTEM_NAME

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
        lifted_order=20,
        iters_list=(0, 128, 64, 150),
        learn_rate=0.04,
        opt_weights=(1.0, 1.0, 10.0),
        routine="BO_ZnHP",
        train_method="Horizon",
        device="cuda:0",
        outdir=OUTDIR
    )
