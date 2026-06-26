# save_halfcheetah_expert_v2_tensor.py

import argparse
from pathlib import Path

import torch
from datasets import load_dataset


DATASET_ID = "edbeeching/decision_transformer_gym_replay"
DATASET_CONFIG = "halfcheetah-expert-v2"
SPLIT = "train"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default="Data\halfcheetah_expert_v2_first4k.pt",
        help="Output .pt file path",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=16,
        help="Number of time steps per trajectory segment",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATASET_ID} / {DATASET_CONFIG} / split={SPLIT}")

    ds = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=SPLIT,
        cache_dir=args.cache_dir,
    )

    num_steps = 1 + args.num_steps
    segments = []

    for i, row in enumerate(ds):
        obs = torch.as_tensor(row["observations"], dtype=torch.float32)
        act = torch.as_tensor(row["actions"], dtype=torch.float32)

        if obs.ndim != 2:
            raise ValueError(f"Expected observations to have shape (T, obs_dim), got {obs.shape}")

        if act.ndim != 2:
            raise ValueError(f"Expected actions to have shape (T, act_dim), got {act.shape}")

        if obs.shape[0] != act.shape[0]:
            raise ValueError(
                f"Observation/action time mismatch in trajectory {i}: "
                f"{obs.shape[0]} vs {act.shape[0]}"
            )

        # Concatenate at each time step:
        # obs: (T, obs_dim)
        # act: (T, act_dim)
        # xu:  (T, obs_dim + act_dim)
        xu = torch.cat([obs, act], dim=1)

        T, num_states = xu.shape

        # Drop incomplete remainder at the end of each original trajectory.
        n_segments = T // num_steps
        usable_T = n_segments * num_steps

        if usable_T == 0:
            continue

        xu = xu[:usable_T, :]

        # Split into 16-step segments:
        # (usable_T, num_states)
        # -> (n_segments, num_steps, num_states)
        # -> (n_segments, num_states, num_steps)
        xu_segments = xu.reshape(n_segments, num_steps, num_states)
        xu_segments = xu_segments.permute(0, 2, 1).contiguous()

        segments.append(xu_segments)

        if (i + 1) % 100 == 0 or (i + 1) == len(ds):
            print(f"Processed {i + 1}/{len(ds)} original trajectories")

    if len(segments) == 0:
        raise RuntimeError("No valid trajectory segments were created.")

    # Final tensor shape:
    # (num_trajectories, num_states, num_steps)
    data_tensor = torch.cat(segments, dim=0)[:4000]

    data = {
        # shape: [num_trajectories, state_dim, num_steps+1]
        "trajectories": data_tensor,
        "initial_conditions": None,
        "sample_time": 1.0,
        "num_steps": num_steps-1,
        "num_trajectories": data_tensor.shape[0],
    }

    torch.save(data, out_path)

    print(f"\nSaved tensor dataset to: {out_path.resolve()}")
    print(f"Final tensor shape: {tuple(data_tensor.shape)}")


if __name__ == "__main__":
    main()
