from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import GPKoopman as gpk


DTYPE = torch.float64


@dataclass(frozen=True)
class SystemSpec:
    cli_name: str
    display_name: str
    func_name: str
    state_dim: int
    sample_time: float
    num_steps: int
    num_trajectories: int
    is_discrete: bool
    ic_low: tuple[float, ...]
    ic_high: tuple[float, ...]
    params: Optional[tuple[float, ...]] = None


SYSTEM_SPECS: dict[str, SystemSpec] = {
    "unforced_duffing": SystemSpec(
        cli_name="unforced_duffing",
        display_name="Unforced Duffing",
        func_name="f_UDO",
        state_dim=2,
        sample_time=0.01,
        num_steps=150,
        num_trajectories=60,
        is_discrete=False,
        ic_low=(1.5, -1.5),
        ic_high=(2.5, 1.5),
        params=None,
    ),
    "van_der_pol": SystemSpec(
        cli_name="van_der_pol",
        display_name="van der Pol",
        func_name="f_VDP",
        state_dim=2,
        sample_time=0.01,
        num_steps=150,
        num_trajectories=60,
        is_discrete=False,
        ic_low=(-4.0, -4.0),
        ic_high=(4.0, 4.0),
        params=None,
    ),
    "reverse_van_der_pol": SystemSpec(
        cli_name="reverse_van_der_pol",
        display_name="Reverse van der Pol",
        func_name="f_RVDP",
        state_dim=2,
        sample_time=0.10,
        num_steps=400,
        num_trajectories=40,
        is_discrete=False,
        ic_low=(-1.0, -1.0),
        ic_high=(1.0, 1.0),
        params=None,
    ),
    "simple_pendulum": SystemSpec(
        cli_name="simple_pendulum",
        display_name="Simple Pendulum",
        func_name="f_SDP",
        state_dim=2,
        sample_time=0.02,
        num_steps=800,
        num_trajectories=50,
        is_discrete=False,
        ic_low=(-2.0, -3.0),
        ic_high=(2.0, 3.0),
        params=None,
    ),
    "chaotic_lorenz": SystemSpec(
        cli_name="chaotic_lorenz",
        display_name="Chaotic-Lorenz",
        func_name="f_Lorenz",
        state_dim=3,
        sample_time=0.01,
        num_steps=150,
        num_trajectories=100,
        is_discrete=False,
        ic_low=(-20.0, -20.0, -20.0),
        ic_high=(20.0, 20.0, 20.0),
        params=(10.0, 8.0 / 3.0, 28.0),
    ),
    "lotka_volterra": SystemSpec(
        cli_name="lotka_volterra",
        display_name="Lotka Volterra",
        func_name="f_LotkaVolterra",
        state_dim=2,
        sample_time=0.20,
        num_steps=300,
        num_trajectories=60,
        is_discrete=False,
        ic_low=(0.0, 0.0),
        ic_high=(2.0, 1.0),
        params=(0.2, 0.8, 0.25, 0.4),
    ),
    "piecewise_linear": SystemSpec(
        cli_name="piecewise_linear",
        display_name="Piecewise Linear",
        func_name="f_PWL1",
        state_dim=1,
        sample_time=2.00,
        num_steps=120,
        num_trajectories=50,
        is_discrete=False,
        ic_low=(0.0,),
        ic_high=(1.0,),
        params=None,
    ),
    "pwl_discrete": SystemSpec(
        cli_name="pwl_discrete",
        display_name="PWL Discrete",
        func_name="df_PWL",
        state_dim=1,
        sample_time=1.00,
        num_steps=120,
        num_trajectories=50,
        is_discrete=True,
        ic_low=(0.0,),
        ic_high=(1.0,),
        params=None,
    ),
    "scalar_nl": SystemSpec(
        cli_name="scalar_nl",
        display_name="Scalar NL",
        func_name="df_scalarNL",
        state_dim=1,
        sample_time=1.00,
        num_steps=50,
        num_trajectories=100,
        is_discrete=True,
        ic_low=(-6.0,),
        ic_high=(6.0,),
        params=None,
    ),
    "reciprocal_relaxer": SystemSpec(
        cli_name="reciprocal_relaxer",
        display_name="Reciprocal Relaxer",
        func_name="f_RR",
        state_dim=1,
        sample_time=0.02,
        num_steps=200,
        num_trajectories=50,
        is_discrete=False,
        ic_low=(-2.0,),
        ic_high=(2.0,),
        params=(1.0, 1.0, 1e-3),
    ),
    "reciprocal_biased_damped_pendulum": SystemSpec(
        cli_name="reciprocal_biased_damped_pendulum",
        display_name="Reciprocal-Biased Damped Pendulum",
        func_name="f_RBDP",
        state_dim=2,
        sample_time=0.05,
        num_steps=100,
        num_trajectories=200,
        is_discrete=False,
        ic_low=(-1.5, -1.5),
        ic_high=(1.5, 1.5),
        params=(0.1, 1.0, 0.5),
    ),
    "inhibited_predator_prey": SystemSpec(
        cli_name="inhibited_predator_prey",
        display_name="Inhibited Predator-Prey",
        func_name="f_IPP",
        state_dim=2,
        sample_time=0.2,
        num_steps=100,
        num_trajectories=200,
        is_discrete=False,
        ic_low=(0.1, 0.1),
        ic_high=(4.0, 3.0),
        params=(1.0, 5.0, 1.0, 1.0, 2.0, 0.5, 0.3),
    ),
    "ipp_large": SystemSpec(
        cli_name="ipp_large",
        display_name="IPP-Large",
        func_name="f_IPP",
        state_dim=2,
        sample_time=0.2,
        num_steps=200,
        num_trajectories=1000,
        is_discrete=False,
        ic_low=(0.1, 0.1),
        ic_high=(4.0, 3.0),
        params=(1.0, 5.0, 1.0, 1.0, 2.0, 0.5, 0.3),
    ),
    "unforced_poc": SystemSpec(
        cli_name="unforced_poc",
        display_name="Unforced POC",
        func_name="f_uPoC",
        state_dim=4,
        sample_time=0.02,
        num_steps=100,
        num_trajectories=100,
        is_discrete=False,
        ic_low=(2.0, -0.4, 0.0, -0.3),
        ic_high=(4.0, 0.4, 1.0, 0.3),
        params=(0.4, 1., 9.81, 0.5, 6., 0.1/12),
    )
}


ALIASES = {
    "Unforced Duffing": "unforced_duffing",
    "van der Pol": "van_der_pol",
    "Reverse van der Pol": "reverse_van_der_pol",
    "Simple Pendulum": "simple_pendulum",
    "Chaotic-Lorenz": "chaotic_lorenz",
    "Lotka Volterra": "lotka_volterra",
    "Piecewise Linear": "piecewise_linear",
    "PWL Discrete": "pwl_discrete",
    "Scalar NL": "scalar_nl",
    "Reciprocal Relaxer": "reciprocal_relaxer",
    "Reciprocal-Biased Damped Pendulum": "reciprocal_biased_damped_pendulum",
    "Inhibited Predator-Prey": "inhibited_predator_prey",
    "IPP-Large": "ipp_large",
    "Unforced POC": "unforced_poc",
}


def resolve_system_name(name: str) -> str:
    key = name.strip()
    key = ALIASES.get(key, key)
    key = key.lower().replace(" ", "_").replace("-", "_")
    if key not in SYSTEM_SPECS:
        valid = ", ".join(sorted(SYSTEM_SPECS))
        raise ValueError(f"Unknown system '{name}'. Valid systems: {valid}")
    return key


def _to_tensor(values: Optional[Sequence[float]]) -> Optional[torch.Tensor]:
    if values is None:
        return None
    return torch.tensor(values, dtype=DTYPE)


def _parse_number_list(text: Optional[str]) -> Optional[list[float]]:
    if text is None:
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return None
    return [float(p) for p in parts]


def _expand_vector(values: Sequence[float], state_dim: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 1:
        arr = np.repeat(arr, state_dim)
    if arr.size != state_dim:
        raise ValueError(
            f"{name} must have length 1 or {state_dim}, but got length {arr.size}."
        )
    return arr


def sample_initial_conditions(
    state_dim: int,
    num_trajectories: int,
    ic_low: Sequence[float],
    ic_high: Sequence[float],
    rng: np.random.RandomState,
) -> torch.Tensor:
    """
    Sample initial conditions using the legacy TrajDataGen_A pattern:
    draw one state dimension at a time and then stack the results.

    This preserves the same random-number consumption order as code of the form
        x1 = np.random.uniform(low_1, high_1, size=(1, num_trajectories))
        x2 = np.random.uniform(low_2, high_2, size=(1, num_trajectories))
        ...
        x0 = np.vstack([x1, x2, ...])
    which is needed for exact backward-compatible datasets.
    """
    low = _expand_vector(ic_low, state_dim, "ic_low")
    high = _expand_vector(ic_high, state_dim, "ic_high")
    if np.any(high <= low):
        raise ValueError(
            "Every entry of ic_high must be strictly greater than ic_low.")

    draws = []
    for i in range(state_dim):
        draw_i = rng.uniform(
            low=low[i], high=high[i], size=(1, num_trajectories))
        draws.append(draw_i)

    x0 = np.vstack(draws)
    return torch.tensor(x0, dtype=DTYPE)


def get_resolved_config(
    system: str,
    *,
    num_trajectories: Optional[int] = None,
    num_steps: Optional[int] = None,
    sample_time: Optional[float] = None,
    params: Optional[Sequence[float]] = None,
    ic_low: Optional[Sequence[float]] = None,
    ic_high: Optional[Sequence[float]] = None,
    seed: int = 1234,
    output_dir: str | Path = "Data",
    filename: Optional[str] = None,
    save: bool = False,
):
    system_key = resolve_system_name(system)
    spec = SYSTEM_SPECS[system_key]

    resolved = {
        "system_key": system_key,
        "system_name": spec.display_name,
        "func_name": spec.func_name,
        "state_dim": spec.state_dim,
        "sample_time": spec.sample_time if sample_time is None else float(sample_time),
        "num_steps": spec.num_steps if num_steps is None else int(num_steps),
        "num_trajectories": spec.num_trajectories if num_trajectories is None else int(num_trajectories),
        "is_discrete": spec.is_discrete,
        "params": spec.params if params is None else tuple(params),
        "ic_low": spec.ic_low if ic_low is None else tuple(ic_low),
        "ic_high": spec.ic_high if ic_high is None else tuple(ic_high),
        "seed": int(seed),
        "save": bool(save),
        "output_dir": Path(output_dir),
        "filename": filename or f"DataAuto_{spec.display_name}.pt",
    }

    if resolved["num_steps"] <= 0:
        raise ValueError("num_steps must be positive.")
    if resolved["num_trajectories"] <= 0:
        raise ValueError("num_trajectories must be positive.")
    if resolved["sample_time"] <= 0:
        raise ValueError("sample_time must be positive.")

    return resolved


def generate_dataset(
    system: str,
    *,
    num_trajectories: Optional[int] = None,
    num_steps: Optional[int] = None,
    sample_time: Optional[float] = None,
    params: Optional[Sequence[float]] = None,
    ic_low: Optional[Sequence[float]] = None,
    ic_high: Optional[Sequence[float]] = None,
    seed: int = 1234,
    save: bool = False,
    output_dir: str | Path = "Data",
    filename: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    cfg = get_resolved_config(
        system,
        num_trajectories=num_trajectories,
        num_steps=num_steps,
        sample_time=sample_time,
        params=params,
        ic_low=ic_low,
        ic_high=ic_high,
        seed=seed,
        output_dir=output_dir,
        filename=filename,
        save=save,
    )

    fx = getattr(gpk, cfg["func_name"])
    param_tensor = _to_tensor(cfg["params"])
    # Use the legacy NumPy RandomState so identical seeds reproduce the
    # same initial conditions as the older TrajDataGen_A script.
    # rng = np.random.RandomState(cfg["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed=cfg["seed"])

    x0_mat = sample_initial_conditions(
        state_dim=cfg["state_dim"],
        num_trajectories=cfg["num_trajectories"],
        ic_low=cfg["ic_low"],
        ic_high=cfg["ic_high"],
        rng=rng,
    )

    simulator = gpk.sim_discrete if cfg["is_discrete"] else gpk.sim_RK4

    trajectories = []
    for j in range(cfg["num_trajectories"]):
        x0 = x0_mat[:, j]
        states = simulator(
            fx,
            x0,
            cfg["sample_time"],
            cfg["num_steps"] + 1,
            params=param_tensor,
        )
        trajectories.append(states)

    data = {
        "system_name": cfg["system_name"],
        "system_key": cfg["system_key"],
        "trajectories": torch.stack(trajectories),
        "initial_conditions": x0_mat.T.contiguous(),
        "sample_time": cfg["sample_time"],
        "num_steps": cfg["num_steps"],
        "num_trajectories": cfg["num_trajectories"],
        "state_dim": cfg["state_dim"],
        "is_discrete": cfg["is_discrete"],
        "params": None if param_tensor is None else param_tensor.clone(),
        "ic_low": torch.tensor(cfg["ic_low"], dtype=DTYPE),
        "ic_high": torch.tensor(cfg["ic_high"], dtype=DTYPE),
        "seed": cfg["seed"],
    }

    if save:
        cfg["output_dir"].mkdir(parents=True, exist_ok=True)
        output_path = cfg["output_dir"] / cfg["filename"]
        torch.save(data, output_path)
        data["outfile"] = str(output_path)
        if verbose:
            print(f"Saved dataset to: {output_path}")

    return data


def format_config(cfg: dict) -> str:
    lines = [
        "Resolved dataset options:",
        f"  system           : {cfg['system_name']} ({cfg['system_key']})",
        f"  dynamics         : {cfg['func_name']}",
        f"  state_dim        : {cfg['state_dim']}",
        f"  sample_time      : {cfg['sample_time']}",
        f"  num_steps        : {cfg['num_steps']}",
        f"  num_trajectories : {cfg['num_trajectories']}",
        f"  is_discrete      : {cfg['is_discrete']}",
        f"  params           : {cfg['params']}",
        f"  ic_low           : {cfg['ic_low']}",
        f"  ic_high          : {cfg['ic_high']}",
        f"  seed             : {cfg['seed']}",
        f"  save             : {cfg['save']}",
    ]
    if cfg["save"]:
        lines.append(
            f"  output_file      : {cfg['output_dir'] / cfg['filename']}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate autonomous-system trajectory datasets with per-system defaults."
    )
    parser.add_argument(
        "system",
        nargs="?",
        help="System name or alias. Example: scalar_nl, unforced_duffing, chaotic_lorenz",
    )
    parser.add_argument("--list-systems", action="store_true",
                        help="List all available systems and exit.")
    parser.add_argument("--num-trajectories", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--sample-time", type=float, default=None)
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Comma-separated parameter override, e.g. --params 1.0,5.0,1.0,1.0,2.0,0.5,0.3",
    )
    parser.add_argument(
        "--ic-low",
        type=str,
        default=None,
        help="Comma-separated lower bounds for IC sampling. Length 1 or state_dim.",
    )
    parser.add_argument(
        "--ic-high",
        type=str,
        default=None,
        help="Comma-separated upper bounds for IC sampling. Length 1 or state_dim.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=str, default="Data")
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--no-save", action="store_true",
                        help="Generate the dataset without saving a file.")
    return parser


def list_systems() -> str:
    lines = ["Available systems:"]
    for key, spec in sorted(SYSTEM_SPECS.items()):
        lines.append(f"  {key:38s} -> {spec.display_name}")
    return "\n".join(lines)


def main() -> dict:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_systems:
        print(list_systems())
        return {}

    if not args.system:
        parser.error("Please provide a system name, or use --list-systems.")

    params = _parse_number_list(args.params)
    ic_low = _parse_number_list(args.ic_low)
    ic_high = _parse_number_list(args.ic_high)

    cfg = get_resolved_config(
        args.system,
        num_trajectories=args.num_trajectories,
        num_steps=args.num_steps,
        sample_time=args.sample_time,
        params=params,
        ic_low=ic_low,
        ic_high=ic_high,
        seed=args.seed,
        output_dir=args.output_dir,
        filename=args.filename,
        save=not args.no_save,
    )
    print(format_config(cfg))

    return generate_dataset(
        args.system,
        num_trajectories=args.num_trajectories,
        num_steps=args.num_steps,
        sample_time=args.sample_time,
        params=params,
        ic_low=ic_low,
        ic_high=ic_high,
        seed=args.seed,
        save=not args.no_save,
        output_dir=args.output_dir,
        filename=args.filename,
        verbose=True,
    )


if __name__ == "__main__":
    main()
