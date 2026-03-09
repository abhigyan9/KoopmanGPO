import os
import torch
import numpy as np
import GPKoopman as gpk
from matplotlib import pyplot as plt  # (kept if you use it later)

# Manual seed for reproducibility
torch.manual_seed(1234)
np.random.seed(1234)

# -----------------------------
# Initial-condition generators
# -----------------------------


def generate_initial_conditions(system, num_trajectories):
    if system == "Unforced Duffing":
        x0 = torch.tensor(np.random.uniform(1.5, 2.5, size=(
            1, num_trajectories)), dtype=torch.float64)
        x1 = torch.tensor(np.random.uniform(-1.5, 1.5,
                          size=(1, num_trajectories)), dtype=torch.float64)
        return torch.vstack([x0, x1])

    elif system == "van der Pol":
        return torch.tensor(np.random.uniform(-4., 4., size=(2, num_trajectories)), dtype=torch.float64)

    elif system == "Reverse van der Pol":
        x0 = torch.tensor(
            np.random.uniform(-1., 1., size=(1, num_trajectories)), dtype=torch.float64)
        x1 = torch.tensor(
            np.random.uniform(-1., 1., size=(1, num_trajectories)), dtype=torch.float64)
        return torch.vstack([x0, x1])

    elif system == "Simple Pendulum":
        x0 = torch.tensor(
            np.random.uniform(-2., 2., size=(1, num_trajectories)), dtype=torch.float64)
        x1 = torch.tensor(
            np.random.uniform(-3., 3., size=(1, num_trajectories)), dtype=torch.float64)
        return torch.vstack([x0, x1])

    elif system == "Chaotic-Lorenz":
        return torch.tensor(np.random.uniform(-20, 20, size=(3, num_trajectories)), dtype=torch.float64)

    elif system == "Lotka Volterra":
        x0 = torch.tensor(np.random.uniform(0., 2., size=(
            1, num_trajectories)), dtype=torch.float64)
        x1 = torch.tensor(np.random.uniform(0., 1., size=(
            1, num_trajectories)), dtype=torch.float64)
        return torch.vstack([x0, x1])

    elif system == "Piecewise Linear" or system == "PWL Discrete":
        return torch.tensor(np.random.uniform(0., 1., size=(1, num_trajectories)), dtype=torch.float64)

    elif system == "Scalar NL":
        return torch.tensor(np.random.uniform(-5., 5., size=(1, num_trajectories)), dtype=torch.float64)

    # --- New systems below ---
    elif system == "Reciprocal Relaxer":
        # 1D; avoid extreme magnitude; eps in dynamics already regularizes near 0
        return torch.tensor(np.random.uniform(-2., 2., size=(1, num_trajectories)), dtype=torch.float64)

    elif system == "Reciprocal-Biased Damped Pendulum":
        # Angle & angular velocity
        x0 = torch.tensor(np.random.uniform(-1.5, 1.5,
                          size=(1, num_trajectories)), dtype=torch.float64)
        x1 = torch.tensor(np.random.uniform(-1.5, 1.5,
                          size=(1, num_trajectories)), dtype=torch.float64)
        return torch.vstack([x0, x1])

    elif system == "Inhibited Predator-Prey" or system == "IPP-Large":
        # Prey/predator strictly positive
        prey0 = torch.tensor(np.random.uniform(
            0.1, 4.0, size=(1, num_trajectories)), dtype=torch.float64)
        pred0 = torch.tensor(np.random.uniform(
            0.1, 3.0, size=(1, num_trajectories)), dtype=torch.float64)
        return torch.vstack([prey0, pred0])

    else:
        raise ValueError('Invalid System Name')

# (kept for future use)


def generate_random_inputs(num_steps, input_dim):
    return torch.tensor(np.random.uniform(-1, 1, size=(input_dim, num_steps)), dtype=torch.float64)

# -----------------------------
# Main generator
# -----------------------------


def generate_and_save_data():
    """
    systems: dict of
      system_name: (fx, state_dim, sample_time, params, num_steps, num_trajectories, is_discrete)
    """
    systems = {
        # --- Examples you can (re)enable as you wish ---
        # "Unforced Duffing":     (gpk.f_UDO,            2, 0.01, None,                               1500, 60,  False),
        # "van der Pol":          (gpk.f_VDP,            2, 0.01, None,                               1500, 60,  False),
        # "Reverse van der Pol":  (gpk.f_RVDP,           2, 0.10, None,                                400, 40,  False),
        # "Simple Pendulum":      (gpk.f_SDP,            2, 0.02, None,                                800, 50,  False),
        # "Chaotic-Lorenz":               (gpk.f_Lorenz,         3, 0.01, torch.tensor([10., 8./3., 28.], dtype=torch.float64), 200, 100, False),
        # "Lotka Volterra":       (gpk.f_LotkaVolterra,  2, 0.20, torch.tensor([0.2, 0.8, 0.25, 0.4], dtype=torch.float64), 300, 60, False),
        # "Piecewise Linear":     (gpk.f_PWL1,           1, 2.00, None,                                120, 50,  False),
        # "PWL Discrete":         (gpk.df_PWL,           1, 1.00, None,                                120, 50,  True),

        # --- Your existing discrete scalar NL example ---
        # "Scalar NL": (gpk.df_scalarNL, 1, 1.00, None, 200, 50, True),

        # --- New systems ---
        # "Reciprocal Relaxer": (
        #     gpk.f_RR,  # \dot x = -alpha x + beta/(eps + x^2)
        #     1,
        #     0.02,  # sample time
        #     torch.tensor([1.0, 1.0, 1e-3], dtype=torch.float64),  # [alpha, beta, eps]
        #     200,   # num_steps
        #     50,    # num_trajectories
        #     False  # continuous
        # ),

        # "Reciprocal-Biased Damped Pendulum": (
        #     gpk.f_RBDP,  # x1' = x2; x2' = -kappa x2 - omega^2 sin x1 + a/(1 + x1^2)
        #     2,
        #     0.05,
        #     torch.tensor([0.1, 1.0, 0.5], dtype=torch.float64),  # [kappa, omega, a]
        #     100,
        #     200,
        #     False
        # ),

        # "Inhibited Predator-Prey": (
        #     gpk.f_IPP,   # Holling-type inhibition: 1/(1 + h x^n)
        #     2,
        #     0.2,
        #     torch.tensor([1.0, 5.0, 1.0, 1.0, 2.0, 0.5, 0.3], dtype=torch.float64),  # [r, K, a, h, n, eta, d]
        #     100,
        #     200,
        #     False
        # ),
        "IPP-Large": (
            gpk.f_IPP,
            2,
            0.2,
            torch.tensor([1.0, 5.0, 1.0, 1.0, 2.0, 0.5, 0.3], dtype=torch.float64),
            200,
            1000,
            False
        )
    }

    os.makedirs("Data", exist_ok=True)

    for system_name, (fx, state_dim, ts, params, num_steps, num_trajectories, is_discrete) in systems.items():
        trajectories = []
        initial_conditions = []

        x0_mat = generate_initial_conditions(system_name, num_trajectories)

        for j in range(num_trajectories):
            x0 = x0_mat[:, j]
            if is_discrete:
                states = gpk.sim_discrete(
                    fx, x0, ts, num_steps + 1, params=params)
            else:
                states = gpk.sim_RK4(fx, x0, ts, num_steps + 1, params=params)

            trajectories.append(states)
            initial_conditions.append(x0)  # store IC for this trajectory

        data = {
            # shape: [num_trajectories, state_dim, num_steps+1]
            "trajectories": torch.stack(trajectories),
            # shape: [num_trajectories, state_dim]
            "initial_conditions": torch.stack(initial_conditions),
            "sample_time": ts,
            "num_steps": num_steps,
            "num_trajectories": num_trajectories,
        }

        outfile = f"Data/DataAuto_{system_name}.pt"
        torch.save(data, outfile)
        print(f"Data for {system_name} saved to {outfile}")


if __name__ == "__main__":
    generate_and_save_data()
