import torch
import numpy as np
import GPKoopman as gpk
from matplotlib import pyplot as plt

# Manual seed for reproducibility
torch.manual_seed(1234)
np.random.seed(1234)

# Simulation parameters
num_steps = 100  # Number of steps
num_trajectories = 100  # Number of random initial conditions

# Function to generate random initial conditions


def generate_initial_conditions(system, num_trajectories):
    if system == "Unforced Duffing":
        x0 = torch.tensor(
            np.random.uniform(1.5, 2.5, size=(1, num_trajectories)), dtype=torch.float64)
        x1 = torch.tensor(
            np.random.uniform(-1.5, 1.5, size=(1, num_trajectories)), dtype=torch.float64)
        return torch.vstack([x0, x1])
    elif system == "van der Pol":
        return torch.tensor(np.random.uniform(-4., 4., size=(2, num_trajectories)), dtype=torch.float64)
    elif system == "Simple Pendulum":
        x0 = torch.tensor(
            np.random.uniform(-2., 2., size=(1, num_trajectories)), dtype=torch.float64)
        x1 = torch.tensor(
            np.random.uniform(-3., 3., size=(1, num_trajectories)), dtype=torch.float64)
        return torch.vstack([x0, x1])
    elif system == "Lorenz":
        return torch.tensor(np.random.uniform(-15, 15, size=(3, num_trajectories)), dtype=torch.float64)
    elif system == "Lotka Volterra":
        x0 = torch.tensor(np.random.uniform(1., 2., size=(
            1, num_trajectories)), dtype=torch.float64)
        x1 = torch.tensor(np.random.uniform(1., 1.5, size=(
            1, num_trajectories)), dtype=torch.float64)
        return torch.vstack([x0, x1])
    elif system == "Piecewise Linear":
        return torch.tensor(np.random.uniform(0., 1., size=(1, num_trajectories)), dtype=torch.float64)
    else:
        raise ValueError('Invalid System Name')
# Input generation function


def generate_random_inputs(num_steps, input_dim):
    return torch.tensor(np.random.uniform(-1, 1, size=(input_dim, num_steps)), dtype=torch.float64)


# Main script
def generate_and_save_data():
    systems = {
        # "Unforced Duffing": (gpk.f_UDO, 2, 0.01)
        # "van der Pol": (gpk.f_VDP, 2, 0.01)
        # "Simple Pendulum": (gpk.f_SDP, 2, 0.02),
        # "Lorenz": (gpk.f_Lorenz, 3, 0.01), # params=torch.tensor([10., 8./3., 0.8])
        # "Lotka Volterra": (gpk.f_LotkaVolterra, 2, 0.1),
        "Piecewise Linear": (gpk.f_PWL1, 1, 1.)
    }

    for system_name, (fx, state_dim, ts) in systems.items():
        trajectories = []
        initial_conditions = []

        x0 = generate_initial_conditions(system_name, num_trajectories)
        for j in range(num_trajectories):
            states = gpk.sim_RK4(
                fx, x0[:, j], ts, num_steps+1, params=None)

            trajectories.append(states)
            initial_conditions.append(x0)

        data = {
            "trajectories": torch.stack(trajectories),
            "initial_conditions": torch.stack(initial_conditions),
            "sample_time": ts,
            "num_steps": num_steps,
            "num_trajectories": num_trajectories
        }

        torch.save(data, f"Data/DataAuto_{system_name}.pt")
        print(f"Data for {
              system_name} saved to Data/DataAuto_{system_name}.pt")


if __name__ == "__main__":
    generate_and_save_data()
