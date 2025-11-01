import torch
import numpy as np
import GPKoopman as gpk

# Manual seed for reproducibility
torch.manual_seed(1234)
np.random.seed(1234)

# Simulation parameters
ts = 0.02  # Time step
num_steps = 100  # Number of steps
num_trajectories = 100  # Number of random initial conditions

# Function to generate random initial conditions


def generate_initial_conditions(system):
    if system == "Duffing":
        return torch.tensor(np.random.uniform(-2, 2, size=(2,)), dtype=torch.float64)
    elif system == "Pendulum":
        return torch.tensor(np.random.uniform([-2, -3], [2, 3], size=(2,)), dtype=torch.float64)
    elif system == "Cart":
        return torch.tensor(np.random.uniform([2, -0.4, 0, -0.3], [4, 0.4, 1, 0.3], size=(4,)), dtype=torch.float64)

# Input generation function


def generate_random_inputs(num_steps, input_dim):
    return torch.tensor(np.random.uniform(-1, 1, size=(input_dim, num_steps)), dtype=torch.float64)


# Main script
def generate_and_save_data():
    systems = {
        # "Duffing": (gpk.fc_DO, 1, 2, 3.),
        # "Pendulum": (gpk.fc_SDP, 1, 2, 5.),
        "Cart": (gpk.fc_PoC, 1, 4, 0.),
    }

    for system_name, (fx, input_dim, state_dim, u_scaling) in systems.items():
        trajectories = []
        inputs = []
        initial_conditions = []

        for _ in range(num_trajectories):
            x0 = generate_initial_conditions(system_name)
            u = u_scaling * generate_random_inputs(num_steps+1, input_dim)
            states = gpk.sim_RK4_nonautonomous(fx, x0, ts, num_steps+1, u)

            trajectories.append(states)
            inputs.append(u)
            initial_conditions.append(x0)

        data = {
            "trajectories": torch.stack(trajectories),
            "inputs": torch.stack(inputs),
            "initial_conditions": torch.stack(initial_conditions),
            "num_trajectories": num_trajectories,
            "sample_time": ts,
            "num_steps": num_steps,
        }

        torch.save(data, f"Data/DataAuto_{system_name}_data.pt")
        print(f"Data for {system_name} saved to Data/{system_name}_data.pt")


if __name__ == "__main__":
    generate_and_save_data()
