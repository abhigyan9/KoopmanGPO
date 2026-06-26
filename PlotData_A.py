import torch
import matplotlib.pyplot as plt

# Function to load and plot data


def plot_trajectories(system_name, x_idx1, x_idx2, title):
    # Load data
    data = torch.load(f"Data/DataAuto_{system_name}.pt", weights_only=True)

    # Shape: (num_trajectories, state_dim, num_steps)
    trajectories = data["trajectories"]
    ts = data["sample_time"]
    num_trajectories = data["num_trajectories"]
    num_steps = data["num_steps"]

    # Plot phase plots
    plt.figure(figsize=(8, 6))
    for j in range(num_trajectories):
        # Extract phase plot data
        x1 = trajectories[j, x_idx1, :]
        x2 = trajectories[j, x_idx2, :]

        # Plot trajectory
        plt.plot(x1, x2, color='blue', linewidth=0.5)

        # Mark initial condition
        plt.plot(x1[0], x2[0], 'o', color='red')

    plt.title(f"{title}, {num_trajectories} trajectories, {
              num_steps} steps, ts={ts}s")
    plt.xlabel(f"X{x_idx1+1}")
    plt.ylabel(f"X{x_idx2+1}")
    plt.grid()
    plt.savefig(f'Plots/PhasePlot_{system_name}.png',
                dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # # Plot Duffing Oscillator phase plot and inputs (x[0] vs x[1])
    # plot_trajectories("Unforced Duffing", x_idx1=0, x_idx2=1,
    #                   title="Unforced Duffing Oscillator")

    # # van der Pol Oscillator
    # plot_trajectories("van der Pol", x_idx1=0, x_idx2=1,
    #                   title="van der Pol Oscillator")

    # # Reverse van der Pol Oscillator
    # plot_trajectories("Reverse van der Pol", x_idx1=0, x_idx2=1,
    #                   title="Reverse van der Pol Oscillator")
    
    # # Plot Simple Damped Pendulum phase plot and inputs (x[0] vs x[1])
    # plot_trajectories("Simple Pendulum", x_idx1=0, x_idx2=1,
    #                   title="Simple Damped Pendulum")

    # # Plot Lotka Volterra phase plot and inputs (x[0] vs x[2])
    # plot_trajectories("Lotka Volterra", x_idx1=0, x_idx2=1,
    #                   title="Lotka Volterra Dynamics")

    plot_trajectories("Cart_data", x_idx1=0, x_idx2=2,
                      title="Unforced PoC")
    
    # plot_trajectories("Reciprocal Relaxer", x_idx1=0, x_idx2=1,
    #                   title="Reciprocal Relaxer")
    
    plot_trajectories("Reciprocal-Biased Damped Pendulum", x_idx1=0, x_idx2=1,
                      title="Reciprocal Biased Damped Pendulum")
    
    plot_trajectories("Inhibited Predator-Prey", x_idx1=0, x_idx2=1,
                      title="Predator-Prey with inhibited predation")
