import torch
import matplotlib.pyplot as plt

# Function to load and plot data


def plot_trajectories_and_inputs(system_name, x_idx1, x_idx2, title):
    # Load data
    data = torch.load(f"Data/{system_name}_data.pt", weights_only=True)

    # Shape: (num_trajectories, state_dim, num_steps)
    trajectories = data["trajectories"]
    inputs = data["inputs"]  # Shape: (num_trajectories, input_dim, num_steps)
    ts = data["sample_time"]
    num_trajectories = trajectories.shape[0]
    num_steps = trajectories.shape[2] - 1

    # Plot phase plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for j in range(num_trajectories):
        # Extract phase plot data
        x1 = trajectories[j, x_idx1, :]
        x2 = trajectories[j, x_idx2, :]

        # Plot trajectory
        plt.plot(x1, x2, color='blue', linewidth=0.5)

        # Mark initial condition
        plt.plot(x1[0], x2[0], 'o', color='red')

    plt.title(f"{title}: Phase Plot")
    plt.xlabel(f"x{x_idx1+1}")
    plt.ylabel(f"x{x_idx2+1}")
    plt.grid()

    # Plot input sequences
    plt.subplot(1, 2, 2)
    time = torch.arange(0, num_steps + 1) * ts
    for j in range(num_trajectories):
        for i in range(inputs.shape[1]):
            plt.plot(time, inputs[j, i, :], linewidth=0.5,
                     label=f"Input {i+1}" if j == 0 else "")

    plt.title(f"{title}: Input Sequences")
    plt.xlabel("Time (s)")
    plt.ylabel("Input Value")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(f'Plots/ForcedPlot_{system_name}.png',
                dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # Plot Duffing Oscillator phase plot and inputs (x[0] vs x[1])
    plot_trajectories_and_inputs(
        "Duffing", x_idx1=0, x_idx2=1, title="Duffing Oscillator")

    # Plot Simple Damped Pendulum phase plot and inputs (x[0] vs x[1])
    plot_trajectories_and_inputs(
        "Pendulum", x_idx1=0, x_idx2=1, title="Simple Damped Pendulum")

    # Plot Inverted Pendulum on Cart phase plot and inputs (x[0] vs x[2])
    plot_trajectories_and_inputs(
        "Cart", x_idx1=0, x_idx2=2, title="Inverted Pendulum on Cart")
