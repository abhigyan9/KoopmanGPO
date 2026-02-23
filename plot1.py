# """
# Plot %NRMSE variation vs noise level for multiple models
# based on the LaTeX table provided.

# Models: Poly-eDMD, RBF-eDMD, SSID-GPK, iGPK
# Noise cases: Clean, Uniform 5%, Uniform 10%
# Values are mean ± std (in %NRMSE).
# """

# import numpy as np
# import matplotlib.pyplot as plt

# # -------------------------
# # Data from the LaTeX table
# # -------------------------
# noise_labels = ["Clean", "Gaussian 5%", "Gaussian 10%", "Gaussian 20%"]
# noise_levels = np.array([0.0, 5.0, 10.0, 20.0])  # for x-axis (percent)

# # data = {
# #     "Poly-eDMD": {"mean": [11.69, 14.21, 20.62, 30.27], "std": [13.09, 20.19, 29.10, 37.23]},
# #     "RBF-eDMD":  {"mean": [9.71, 10.03, 20.32, 43.28], "std": [6.81, 5.86, 10.72, 23.19]},
# #     # "SSID-GPK":  {"mean": [385.12, 62.23, 670.08, 43.21], "std": [363.04, 50.72, 712.52, 34.45]},
# #     "iGPK":      {"mean": [32.48, 19.71, 20.69, 33.70], "std": [51.21, 25, 20.91, 39.21]},
# # }

# data = {
#     "Poly-eDMD": {"mean": [11.69, 15.49, 24.29, 31.07], "std": [13.09, 20.19, 29.10, 37.23]},
#     "RBF-eDMD":  {"mean": [9.71, 17.46, 37.35, 65.61], "std": [6.81, 5.86, 10.72, 23.19]},
#     "SSID-GPK":  {"mean": [None, 49.57, 24.37, 20.30], "std": [363.04, 50.72, 712.52, 34.45]},
#     "iGPK":      {"mean": [32.48, 14.37, 24.86, 31.26], "std": [51.21, 25, 20.91, 39.21]},
# }

# # -------------------------
# # Plot (error bars)
# # -------------------------
# plt.figure(figsize=(5.0, 4.0))

# for model, stats in data.items():
#     mean = np.array(stats["mean"], dtype=float)
#     std = np.array(stats["std"], dtype=float)
#     # plt.errorbar(
#     #     noise_levels, mean, yerr=std,
#     #     marker="o", linewidth=2, capsize=4,
#     #     label=model
#     # )
#     plt.plot(noise_levels, mean, marker="o", linewidth=2, label=model)

# plt.xticks(noise_levels, noise_labels)
# plt.xlabel("Noise level")
# plt.ylabel("Test Error (% NRMSE)")
# # plt.title("%NRMSE vs Noise Level (mean ± 1 std)")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # -------------------------
# # Console summary
# # -------------------------
# print("\nSummary (mean ± std) in %NRMSE:")
# header = f"{'Model':<10} | {'Clean':>14} | {'Uniform 5%':>14} | {'Uniform 10%':>14} | {'Uniform 20%':>14}"
# print(header)
# print("-" * len(header))
# for model, stats in data.items():
#     vals = [f"{m:.1f} ± {s:.1f}" for m, s in zip(stats["mean"], stats["std"])]
#     print(f"{model:<10} | {vals[0]:>14} | {vals[1]:>14} | {vals[2]:>14}")

"""
Plot NLPD variation for Clean and Gaussian-noise cases
with log-scale y-axis (landscape layout).

Models: iGPK, SSID-GPK
Metrics: mean ± std NLPD
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Data from the LaTeX table
# -------------------------
conditions = ["Clean", "Gaussian 10%",
              "Gaussian 20%", "Uniform 10%", "Gaussian 20%"]
x = np.arange(len(conditions))  # categorical axis

data = {
    "iGPK": {
        "mean": [3.18, 3.32, 8.06, 2.76, 3.88],
        "std":  [0.40, 2.90, 6.70, 0.4, 2.9],
    },
    "SSID-GPK": {
        "mean": [25.12, 11.78, 108.42, 4.57, 8.64],
        "std":  [60.60, 21.40, 133.20, 6.17, 14.1],
    },
}

# -------------------------
# Plot (landscape + log y)
# -------------------------
plt.figure(figsize=(7, 3))  # landscape aspect ratio

bar_width = 0.25
offsets = [-bar_width / 2, bar_width / 2]

for (model, stats), dx in zip(data.items(), offsets):
    mean = np.array(stats["mean"], dtype=float)
    std = np.array(stats["std"], dtype=float)

    plt.bar(
        x + dx, mean, bar_width,
        yerr=std, capsize=5,
        label=model
    )

plt.xticks(x, conditions)
plt.xlabel("Noise condition")
plt.ylabel("log(NLPD)")
# plt.title("NLPD Comparison (mean ± 1 std)")
plt.yscale("log")

# Prevent pathological lower bounds in log-space
plt.ylim(bottom=0.3)

plt.grid(True, which="both", axis="y", alpha=0.35)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# -------------------------
# Console summary
# -------------------------
print("\nNLPD Summary (mean ± std):")
header = f"{'Model':<10} | {'Clean':>16} | {'Gaussian 10%':>16} | {'Gaussian 20%':>16} | {'Uniform 10%':>16} | {'Uniform 20%':>16}"
print(header)
print("-" * len(header))

for model, stats in data.items():
    vals = [f"{m:.2f} ± {s:.2f}" for m, s in zip(stats["mean"], stats["std"])]
    print(f"{model:<10} | {vals[0]:>16} | {vals[1]:>16} | {vals[2]:>16}")
