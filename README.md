# Overview
Welcome to the repository for iGPK - Inverted Gaussian Process optimization based Koopman operator Discovery. This repository includes the GPKoopman package, and necessary experiments scripts and Jupyter notebooks.

## Checklist:
Currently, GPKoopman is only on Windows (x86 with Nvidia GPU) and requires
- [Python 3.12.XX](https://www.python.org/downloads/release/python-31210/) (3.12.10 recommended)
- [Git](https://git-scm.com/install/windows)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)


# Repository Structure
```
 ├── GPKoopman/
 │ ├── __init__.py      # Public package imports
 │ ├── autonomous.py    # Autonomous systems and simulation routines
 │ ├── GPObs.py         # GPObservable and GPObservablesManager
 │ ├── kernels.py       # GP covariance kernels
 │ ├── prior_means.py   # GP prior-mean functions
 │ ├── utilities.py     # Data, simulation, evaluation, and plotting utilities
 │ └── traditional.py   # Baseline Koopman methods
 ├── Data/              # Generated trajectory datasets
 ├── Figures/           # Figures and experiment outputs
 ├── get_iGPK_fcn.py    # Main iGPK training routine
 ├── scalarNL_script.py # Single-experiment and baseline comparison driver
 ├── sweep_noise.py     # Experiment sweep across noise types and intensities
 ├── TrajDataGen_A.py   # Generates trajectory datasets (will be replaced)
 ├── TrajDataGen_NA.py  # Generates trajectories for controlled systems
 └── README.md
```
Apart from these, the repository also has several Jupyter Notebooks for different kinds of experiments
- plotting_siam.ipynb: Plots for journal submissions and NSF Proposal
- Playground_<...>.ipynb: Experimental files - not useful otherwise
- gridsearch.ps1: Searches best optimizer parameters via gridsearch using the gridsearch_runner.py and gridsearch_results.py scripts 
- seed_sensitivity.ps1: Runs different hp and Z seeds using associated scripts.
- run_numtraj_sweep.ps1: Runs training at different number of training trajectories using run_igpk_numtraj.py

Please ignore all other files.

# Installation

To install, first clone this repository
```
git clone https://github.com/abhigyan9/KoopmanGPO.git
```

Create and activate a Python 3.12 virtual environemnt
```
python -3.12 -m venv .gpk312
.\gkp312\Scripts\Activate.ps1
```

Install the GPKoopman package with extra identifier for CUDA Support
```
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu124
```

