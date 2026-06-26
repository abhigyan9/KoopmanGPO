# Overview
Welcome to the repository for iGPK - Inverted Gaussian Process optimization based Koopman operator Discovery. This repository includes the GPKoopman package, and necessary experiments scripts and Jupyter notebooks.

Please note that this repository is under active iterative development and some parts might still be a work in progress.

## Checklist:
Currently, GPKoopman is only on Windows (x86 CPU with Nvidia GPU) and requires
- [Python 3.12.XX](https://www.python.org/downloads/release/python-31210/) (3.12.10 recommended)
- [Git](https://git-scm.com/install/windows)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)


# Repository Structure
```
 ├── GPKoopman/
 │ ├── __init__.py          # Public package imports
 │ ├── autonomous.py        # Autonomous systems and simulation routines
 │ ├── GPObs.py             # GPObservable and GPObservablesManager
 │ ├── kernels.py           # GP covariance kernels
 │ ├── prior_means.py       # GP prior-mean functions
 │ ├── utilities.py         # Data, simulation, evaluation, and plotting utilities
 │ └── traditional.py       # Baseline Koopman methods
 ├── Data/                  # Generated trajectory datasets
 ├── Figures/               # Figures and experiment outputs
 ├── get_iGPK_fcn.py        # Main iGPK training routine
 ├── scalarNL_script.py     # Single-experiment and baseline comparison driver
 ├── sweep_noise.py         # Experiment sweep across noise types and intensities
 ├── TrajDataGen_A.py       # Generates trajectory datasets (will be replaced)
 └── README.md
```
Apart from these, the repository also has several Jupyter Notebooks for different kinds of experiments
- system_plots.ipynb: Trajectory & Metrics plotted for different systems
- d4rl_halfcheetah_exp.ipynb: Experimental files - not useful otherwise
- gridsearch.ps1: Searches best optimizer parameters via gridsearch using the gridsearch_runner.py and gridsearch_results.py scripts 
- seed_sensitivity.ps1: Runs different hp and Z seeds using associated scripts.
- run_numtraj_sweep.ps1: Runs training at different number of training trajectories using run_igpk_numtraj.py

Please ignore all other files.

# Installation

To install, make sure the prerequisites are met and then clone this repository
```
git clone https://github.com/abhigyan9/KoopmanGPO.git
```

Create and activate a Python 3.12 virtual environemnt (3.12.10 recommended)
```
py -3.12 -m venv .gpk312
.\.gpk312\Scripts\Activate.ps1
```

Install the GPKoopman package with extra identifier for CUDA Support
```
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu124
```

# References

This repository contains an implementation of the paper:

Majumdar, A., Mojahed, N., & Nazari, S. (2025). Inverted Gaussian Process Optimization for Probabilistic Koopman Operator Discovery. arXiv preprint arXiv:2504.00304 [[arxiv]](https://arxiv.org/abs/2504.00304)

```
@article{majumdar2025inverted,
  title={Inverted Gaussian Process Optimization for Probabilistic Koopman Operator Discovery},
  author={Majumdar, Abhigyan and Mojahed, Navid and Nazari, Shima},
  journal={arXiv preprint},
  year={2025},
  doi={10.48550/arXiv.2504.00304},
}
```
