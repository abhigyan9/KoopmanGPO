import torch
import numpy as np
from matplotlib import pyplot as plt

# import functions from autonomous.py
from .autonomous import f_UDO, f_VDP, f_Lorenz, f_LotkaVolterra, f_SDP, f_PWL1, df_PWL, f_RVDP, df_scalarNL
from .autonomous import f_IPP, f_RR, f_RBDP
from .autonomous import f_uPoC
from .autonomous import sim_RK4, sim_LTI, sim_discrete

# import functions from non_autonomous.py
from .non_autonomous import sim_LTI_nonautonomous, sim_RK4_nonautonomous, fc_DO, fc_PoC, fc_SDP

# import kernels
from .kernels import Kernel, TwoPositiveParameterKernel, TwoParameterKernel
from .kernels import SumKernel, ProductKernel, KERNEL_CLASSES, build_kernel
from .kernels import GaussianKernel, ExpSineSqrKernel, RationalQuadraticKernel

# import prior means
from .prior_means import MeanFunction, ZeroMean, MonomialMean

# import dictionary, functions and classes from GPObs.py
# from .GPObs import KERNEL_FUNCTIONS
# from .GPObs import GaussianKernel, ExpSineSqrKernel, ThinSplineKernel, RationalQuadraticKernel, CosineKernel
# from .GPObs import GibbsExpAttractorKernel, ExplicitAttractorKernel
from .GPObs import getKoopman, getKoopman_control
from .GPObs import GPObservable, GPObservablesManager

# import utility functions
from .utilities import plot_phase, plot_phase_w_bounds, plot_time_series_with_bounds, plot_predicted_sd_error
from .utilities import plot_NRMSE_metrics, compare_model_predictions
from .utilities import check_pd, MatViz3d, MatViz, plot_eigen
from .utilities import get_kmeans
from .utilities import sim_and_eval
from .utilities import load_SimData, normalize_data, add_noise

# import eDMD functions
from .traditional import generate_basis, generate_basis_batch, eDMD_poly
from .traditional import rbf_observable, eDMD_RBF_kmeans
from .traditional import SSID, get_ssidgpk
