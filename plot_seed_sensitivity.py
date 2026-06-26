import os
import math
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import GPKoopman as gpk
from get_iGPK_new import get_iGPK
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

runs = torch.load('Figures/iGPK-Seed_Sensitivity/iGPK-SEED_sweep-runs.pt', weights_only=True)
OUTDIR = f"Figures/iGPK-Seed_Sensitivity"

# ---------------------------
# 3) PLOT AND SAVE RESULTS  #
# ---------------------------
# Gather major Results
cost_pre, cost_post = [], []
train_nrmse, test_nrmse = [], []
train_nlpd, test_nlpd = [], []
for run in runs:
    cost_pre.append(float(run['cost_history'][-1]))
    cost_post.append(float(run['post_mle_cost']))
    train_nrmse.append(float(100*run['Train']['NRMSE']))
    train_nlpd.append(float(run['Train']['NLPD']))
    test_nrmse.append(float(100*run['Test']['NRMSE']))
    test_nlpd.append(float(run['Test']['NLPD']))

if True:    # PLOT 1 : COST HISTORY
    plt.figure(figsize=(10, 6))
    for r in runs:
        ch = r["cost_history"]
        if ch is None or len(ch) == 0:
            continue
        ch_plot = np.clip(ch, 1e-16, None)
        plt.plot(np.arange(len(ch_plot)), ch_plot,
                linewidth=1.0, alpha=0.85, label=r["tag"])

    plt.yscale("log")
    plt.xlabel("GD Iteration")
    plt.ylabel("Training Cost (log scale)")
    plt.title("iGPK Cost Histories Across Initializations")
    if len(runs) <= 16:
        plt.legend(fontsize=8, ncol=2)
    else:
        plt.text(0.01, 0.01, f"{len(runs)} runs (legend suppressed)",
                transform=plt.gca().transAxes, fontsize=9, va="bottom")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "cost_histories_log.png"), dpi=200)
    plt.grid()
    plt.close()

# Plot 2: final_train_cost distribution
if True:    # PLOT 2 : PRE AND POST MLE COST
    plt.figure(figsize=(5, 5))
    plt.scatter(cost_pre, cost_post, alpha=0.75)
    # plt.xscale('log'), plt.yscale('log')
    plt.xlabel('Pre-MLE Cost'), plt.ylabel('Post-MLE Cost')
    plt.grid()
    plt.title('Post-MLE v/s Pre-MLE Training Cost')
    plt.tight_layout()
    plt.savefig(os.path.join(
        OUTDIR, "iGPK-pr_vs_post-train_cost.png"), dpi=200)
    plt.close()

if True:    # PLOT 3 : PRE and POST MLE COST V/S MEAN TRAIN %-NRMSE
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(8, 5))
    ax[0].scatter(cost_pre, train_nrmse, alpha=0.75)
    ax[1].scatter(cost_post, train_nrmse, alpha=0.75)
    # ax[0].set_xscale('log'), ax[1].set_xscale('log')
    ax[0].set_xlabel('Pre-MLE Cost'), ax[1].set_xlabel('Post-MLE Cost')
    ax[0].set_ylabel('Train NRMSE [%]')
    # Cleaner ticks
    for a in ax:
        a.grid(True)
        a.xaxis.set_major_locator(MaxNLocator(nbins=5))
        # Consistent decimal formatting
        a.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    fig.suptitle('Train NRMSE [%] v/s Pre and Post-MLE Training Cost')
    plt.tight_layout()
    plt.savefig(os.path.join(
        OUTDIR, "iGPK-train_nrmse.png"), dpi=200)
    plt.close()

if True:    # PLOT 4 : POST-MLE COST V/S MEAN TEST %-NRMSE
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(8, 5))
    ax[0].scatter(cost_pre, test_nrmse, alpha=0.75)
    ax[1].scatter(cost_post, test_nrmse, alpha=0.75)
    # ax[0].set_xscale('log'), ax[1].set_xscale('log')
    ax[0].set_xlabel('Pre-MLE Cost'), ax[1].set_xlabel('Post-MLE Cost')
    ax[0].set_ylabel('Test NRMSE [%]')
    # Cleaner ticks
    for a in ax:
        a.grid(True)
        a.xaxis.set_major_locator(MaxNLocator(nbins=5))
        # Consistent decimal formatting
        a.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    fig.suptitle('Test NRMSE [%] v/s Pre and Post-MLE Training Cost')
    plt.tight_layout()
    plt.savefig(os.path.join(
        OUTDIR, "iGPK-test_nrmse.png"), dpi=200)
    plt.close()

if True:    # PLOT 5 : POST-MLE COST V/S MEAN TEST NLPD
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(8, 5))
    ax[0].scatter(cost_pre, train_nlpd, alpha=0.75)
    ax[1].scatter(cost_post, train_nlpd, alpha=0.75)
    # ax[0].set_xscale('log'), ax[1].set_xscale('log')
    ax[0].set_xlabel('Pre-MLE Cost'), ax[1].set_xlabel('Post-MLE Cost')
    ax[0].set_ylabel('Train NLPD')
    # Cleaner ticks
    for a in ax:
        a.grid(True)
        a.xaxis.set_major_locator(MaxNLocator(nbins=5))
        # Consistent decimal formatting
        a.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    fig.suptitle('Train NLPD v/s Pre and Post-MLE Training Cost')
    plt.tight_layout()
    plt.savefig(os.path.join(
        OUTDIR, "iGPK-train_nlpd.png"), dpi=200)
    plt.close()

if True:    # PLOT 6 : POST-MLE COST V/S MEAN TEST NLPD
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(8, 5))
    ax[0].scatter(cost_pre, test_nlpd, alpha=0.75)
    ax[1].scatter(cost_post, test_nlpd, alpha=0.75)
    # ax[0].set_xscale('log'), ax[1].set_xscale('log')
    ax[0].set_xlabel('Pre-MLE Cost'), ax[1].set_xlabel('Post-MLE Cost')
    ax[0].set_ylabel('Test NLPD')
    # Cleaner ticks
    for a in ax:
        a.grid(True)
        a.xaxis.set_major_locator(MaxNLocator(nbins=5))
        # Consistent decimal formatting
        a.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    fig.suptitle('Test NLPD v/s Pre and Post-MLE Training Cost')
    plt.tight_layout()
    plt.savefig(os.path.join(
        OUTDIR, "iGPK-test_nlpd.png"), dpi=200)
    plt.close()

print(f'Finishied Plotting')
