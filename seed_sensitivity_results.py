import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


SUMMARY_COLUMNS = [
    "z_seed",
    "hp_seed",
    "pre_mle_cost",
    "post_mle_cost",
    "train_nrmse",
    "test_nrmse",
    "train_nlpd",
    "test_nlpd",
    "iters",
    "best_iter",
    "opt_time",
    "result_file",
]

STAT_COLUMNS = [
    "pre_mle_cost",
    "post_mle_cost",
    "train_nrmse",
    "test_nrmse",
    "train_nlpd",
    "test_nlpd",
]

REFERENCE_Z_SEED = 1234
REFERENCE_HP_SEED = 1234


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_results(outdir):
    rows = []
    for path in sorted(Path(outdir).glob("seed_z-*_seed_hp-*.json")):
        try:
            row = load_json(path)
        except json.JSONDecodeError as exc:
            row = {
                "status": "failed",
                "result_file": str(path),
                "error_type": "JSONDecodeError",
                "error": str(exc),
            }

        row.setdefault("result_file", str(path))
        rows.append(row)

    return rows


def finite_or_none(value):
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def write_json(payload, path):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def build_stats(df):
    stats = {}
    for col in STAT_COLUMNS:
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            stats[col] = {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
            }
            continue

        stats[col] = {
            "min": finite_or_none(values.min()),
            "max": finite_or_none(values.max()),
            "mean": finite_or_none(values.mean()),
            "std": finite_or_none(values.std(ddof=0)),
        }

    return stats


def reference_seed_rows(df):
    return df[
        (df["z_seed"] == REFERENCE_Z_SEED)
        & (df["hp_seed"] == REFERENCE_HP_SEED)
    ]


def add_reference_seed_marker(ax, df, x_col, y_col, label=True):
    ref = reference_seed_rows(df)
    if ref.empty:
        return

    ax.scatter(
        ref[x_col],
        ref[y_col],
        color="red",
        marker="*",
        s=180,
        edgecolors="black",
        linewidths=0.6,
        label="z=1234, hp=1234" if label else None,
        zorder=5,
    )


def plot_pre_vs_post(df, plots_dir):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df["pre_mle_cost"], df["post_mle_cost"], alpha=0.75)
    add_reference_seed_marker(ax, df, "pre_mle_cost", "post_mle_cost")
    ax.set_xlabel("Pre-MLE Cost")
    ax.set_ylabel("Post-MLE Cost")
    ax.set_title("Post-MLE vs Pre-MLE Training Cost")
    ax.grid(True)
    if not reference_seed_rows(df).empty:
        ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "iGPK-pr_vs_post-train_cost.png", dpi=200)
    plt.close(fig)


def plot_metric_vs_costs(df, metric, ylabel, title, fname, plots_dir):
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(8, 5))
    ax[0].scatter(df["pre_mle_cost"], df[metric], alpha=0.75)
    ax[1].scatter(df["post_mle_cost"], df[metric], alpha=0.75)
    add_reference_seed_marker(ax[0], df, "pre_mle_cost", metric)
    add_reference_seed_marker(ax[1], df, "post_mle_cost", metric, label=False)
    ax[0].set_xlabel("Pre-MLE Cost")
    ax[1].set_xlabel("Post-MLE Cost")
    ax[0].set_ylabel(ylabel)

    for a in ax:
        a.grid(True)
        a.xaxis.set_major_locator(MaxNLocator(nbins=5))
        a.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    fig.suptitle(title)
    if not reference_seed_rows(df).empty:
        ax[0].legend()
    fig.tight_layout()
    fig.savefig(plots_dir / fname, dpi=200)
    plt.close(fig)


def plot_surface(df, value_col, zlabel, title, fname, plots_dir):
    z_vals = sorted(df["z_seed"].dropna().unique())
    hp_vals = sorted(df["hp_seed"].dropna().unique())
    grid = np.full((len(z_vals), len(hp_vals)), np.nan, dtype=float)

    z_to_i = {z: i for i, z in enumerate(z_vals)}
    hp_to_j = {h: j for j, h in enumerate(hp_vals)}

    for row in df.itertuples(index=False):
        grid[z_to_i[row.z_seed], hp_to_j[row.hp_seed]] = float(getattr(row, value_col))

    hp_grid, z_grid = np.meshgrid(hp_vals, z_vals)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(hp_grid, z_grid, grid, linewidth=0, antialiased=True)
    ax.set_xlabel("HP Seed")
    ax.set_ylabel("Z Seed")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=zlabel)
    fig.tight_layout()
    fig.savefig(plots_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(df, value_col, zlabel, title, fname, plots_dir):
    z_vals = sorted(df["z_seed"].dropna().unique())
    hp_vals = sorted(df["hp_seed"].dropna().unique())
    grid = np.full((len(z_vals), len(hp_vals)), np.nan, dtype=float)

    z_to_i = {z: i for i, z in enumerate(z_vals)}
    hp_to_j = {h: j for j, h in enumerate(hp_vals)}

    for row in df.itertuples(index=False):
        grid[z_to_i[row.z_seed], hp_to_j[row.hp_seed]] = float(getattr(row, value_col))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, origin="lower", aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(hp_vals)))
    ax.set_yticks(np.arange(len(z_vals)))
    ax.set_xticklabels(hp_vals)
    ax.set_yticklabels(z_vals)
    ax.set_xlabel("HP Seed")
    ax.set_ylabel("Z Seed")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(zlabel)
    fig.tight_layout()
    fig.savefig(plots_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_plots(df, outdir):
    plots_dir = Path(outdir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_pre_vs_post(df, plots_dir)
    plot_metric_vs_costs(
        df,
        "train_nrmse",
        "Train NRMSE [%]",
        "Train NRMSE [%] vs Pre and Post-MLE Training Cost",
        "iGPK-train_nrmse.png",
        plots_dir,
    )
    plot_metric_vs_costs(
        df,
        "test_nrmse",
        "Test NRMSE [%]",
        "Test NRMSE [%] vs Pre and Post-MLE Training Cost",
        "iGPK-test_nrmse.png",
        plots_dir,
    )
    plot_metric_vs_costs(
        df,
        "train_nlpd",
        "Train NLPD",
        "Train NLPD vs Pre and Post-MLE Training Cost",
        "iGPK-train_nlpd.png",
        plots_dir,
    )
    plot_metric_vs_costs(
        df,
        "test_nlpd",
        "Test NLPD",
        "Test NLPD vs Pre and Post-MLE Training Cost",
        "iGPK-test_nlpd.png",
        plots_dir,
    )

    plot_surface(
        df,
        "pre_mle_cost",
        "Pre-MLE Cost",
        "Pre-MLE Cost Surface Across Seeds",
        "pre_mle_cost_surface.png",
        plots_dir,
    )
    plot_heatmap(
        df,
        "pre_mle_cost",
        "Pre-MLE Cost",
        "Pre-MLE Cost Heatmap Across Seeds",
        "pre_mle_cost_heatmap.png",
        plots_dir,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate iGPK seed-sensitivity JSON result files."
    )
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)

    rows = load_results(outdir)
    if not rows:
        raise SystemExit(f"No seed sensitivity JSON files found in {outdir}")

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    failed_rows = [r for r in rows if r.get("status") != "ok"]

    if failed_rows:
        failed_df = pd.DataFrame(failed_rows)
        failed_df.to_csv(outdir / "seed_sensitivity_failed_results.csv", index=False)
        print(f"Failed or incomplete runs: {len(failed_rows)}")
        print(f"Failure details saved to: {outdir / 'seed_sensitivity_failed_results.csv'}")

    if not ok_rows:
        raise SystemExit("No successful seed-sensitivity runs found.")

    df = pd.DataFrame(ok_rows)
    numeric_cols = [
        "z_seed",
        "hp_seed",
        "pre_mle_cost",
        "post_mle_cost",
        "train_nrmse",
        "test_nrmse",
        "train_nlpd",
        "test_nlpd",
        "iters",
        "best_iter",
        "opt_time",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["z_seed", "hp_seed"], ascending=True, na_position="last")

    cols = [c for c in SUMMARY_COLUMNS if c in df.columns]
    df[cols].to_csv(outdir / "seed_sensitivity_results.csv", index=False)

    stats_payload = {
        "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "num_successful_runs": int(len(df)),
        "num_failed_runs": int(len(failed_rows)),
        "statistics": build_stats(df),
    }
    write_json(stats_payload, outdir / "seed_sensitivity_summary_stats.json")

    save_plots(df, outdir)

    print("")
    print(f"Successful runs: {len(df)}")
    print(f"Combined CSV saved to: {outdir / 'seed_sensitivity_results.csv'}")
    print(f"Summary stats saved to: {outdir / 'seed_sensitivity_summary_stats.json'}")
    print(f"Plots saved to: {outdir / 'plots'}")


if __name__ == "__main__":
    main()
