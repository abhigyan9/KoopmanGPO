import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_COLUMNS = [
    "learn_rate",
    "momentum",
    "stop_tol",
    "traj_batch_label",
    "final_full_cost",
    "best_full_cost",
    "post_mle_cost",
    "iters",
    "best_iter",
    "opt_time",
    "train_nrmse",
    "test_nrmse",
    "train_nlpd",
    "test_nlpd",
    "result_file",
]


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_results(outdir):
    rows = []
    for path in sorted(Path(outdir).glob("*.json")):
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


def print_best(df, metric, top_k):
    cols = [
        "learn_rate",
        "momentum",
        "stop_tol",
        "traj_batch_label",
        metric,
        "final_full_cost",
        "post_mle_cost",
        "iters",
        "opt_time",
    ]
    cols = [c for c in cols if c in df.columns]

    best = df.sort_values(metric, ascending=True).head(top_k)
    print("")
    print(f"Best by {metric}:")
    print(best[cols].to_string(index=False))


def save_best_csvs(df, outdir, top_k):
    summary_dir = Path(outdir) / "best_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["test_nrmse", "test_nlpd", "final_full_cost", "post_mle_cost"]:
        if metric not in df.columns:
            continue

        df.sort_values(metric, ascending=True).head(top_k).to_csv(
            summary_dir / f"best_by_{metric}.csv",
            index=False,
        )


def scatter_final_cost(df, y_metric, ylabel, outdir):
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(df["final_full_cost"], df[y_metric], alpha=0.8)
    ax.set_xlabel("Final Full Cost")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Final Full Cost vs {ylabel}")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(Path(outdir) / f"final_full_cost_vs_{y_metric}.png", dpi=250)
    plt.close(fig)


def save_plots(df, outdir):
    plots_dir = Path(outdir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    scatter_final_cost(df, "test_nrmse", "Test NRMSE [%]", plots_dir)
    scatter_final_cost(df, "test_nlpd", "Test NLPD", plots_dir)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    axes[0].scatter(df["final_full_cost"], df["test_nrmse"], alpha=0.8)
    axes[0].set_xlabel("Final Full Cost")
    axes[0].set_ylabel("Test NRMSE [%]")
    axes[0].grid(True, alpha=0.35)

    axes[1].scatter(df["final_full_cost"], df["test_nlpd"], alpha=0.8)
    axes[1].set_xlabel("Final Full Cost")
    axes[1].set_ylabel("Test NLPD")
    axes[1].grid(True, alpha=0.35)

    fig.suptitle("Grid Search Results")
    fig.tight_layout()
    fig.savefig(plots_dir / "final_full_cost_vs_test_metrics.png", dpi=250)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate iGPK grid-search JSON result files."
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)

    rows = load_results(outdir)
    if not rows:
        raise SystemExit(f"No JSON files found in {outdir}")

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    failed_rows = [r for r in rows if r.get("status") != "ok"]

    if failed_rows:
        failed_df = pd.DataFrame(failed_rows)
        failed_df.to_csv(outdir / "grid_failed_results.csv", index=False)
        print(f"Failed or incomplete runs: {len(failed_rows)}")
        print(f"Failure details saved to: {outdir / 'grid_failed_results.csv'}")

    if not ok_rows:
        raise SystemExit("No successful grid-search runs found.")

    df = pd.DataFrame(ok_rows)
    numeric_cols = [
        "learn_rate",
        "momentum",
        "stop_tol",
        "final_full_cost",
        "best_full_cost",
        "post_mle_cost",
        "iters",
        "best_iter",
        "opt_time",
        "train_nrmse",
        "test_nrmse",
        "train_nlpd",
        "test_nlpd",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sort_cols = ["test_nrmse", "test_nlpd"]
    df = df.sort_values(sort_cols, ascending=True, na_position="last")

    cols = [c for c in SUMMARY_COLUMNS if c in df.columns]
    df[cols].to_csv(outdir / "grid_results.csv", index=False)
    save_best_csvs(df, outdir, args.top_k)
    save_plots(df, outdir)

    print("")
    print(f"Successful runs: {len(df)}")
    print(f"Combined CSV saved to: {outdir / 'grid_results.csv'}")
    print_best(df, "test_nrmse", args.top_k)
    print_best(df, "test_nlpd", args.top_k)
    print("")
    print(f"Plots saved to: {outdir / 'plots'}")


if __name__ == "__main__":
    main()
