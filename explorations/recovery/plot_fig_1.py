#!/usr/bin/env python3
"""
Plot Figure 1: maximum constraint violation vs discretization size npts.

Reads:
  explorations/recovery/data/fig_1_constraint_violation_vs_npts.csv

and produces:
  explorations/recovery/figures/fig_1.pdf

The plot is a log-log scatter with dual y-axes:
  x-axis: npts (number of discretization points)
  left y-axis: max_violation (blue x's)
  right y-axis: optimization time in seconds (orange x's)
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Fig. 1: max constraint violation vs npts."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file (default: explorations/recovery/data/fig_1_constraint_violation_vs_npts.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output figure path (default: explorations/recovery/figures/fig_1.pdf).",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    figures_dir = os.path.join(script_dir, "figures")

    if args.csv is None:
        csv_path = os.path.join(data_dir, "fig_1_constraint_violation_vs_npts.csv")
    else:
        csv_path = args.csv

    if args.output is None:
        os.makedirs(figures_dir, exist_ok=True)
        fig_path = os.path.join(figures_dir, "fig_1.pdf")
    else:
        fig_path = args.output
        out_dir = os.path.dirname(os.path.abspath(fig_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    data = pd.read_csv(csv_path)
    if "npts" not in data.columns or "max_violation" not in data.columns:
        raise ValueError("CSV must contain 'npts' and 'max_violation' columns.")
    if "iteration_time" not in data.columns:
        raise ValueError("CSV must contain 'iteration_time' for dual-axis plot.")

    # Sort by npts
    data = data.sort_values("npts").reset_index(drop=True)

    x = data["npts"].to_numpy(dtype=float)
    y = data["max_violation"].to_numpy(dtype=float)
    t = data["iteration_time"].to_numpy(dtype=float)

    # Clamp zero/negative values for log scale
    y_clamped = np.where(y > 0.0, y, 1e-16)
    t_clamped = np.where(t > 0.0, t, 1e-16)

    # Use serif fonts consistent with other recovery plots
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 12

    fig, ax = plt.subplots(figsize=(8, 6))

    # Left y-axis: max violation (blue, polynomial approximation color)
    ax.scatter(x, y_clamped, marker="x", color="#0072B2", s=60, linewidths=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of discretization points")
    ax.set_ylabel("Maximum constraint violation\n$\\max_{x \\in [-1,1]} |f(x)| - 1$", color="#0072B2")
    ax.tick_params(axis="y", labelcolor="#0072B2")

    # Right y-axis: optimization time (orange, retraction color)
    ax2 = ax.twinx()
    ax2.scatter(x, t_clamped, marker="x", color="#E69F00", s=60, linewidths=2)
    ax2.set_yscale("log")
    ax2.set_ylabel("Optimization time (s)", color="#E69F00", rotation=270, labelpad=15)
    ax2.tick_params(axis="y", labelcolor="#E69F00")

    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Figure saved to: {fig_path}")


if __name__ == "__main__":
    main()

