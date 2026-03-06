#!/usr/bin/env python3
"""
Plot degree-scaling summary for matrix inversion and uniform singular value
amplification (data produced by degree_scaling_data.py).

Reads CSV with degree, max_error_poly, max_error_qsp, constraint_violated, etc.,
and produces a log-log plot of polynomial degree vs maximum error vs target,
with the same style as degree_scaling_thresh_proj (polynomial: x if constraints
violated, o if satisfied; retraction: o).
"""

import argparse
import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator
from scipy.fft import dct

BLUE = "#0072B2"
MAIZE = "#E69F00"


def chebval_dct(c: np.ndarray, M: int) -> np.ndarray:
    """
    Evaluate Chebyshev series with full coefficients c at M Chebyshev nodes.

    Nodes: x_j = cos(j*pi/(M-1)), j=0..M-1.
    Implementation: zero-pad coefficients to length M and apply DCT-I.
    """
    if M < 2:
        raise ValueError("M must be >= 2 for DCT-I evaluation")
    c = np.asarray(c, dtype=float)
    if len(c) > M:
        raise ValueError("Number of coefficients exceeds desired number of samples M")
    c_pad = np.zeros(M, dtype=float)
    c_pad[: len(c)] = c
    if M > 2:
        c_pad[1 : M - 1] *= 0.5
    return dct(c_pad, type=1, norm=None)


def _parse_constraint_violated(val) -> Optional[bool]:
    """Parse constraint_violated from CSV (bool or string 'True'/'False'). Returns None if missing/NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("true", "1"):
        return True
    if s in ("false", "0"):
        return False
    return None


def _constraint_violated_from_row(row, M_cheb: int = 1_000_000) -> bool:
    """Compute constraint_violated from coef_full in a CSV row (no CSV write)."""
    coef_full = np.array(json.loads(row["coef_full"]), dtype=float)
    try:
        vals = chebval_dct(coef_full, M_cheb)
    except MemoryError:
        vals = chebval_dct(coef_full, 100_000)
    return (np.max(np.abs(vals)) - 1.0) > 0.0


def plot_summary(
    csv_path: str,
    fig_path: str,
    leave_out_last: bool = False,
    max_exp: Optional[int] = None,
) -> None:
    """Plot degree (log) vs max error (log) for fit and QSP.

    Polynomial: x if constraint violated, o (larger) if satisfied. Retraction: o (larger).
    Does not write to CSV. If max_exp is set (e.g. 8), only plot degrees with exp2 <= max_exp.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    data = pd.read_csv(csv_path)
    if "degree" not in data.columns:
        raise ValueError("CSV must contain 'degree' column.")
    if "max_error_poly" not in data.columns or "max_error_qsp" not in data.columns:
        raise ValueError("CSV must contain 'max_error_poly' and 'max_error_qsp' columns.")

    if "constraint_violated" not in data.columns:
        data["constraint_violated"] = None
    constraint_violated_list = []
    for idx, row in data.iterrows():
        cached = _parse_constraint_violated(row.get("constraint_violated"))
        if cached is not None:
            constraint_violated_list.append(cached)
        elif "coef_full" in row and pd.notna(row.get("coef_full")):
            constraint_violated_list.append(_constraint_violated_from_row(row))
        else:
            constraint_violated_list.append(False)
    data["constraint_violated"] = constraint_violated_list

    data = data.sort_values("degree").reset_index(drop=True)
    if leave_out_last and len(data) > 0:
        data = data.iloc[:-1, :].reset_index(drop=True)
    if max_exp is not None:
        if "exp2" not in data.columns:
            data = data[data["degree"] <= 2**max_exp].reset_index(drop=True)
        else:
            data = data[data["exp2"] <= max_exp].reset_index(drop=True)
    degrees = data["degree"].to_numpy(dtype=float)
    exponents = (
        data["exp2"].to_numpy(dtype=float)
        if "exp2" in data.columns
        else np.log2(degrees)
    )
    err_poly = data["max_error_poly"].to_numpy(dtype=float)
    err_qsp = data["max_error_qsp"].to_numpy(dtype=float)
    constraint_violated = data["constraint_violated"].to_numpy(dtype=bool)

    degrees_clamped = np.where(degrees > 0.0, degrees, 1e-16)
    err_poly_clamped = np.where(err_poly > 0.0, err_poly, 1e-16)
    err_qsp_clamped = np.where(err_qsp > 0.0, err_qsp, 1e-16)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 12

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(degrees_clamped)):
        if constraint_violated[i]:
            ax.plot(
                degrees_clamped[i],
                err_poly_clamped[i],
                "x",
                color=BLUE,
                markersize=8,
                markeredgewidth=2,
            )
        else:
            ax.plot(
                degrees_clamped[i],
                err_poly_clamped[i],
                "o",
                color=BLUE,
                markersize=7,
                markeredgewidth=1,
                fillstyle="none",
            )
        ax.plot(
            degrees_clamped[i],
            err_qsp_clamped[i],
            "o",
            color=MAIZE,
            markersize=5,
            markeredgewidth=1,
            fillstyle="none",
        )
    ax.plot(degrees_clamped, err_poly_clamped, "k--", alpha=0.7, linewidth=1)
    ax.plot(degrees_clamped, err_qsp_clamped, "k--", alpha=0.7, linewidth=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Maximum error vs target")
    ax.xaxis.set_major_locator(FixedLocator(degrees_clamped))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks(degrees_clamped)
    ax.set_xticklabels([rf"$2^{{{e:g}}}$" for e in exponents])
    ax.grid(True, which="both", alpha=0.3)
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="x",
            color=BLUE,
            linestyle="None",
            markersize=8,
            markeredgewidth=2,
            label="Polynomial max error (constraints violated)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=BLUE,
            linestyle="None",
            markersize=7,
            fillstyle="none",
            label="Polynomial max error (constraints satisfied)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=MAIZE,
            linestyle="None",
            markersize=5,
            fillstyle="none",
            label="Retracted polynomial max error",
        ),
    ]
    ax.legend(handles=legend_elements, loc="best")

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Figure saved to: {fig_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot degree-scaling summary (degree vs max error) from CSV "
            "produced by degree_scaling_data.py."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV file (e.g. data/degree_scaling_mat_inv_npts8.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output figure path (default: figures/degree_scaling_<base>.pdf from CSV dir).",
    )
    parser.add_argument(
        "--leave-out-last",
        action="store_true",
        help="Omit the largest degree from the plot.",
    )
    parser.add_argument(
        "--max-exp",
        type=int,
        default=None,
        metavar="K",
        help="Only include degrees with exp2 <= K (e.g. --max-exp 8 for degree <= 256).",
    )
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.isabs(csv_path) and not os.path.exists(csv_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt = os.path.join(script_dir, "data", os.path.basename(csv_path))
        if os.path.exists(alt):
            csv_path = alt

    if args.output is not None:
        fig_path = args.output
        out_dir = os.path.dirname(os.path.abspath(fig_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")
        figures_dir = os.path.join(script_dir, "figures")
        base = os.path.splitext(os.path.basename(csv_path))[0]
        fig_path = os.path.join(figures_dir, f"{base}.pdf")
        os.makedirs(figures_dir, exist_ok=True)

    plot_summary(
        csv_path,
        fig_path,
        leave_out_last=args.leave_out_last,
        max_exp=args.max_exp,
    )


if __name__ == "__main__":
    main()
