#!/usr/bin/env python3
"""Plot the degree-181 target and constrained approximation polynomials."""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.chebyshev import chebval

from constrainapprox import solve_odd_bounded_approx
from degree_scaling_data_odd import _qsp_chebyshev_coefficients


DEFAULT_FIG7_CSV = "data/degree_scaling_uniform_sv_amp_npts19_solver2_odd.csv"
DEFAULT_CACHE = "data/degree_181_exact_bound_sdp_coefficients.npz"
DEFAULT_PDF = "figures/degree_181_method_comparison.pdf"
DEFAULT_PNG = "figures/degree_181_method_comparison.png"


def load_retracted_coefficients(csv_path: str, degree: int) -> np.ndarray:
    data = pd.read_csv(csv_path)
    rows = data[data["degree"] == degree]
    if len(rows) != 1:
        raise ValueError(f"Expected one Figure 7 row at degree {degree}, found {len(rows)}")
    phi_proc = np.asarray(json.loads(rows.iloc[0]["phi_proc"]), dtype=float)
    coefficients, _ = _qsp_chebyshev_coefficients(degree, phi_proc, parity=1)
    return coefficients


def load_or_solve_sdp_coefficients(
    cache_path: str,
    degree: int,
    a: float,
    epsilon: float,
    npts: int,
    solver: str,
) -> np.ndarray:
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if int(cached["degree"]) != degree:
            raise ValueError(f"Cached coefficients are not for degree {degree}")
        return np.asarray(cached["coefficients"], dtype=float)

    print(
        f"Solving exact-bound SDP at degree {degree} with {solver.upper()}; "
        "coefficients were not saved by the sweep"
    )
    target = lambda x: (1.0 - epsilon) * x / a
    odd_coefficients, diagnostics = solve_odd_bounded_approx(
        target,
        degree=degree,
        fit_intervals=[0.0, a],
        npts=npts,
        bound=1.0,
        solver=solver,
    )
    if not diagnostics["certificate_passed"]:
        raise RuntimeError("Exact-bound SDP certificate failed")
    coefficients = np.zeros(degree + 1)
    coefficients[1::2] = odd_coefficients
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.savez(
        cache_path,
        coefficients=coefficients,
        degree=degree,
        a=a,
        epsilon=epsilon,
        npts=npts,
        solver=solver.upper(),
    )
    print(f"Saved coefficients to {cache_path}")
    return coefficients


def make_plot(
    retracted: Optional[np.ndarray],
    sdp: np.ndarray,
    degree: int,
    a: float,
    epsilon: float,
    pdf_path: str,
    png_path: str,
    show_retraction: bool = True,
    solver: str = "SCS",
) -> None:
    x_full = np.linspace(-1.0, 1.0, 10001)
    x_fit = np.linspace(-a, a, 4001)
    target_fit = (1.0 - epsilon) * x_fit / a
    sdp_full = chebval(x_full, sdp)
    sdp_fit = chebval(x_fit, sdp)
    if show_retraction:
        if retracted is None:
            raise ValueError("Retracted coefficients are required when show_retraction=True")
        retracted_full = chebval(x_full, retracted)
        retracted_fit = chebval(x_fit, retracted)

    plt.rcParams.update({"font.family": "serif", "font.size": 15})
    fig, (ax, error_ax) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    ax.axvspan(-a, a, color="0.92", zorder=0, label="Fitting interval")
    target_factor = 1.0 - epsilon
    target_label = (
        r"Target $x/a$"
        if abs(target_factor - 1.0) < 1.0e-14
        else rf"Target ${target_factor:g}x/a$"
    )
    ax.plot(
        x_fit,
        target_fit,
        color="black",
        linewidth=2.4,
        zorder=5,
        label=target_label,
    )
    if show_retraction:
        ax.plot(
            x_full,
            retracted_full,
            color="#0072B2",
            linewidth=1.1,
            linestyle="--",
            label="Fig. 7 retracted polynomial (old method)",
        )
    ax.plot(
        x_full,
        sdp_full,
        color="#D55E00",
        linewidth=1.1,
        linestyle="-",
        alpha=0.85,
        label=f"New exact-bound SDP polynomial ({solver.upper()})",
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("Function value")
    title = (
        f"Degree {degree}: old retraction vs new SDP"
        if show_retraction
        else f"Degree {degree}: new SDP polynomial"
    )
    ax.set_title(title)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.08, 1.08)
    ax.grid(True, alpha=0.25)
    ax.legend(
        loc="upper center",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        fontsize=9.5,
    )

    if show_retraction:
        error_ax.plot(
            x_fit,
            retracted_fit - target_fit,
            color="#0072B2",
            linewidth=1.5,
            label="Fig. 7 retracted polynomial (old method)",
        )
    error_ax.plot(
        x_fit,
        sdp_fit - target_fit,
        color="#D55E00",
        linewidth=1.5,
        label="New exact-bound SDP polynomial",
    )
    error_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    error_ax.set_xlabel(r"$x$")
    error_ax.set_ylabel("Fit minus target")
    maximum_sdp_error = float(np.max(np.abs(sdp_fit - target_fit)))
    error_ax.set_title(
        "Pointwise error\n" + rf"max $|p-f|={maximum_sdp_error:.2e}$"
    )
    error_ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    error_ax.grid(True, alpha=0.25)

    fig.tight_layout()
    for path in (pdf_path, png_path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig7-csv", default=DEFAULT_FIG7_CSV)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--pdf", default=DEFAULT_PDF)
    parser.add_argument("--png", default=DEFAULT_PNG)
    parser.add_argument("--degree", type=int, default=181)
    parser.add_argument("--a", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--npts", type=int, default=500)
    parser.add_argument("--solver", default="SCS")
    parser.add_argument(
        "--without-retraction",
        action="store_true",
        help="Plot only the target and new SDP polynomial.",
    )
    args = parser.parse_args()

    retracted = (
        load_retracted_coefficients(args.fig7_csv, args.degree)
        if not args.without_retraction
        else None
    )
    sdp = load_or_solve_sdp_coefficients(
        args.cache, args.degree, args.a, args.epsilon, args.npts, args.solver
    )
    make_plot(
        retracted,
        sdp,
        args.degree,
        args.a,
        args.epsilon,
        args.pdf,
        args.png,
        show_retraction=not args.without_retraction,
        solver=args.solver,
    )


if __name__ == "__main__":
    main()
