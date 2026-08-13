#!/usr/bin/env python3
"""Generate and plot degree scaling for the exact-bound SDP approximation."""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from numpy.polynomial.chebyshev import chebder, chebroots, chebval

from constrainapprox import CERTIFICATE_TOL, solve_odd_bounded_approx


DEFAULT_CSV = "data/degree_scaling_uniform_sv_amp_constrainapprox.csv"
DEFAULT_FIGURE = "figures/degree_scaling_uniform_sv_amp_constrainapprox.pdf"
INDEPENDENT_CHECK_POINTS = 1_000_001
BOUND_TOL = 5.0e-12
FIGURE_7_DEGREES = np.asarray([32, 46, 64, 92, 128, 182, 256, 364, 512], dtype=float)
FIGURE_7_EXPONENTS = np.arange(5.0, 9.0 + 0.5, 0.5)
FIGURE_7_XLIM = (27.857618025475972, 588.1335577584822)
FIGURE_7_YLIM = (0.0026150594886240014, 0.039514915470104515)
FIGURE_7_YTICKS = np.asarray([0.005, 0.01, 0.02])


def _figure_7_log_tick_label(value: float, _position=None) -> str:
    exponent = int(np.floor(np.log10(value) + 1.0e-12))
    multiplier = int(round(value * 10 ** (-exponent)))
    if multiplier == 1:
        return rf"$10^{{{exponent}}}$"
    return rf"${multiplier}\times 10^{{{exponent}}}$"


def odd_degree_grid(exponents: Sequence[float]) -> list[int]:
    """Return the largest odd integer no greater than each 2**exponent."""
    degrees = []
    for exponent in exponents:
        degree = int(np.floor(2.0**float(exponent)))
        degrees.append(degree if degree % 2 else degree - 1)
    if len(set(degrees)) != len(degrees):
        raise ValueError("exponents produce duplicate odd degrees")
    return degrees


def _full_coefficients(coefficients: np.ndarray, degree: int) -> np.ndarray:
    full_coefficients = np.zeros(degree + 1)
    full_coefficients[1::2] = coefficients
    return full_coefficients


def _max_abs_at_endpoints_and_critical_points(
    coefficients: np.ndarray, left: float, right: float
) -> float:
    roots = chebroots(chebder(coefficients))
    real_roots = roots[np.abs(roots.imag) <= 1.0e-7].real
    candidates = np.concatenate(
        ([left, right], real_roots[(real_roots >= left) & (real_roots <= right)])
    )
    return float(np.max(np.abs(chebval(candidates, coefficients))))


def independent_bound_check(coefficients: np.ndarray, degree: int) -> tuple[float, float]:
    """Recheck |F| using both a denser grid and all numerical critical points."""
    full_coefficients = _full_coefficients(coefficients, degree)
    theta = np.linspace(0.0, np.pi, INDEPENDENT_CHECK_POINTS)
    dense_maximum = float(np.max(np.abs(chebval(np.cos(theta), full_coefficients))))
    critical_maximum = _max_abs_at_endpoints_and_critical_points(
        full_coefficients, -1.0, 1.0
    )
    return max(dense_maximum, critical_maximum), dense_maximum


def independent_error_check(
    coefficients: np.ndarray, degree: int, a: float, epsilon: float
) -> tuple[float, float]:
    """Compute max fit error using a dense grid and all error critical points."""
    error_coefficients = _full_coefficients(coefficients, degree)
    error_coefficients[1] -= (1.0 - epsilon) / a
    dense_grid = np.linspace(0.0, a, INDEPENDENT_CHECK_POINTS)
    dense_maximum = float(np.max(np.abs(chebval(dense_grid, error_coefficients))))
    critical_maximum = _max_abs_at_endpoints_and_critical_points(
        error_coefficients, 0.0, a
    )
    return max(dense_maximum, critical_maximum), dense_maximum


def generate_data(
    csv_path: str,
    exponents: Sequence[float],
    npts: int,
    a: float,
    epsilon: float,
    solver: str,
    force: bool = False,
) -> None:
    degrees = odd_degree_grid(exponents)
    target = lambda x: (1.0 - epsilon) * x / a
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    existing = pd.DataFrame()
    if os.path.exists(csv_path) and not force:
        existing = pd.read_csv(csv_path)
    completed = set(existing.get("degree", pd.Series(dtype=int)).astype(int))
    mode = "w" if force or not os.path.exists(csv_path) else "a"
    fields = [
        "degree",
        "exp2",
        "npts",
        "a",
        "epsilon",
        "solver",
        "runtime_seconds",
        "max_error",
        "dense_max_error",
        "grid_objective_value",
        "solver_objective_value",
        "global_max_abs",
        "independent_global_max_abs",
        "independent_dense_max_abs",
        "diagonal_sum_residual",
        "minimum_gram_eigenvalue",
        "restoration_applied",
        "restoration_scale",
        "certificate_passed",
        "constraints_satisfied",
    ]

    with open(csv_path, mode, newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        for exponent, degree in zip(exponents, degrees):
            if degree in completed:
                print(f"Skipping completed degree {degree}")
                continue
            print(f"Solving degree {degree} (near 2^{exponent:g})")
            started = time.perf_counter()
            coefficients, diagnostics = solve_odd_bounded_approx(
                target,
                degree=degree,
                fit_intervals=[0.0, a],
                npts=npts,
                bound=1.0,
                solver=solver,
            )
            runtime = time.perf_counter() - started
            independent_maximum, independent_dense_maximum = independent_bound_check(
                coefficients, degree
            )
            maximum_error, dense_maximum_error = independent_error_check(
                coefficients, degree, a, epsilon
            )
            constraints_satisfied = bool(
                diagnostics["certificate_passed"]
                and diagnostics["diagonal_sum_residual"] <= CERTIFICATE_TOL
                and diagnostics["minimum_gram_eigenvalue"] >= 0.0
                and diagnostics["global_max_abs"] <= 1.0
                and independent_maximum <= 1.0 + BOUND_TOL
            )
            if not constraints_satisfied:
                raise RuntimeError(
                    f"independent constraint verification failed for degree {degree}: "
                    f"global={diagnostics['global_max_abs']:.17g}, "
                    f"independent={independent_maximum:.17g}, "
                    f"residual={diagnostics['diagonal_sum_residual']:.3e}, "
                    f"lambda_min={diagnostics['minimum_gram_eigenvalue']:.3e}"
                )
            writer.writerow(
                {
                    "degree": degree,
                    "exp2": exponent,
                    "npts": npts,
                    "a": a,
                    "epsilon": epsilon,
                    "solver": solver.upper(),
                    "runtime_seconds": runtime,
                    "max_error": maximum_error,
                    "dense_max_error": dense_maximum_error,
                    "grid_objective_value": diagnostics["objective_value"],
                    "solver_objective_value": diagnostics["solver_objective_value"],
                    "global_max_abs": diagnostics["global_max_abs"],
                    "independent_global_max_abs": independent_maximum,
                    "independent_dense_max_abs": independent_dense_maximum,
                    "diagonal_sum_residual": diagnostics["diagonal_sum_residual"],
                    "minimum_gram_eigenvalue": diagnostics["minimum_gram_eigenvalue"],
                    "restoration_applied": diagnostics["restoration_applied"],
                    "restoration_scale": diagnostics["restoration_scale"],
                    "certificate_passed": diagnostics["certificate_passed"],
                    "constraints_satisfied": constraints_satisfied,
                }
            )
            output.flush()
            print(
                f"  error={maximum_error:.6e}, "
                f"runtime={runtime:.2f}s, max|F|={independent_maximum:.12f}"
            )
    print(f"Data written to: {csv_path}")


def _as_bool(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().map({"true": True, "false": False})


def plot_data(csv_path: str, figure_path: str) -> None:
    data = pd.read_csv(csv_path).sort_values("degree")
    required = {"degree", "exp2", "max_error", "runtime_seconds", "constraints_satisfied"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    verified = _as_bool(data["constraints_satisfied"])
    if verified.isna().any() or not verified.all():
        bad = data.loc[~verified.fillna(False), "degree"].tolist()
        raise RuntimeError(f"refusing to plot unverified degree rows: {bad}")

    degrees = data["degree"].to_numpy(float)
    errors = data["max_error"].to_numpy(float)
    if np.any(errors <= 0.0):
        raise ValueError("max_error must be positive for a log-log plot")

    plt.rcParams.update({"font.family": "serif", "font.size": 20})
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        degrees,
        errors,
        color="#0072B2",
        linestyle="--",
        marker="o",
        markersize=9,
        markerfacecolor="none",
        markeredgewidth=1.5,
        label="Exact-bound SDP approximation",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree", fontsize=22)
    ax.set_ylabel("Maximum error vs target", fontsize=22)
    ax.set_xlim(FIGURE_7_XLIM)
    ax.set_ylim(FIGURE_7_YLIM)
    ax.xaxis.set_major_locator(FixedLocator(FIGURE_7_DEGREES))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([rf"$2^{{{value:g}}}$" for value in FIGURE_7_EXPONENTS])
    ax.yaxis.set_major_locator(FixedLocator(FIGURE_7_YTICKS))
    ax.yaxis.set_major_formatter(FuncFormatter(_figure_7_log_tick_label))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(figure_path)), exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {figure_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--output", default=DEFAULT_FIGURE)
    parser.add_argument("--npts", type=int, default=500)
    parser.add_argument("--a", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--solver", default="SCS")
    parser.add_argument("--min-exp", type=float, default=5.0)
    parser.add_argument("--max-exp", type=float, default=9.0)
    parser.add_argument("--step-exp", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if not (0.0 < args.a < 1.0):
        raise ValueError("--a must lie in (0, 1)")
    if not (0.0 <= args.epsilon < 1.0):
        raise ValueError("--epsilon must lie in [0, 1)")
    exponents = np.arange(args.min_exp, args.max_exp + 0.5 * args.step_exp, args.step_exp)
    if not args.plot_only:
        generate_data(
            args.csv, exponents, args.npts, args.a, args.epsilon, args.solver, args.force
        )
    plot_data(args.csv, args.output)


if __name__ == "__main__":
    main()
