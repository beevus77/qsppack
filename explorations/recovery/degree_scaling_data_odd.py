#!/usr/bin/env python3
"""Regenerate Figure 7 with odd parity for the odd amplification target."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import cvxpy as cp
import numpy as np
import pandas as pd
from numpy.polynomial import chebyshev as cheb
from scipy.fft import dct
from scipy.special import eval_chebyt

from degree_scaling_data import build_target_and_domain, run_single_degree
from qsppack.utils import get_entry


DEFAULT_CSV = "data/degree_scaling_uniform_sv_amp_npts19_solver2_odd.csv"
DEFAULT_DEGREES = (31, 45, 63, 91, 127, 181, 255, 363, 511)
DEFAULT_EXPONENTS = tuple(np.arange(5.0, 9.0 + 0.5, 0.5))
FIELDS = (
    "degree",
    "exp2",
    "parity",
    "npts",
    "N_weiss",
    "fit_solver",
    "time_fit",
    "time_qsp",
    "max_error_poly",
    "max_error_qsp",
    "max_error_poly_uniform_1000",
    "max_error_qsp_uniform_1000",
    "constraint_violated",
    "constraint_max_abs",
    "constraint_margin",
    "qsp_constraint_max_abs",
    "qsp_reconstruction_residual",
    "coef",
    "coef_full",
    "phi_proc",
)


def _clarabel_fit_with_explicit_tolerances(
    target,
    degree: int,
    intervals: list[float],
    npts: int,
    epsilon: float,
) -> tuple[np.ndarray, float]:
    """Solve the full sampled LP by CLARABEL constraint generation."""
    reference = np.cos(np.pi * np.arange(2 * npts) / (2 * npts - 1))
    grid = np.union1d(reference, intervals)
    grid = grid[grid >= 0.0]
    fit_mask = np.zeros(grid.size, dtype=bool)
    for left, right in np.asarray(intervals).reshape(-1, 2):
        fit_mask |= (grid >= left) & (grid <= right)

    fit_indices = np.flatnonzero(fit_mask)
    orders = np.arange(1, degree + 1, 2)
    max_active = 4096
    active_bound = set(
        np.linspace(0, grid.size - 1, min(max_active, grid.size), dtype=int)
    )
    active_fit = set(
        fit_indices[
            np.linspace(0, fit_indices.size - 1, min(max_active, fit_indices.size), dtype=int)
        ]
    )

    def extrema_indices(values: np.ndarray, indices: np.ndarray, count: int = 64) -> set[int]:
        if values.size <= 2:
            return set(indices.tolist())
        local = np.flatnonzero(
            (values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:])
        ) + 1
        candidates = np.unique(np.concatenate(([0, values.size - 1], local)))
        selected = candidates[np.argsort(values[candidates])[-count:]]
        return set(indices[selected].tolist())

    started = time.perf_counter()
    solution = None
    for exchange_iteration in range(20):
        bound_indices = np.asarray(sorted(active_bound), dtype=int)
        objective_indices = np.asarray(sorted(active_fit), dtype=int)
        bound_design = np.column_stack(
            [eval_chebyt(int(order), grid[bound_indices]) for order in orders]
        )
        objective_design = np.column_stack(
            [eval_chebyt(int(order), grid[objective_indices]) for order in orders]
        )
        coefficients = cp.Variable(orders.size)
        error_bound = cp.Variable(nonneg=True)
        bound_values = bound_design @ coefficients
        fit_residual = objective_design @ coefficients - target(grid[objective_indices])
        problem = cp.Problem(
            cp.Minimize(error_bound),
            [
                fit_residual <= error_bound,
                fit_residual >= -error_bound,
                bound_values <= 1.0 - epsilon,
                bound_values >= -(1.0 - epsilon),
            ],
        )
        problem.solve(
            solver="CLARABEL",
            tol_gap_abs=1.0e-8,
            tol_gap_rel=1.0e-8,
            tol_feas=1.0e-8,
            max_iter=500,
            verbose=False,
        )
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"CLARABEL fit failed with status {problem.status!r}")
        if coefficients.value is None:
            raise RuntimeError("CLARABEL returned no coefficients")

        solution = np.asarray(coefficients.value, dtype=float)
        full = np.zeros(degree + 1)
        full[1::2] = solution
        full_values = cheb.chebval(grid, full)
        fit_errors = np.abs(full_values[fit_indices] - target(grid[fit_indices]))
        bound_abs = np.abs(full_values)
        max_fit_error = float(np.max(fit_errors))
        max_bound_abs = float(np.max(bound_abs))
        objective_value = float(error_bound.value)
        print(
            f"CLARABEL exchange={exchange_iteration + 1}, status={problem.status}, "
            f"active={len(active_bound) + len(active_fit)}, "
            f"objective={objective_value:.12e}, full_error={max_fit_error:.12e}, "
            f"full_bound={max_bound_abs:.12e}"
        )
        if (
            max_fit_error <= objective_value + 2.0e-8
            and max_bound_abs <= 1.0 - epsilon + 2.0e-8
        ):
            break
        active_fit.update(extrema_indices(fit_errors, fit_indices))
        active_bound.update(extrema_indices(bound_abs, np.arange(grid.size)))
    else:
        raise RuntimeError("CLARABEL constraint generation did not converge")

    elapsed = time.perf_counter() - started
    if solution is None:
        raise RuntimeError("CLARABEL constraint generation returned no solution")
    full = np.zeros(degree + 1)
    full[1::2] = solution
    return full, elapsed


def _critical_max_abs(coefficients: np.ndarray, left: float, right: float) -> float:
    derivative_roots = cheb.chebroots(cheb.chebder(coefficients))
    real_roots = derivative_roots[np.abs(derivative_roots.imag) <= 1.0e-7].real
    candidates = np.concatenate(
        ([left, right], real_roots[(real_roots >= left) & (real_roots <= right)])
    )
    return float(np.max(np.abs(cheb.chebval(candidates, coefficients))))


def _qsp_chebyshev_coefficients(
    degree: int, phi_proc: np.ndarray, parity: int
) -> tuple[np.ndarray, float]:
    """Recover the QSP response polynomial from degree+1 Lobatto values."""
    theta = np.pi * np.arange(degree + 1) / degree
    nodes = np.cos(theta)
    out = {"targetPre": parity == 0, "parity": parity, "typePhi": "full"}
    values = np.asarray(get_entry(nodes, np.array(phi_proc, copy=True), out))
    imaginary_max = float(np.max(np.abs(values.imag)))
    if imaginary_max > 1.0e-9:
        raise RuntimeError(f"QSP response has max imaginary part {imaginary_max:.3e}")

    coefficients = dct(values.real, type=1) / degree
    coefficients[[0, -1]] *= 0.5
    coefficients[np.arange(degree + 1) % 2 != parity] = 0.0

    check_x = np.linspace(-1.0, 1.0, 2001)
    reconstructed = cheb.chebval(check_x, coefficients)
    direct = np.asarray(get_entry(check_x, np.array(phi_proc, copy=True), out)).real
    residual = float(np.max(np.abs(reconstructed - direct)))
    if residual > 1.0e-8:
        raise RuntimeError(f"QSP polynomial reconstruction residual is {residual:.3e}")
    return coefficients, residual


def _critical_error(
    coefficients: np.ndarray,
    target_slope: float,
    left: float,
    right: float,
) -> float:
    error_coefficients = np.asarray(coefficients, dtype=float).copy()
    error_coefficients[1] -= target_slope
    return _critical_max_abs(error_coefficients, left, right)


def generate(
    csv_path: str,
    npts: int,
    a: float,
    epsilon: float,
    N_weiss: int,
    force: bool,
    max_exp: float,
    fit_solver: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    existing = pd.DataFrame()
    if os.path.exists(csv_path) and not force:
        existing = pd.read_csv(csv_path)
    completed = set(existing.get("degree", pd.Series(dtype=int)).astype(int))
    mode = "w" if force or not os.path.exists(csv_path) else "a"

    target, intervals, x_lo, x_hi = build_target_and_domain(
        "uniform_sv_amp", a, epsilon
    )
    target_slope = (1.0 - epsilon) / a

    with open(csv_path, mode, newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        if mode == "w":
            writer.writeheader()
        for degree, exponent in zip(DEFAULT_DEGREES, DEFAULT_EXPONENTS):
            if exponent > max_exp:
                continue
            if degree in completed:
                print(f"Skipping completed degree {degree}")
                continue
            print(f"Solving odd degree {degree} (2^{exponent:g})")
            coefficient_override = None
            fit_time_override = None
            if fit_solver.upper() == "CLARABEL":
                coefficient_override, fit_time_override = (
                    _clarabel_fit_with_explicit_tolerances(
                        target, degree, intervals, npts, epsilon
                    )
                )
            result = run_single_degree(
                degree,
                target=target,
                intervals=intervals,
                x_lo=x_lo,
                x_hi=x_hi,
                npts=npts,
                epsil=epsilon,
                N_weiss=N_weiss,
                cvx_solver=fit_solver,
                coef_full_override=coefficient_override,
                time_fit_override=fit_time_override,
            )
            coefficients = np.asarray(result["coef_full"], dtype=float)
            phi_proc = np.asarray(result["phi_proc"])
            qsp_coefficients, reconstruction_residual = _qsp_chebyshev_coefficients(
                degree, phi_proc, parity=1
            )
            max_error_poly = _critical_error(
                coefficients, target_slope, x_lo, x_hi
            )
            max_error_qsp = _critical_error(
                qsp_coefficients, target_slope, x_lo, x_hi
            )
            constraint_max_abs = _critical_max_abs(coefficients, -1.0, 1.0)
            constraint_margin = 1.0 - constraint_max_abs
            constraint_violated = constraint_margin < 0.0
            qsp_constraint_max_abs = _critical_max_abs(
                qsp_coefficients, -1.0, 1.0
            )

            phi_imaginary_max = float(np.max(np.abs(phi_proc.imag)))
            if phi_imaginary_max > 1.0e-10:
                raise RuntimeError(
                    f"phi_proc has max imaginary part {phi_imaginary_max:.3e}"
                )
            writer.writerow(
                {
                    "degree": degree,
                    "exp2": exponent,
                    "parity": 1,
                    "npts": npts,
                    "N_weiss": N_weiss,
                    "fit_solver": fit_solver.upper(),
                    "time_fit": result["time_fit"],
                    "time_qsp": result["time_qsp"],
                    "max_error_poly": max_error_poly,
                    "max_error_qsp": max_error_qsp,
                    "max_error_poly_uniform_1000": result["max_error_poly"],
                    "max_error_qsp_uniform_1000": result["max_error_qsp"],
                    "constraint_violated": constraint_violated,
                    "constraint_max_abs": constraint_max_abs,
                    "constraint_margin": constraint_margin,
                    "qsp_constraint_max_abs": qsp_constraint_max_abs,
                    "qsp_reconstruction_residual": reconstruction_residual,
                    "coef": json.dumps(np.asarray(result["coef"]).tolist()),
                    "coef_full": json.dumps(coefficients.tolist()),
                    "phi_proc": json.dumps(phi_proc.real.tolist()),
                }
            )
            output.flush()
            print(
                f"  errors: polynomial={max_error_poly:.6e}, "
                f"QSP={max_error_qsp:.6e}; max|p|={constraint_max_abs:.12f}"
            )
    print(f"Data written to: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--npts", type=int, default=2**19)
    parser.add_argument("--a", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--N", type=int, default=2**14)
    parser.add_argument(
        "--solver",
        choices=("OSQP", "CLARABEL"),
        default="OSQP",
        help="CVXPY solver used for the sampled polynomial fit.",
    )
    parser.add_argument("--max-exp", type=float, default=9.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    generate(
        args.csv,
        args.npts,
        args.a,
        args.epsilon,
        args.N,
        args.force,
        args.max_exp,
        args.solver,
    )


if __name__ == "__main__":
    main()
