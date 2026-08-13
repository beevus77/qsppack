"""Odd QSP approximation with an exact semidefinite bound certificate."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from typing import Any

import cvxpy as cp
import numpy as np
from numpy.polynomial.chebyshev import chebder, chebroots, chebval, chebvander


Array = np.ndarray
Target = Callable[[Array], Array]
CERTIFICATE_TOL = 1.0e-5
DENSE_NPTS = 200_001
STRICT_BOUND_BUFFER = 1.0e-8
PSD_FLOOR = 1.0e-8


def _normalize_intervals(
    fit_intervals: Sequence[float] | Sequence[Sequence[float]],
) -> list[tuple[float, float]]:
    values = np.asarray(fit_intervals, dtype=float)
    if values.ndim == 1:
        if values.size == 0 or values.size % 2:
            raise ValueError("fit_intervals must contain endpoint pairs")
        values = values.reshape(-1, 2)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("fit_intervals must have shape (n_intervals, 2)")

    intervals = sorted((float(left), float(right)) for left, right in values)
    previous_right = -np.inf
    for left, right in intervals:
        if not (0.0 <= left < right <= 1.0):
            raise ValueError("each fit interval must satisfy 0 <= left < right <= 1")
        if left < previous_right:
            raise ValueError("fit intervals must not overlap")
        previous_right = right
    return intervals


def _validate_inputs(
    degree: int,
    fit_intervals: Sequence[float] | Sequence[Sequence[float]],
    npts: int,
    bound: float,
    solver: str,
) -> list[tuple[float, float]]:
    if not isinstance(degree, (int, np.integer)) or degree < 1 or degree % 2 == 0:
        raise ValueError("degree must be a positive odd integer")
    if not isinstance(npts, (int, np.integer)) or npts < 2:
        raise ValueError("npts must be an integer greater than one")
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError("bound must be positive and finite")
    if solver.upper() not in cp.installed_solvers():
        raise ValueError(f"CVXPY solver {solver!r} is not installed")
    return _normalize_intervals(fit_intervals)


def _evaluate_target(target: Target, points: Array) -> Array:
    try:
        values = np.asarray(target(points), dtype=float)
        if values.shape == ():
            values = np.full(points.shape, float(values))
        else:
            values = np.broadcast_to(values, points.shape).astype(float, copy=False)
    except (TypeError, ValueError):
        values = np.asarray([target(float(point)) for point in points], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("target returned a nonfinite value on the fit grid")
    return values


def _problem_data(
    target: Target,
    degree: int,
    intervals: Sequence[tuple[float, float]],
    npts: int,
) -> tuple[Array, Array, Array, Array]:
    reference = np.cos(np.pi * np.arange(2 * npts) / (2 * npts - 1))
    endpoints = np.asarray(intervals, dtype=float).ravel()
    grid = np.union1d(reference, endpoints)
    grid = grid[(grid >= 0.0) & (grid <= 1.0)]

    fit_mask = np.zeros(grid.size, dtype=bool)
    for left, right in intervals:
        fit_mask |= (grid >= left) & (grid <= right)
    fit_indices = np.flatnonzero(fit_mask)
    if fit_indices.size == 0:
        raise ValueError("the fit grid is empty")

    design = chebvander(grid, degree)[:, 1::2]
    target_values = np.zeros(grid.size)
    target_values[fit_indices] = _evaluate_target(target, grid[fit_indices])
    return grid, fit_indices, design, target_values


def _solver_options(solver: str) -> dict[str, Any]:
    if solver.upper() == "SCS":
        return {
            "eps": 1.0e-6,
            "max_iters": 200_000,
            "acceleration_lookback": 10,
            "verbose": False,
        }
    if solver.upper() == "CLARABEL":
        return {
            "tol_gap_abs": 1.0e-7,
            "tol_gap_rel": 1.0e-7,
            "tol_feas": 1.0e-7,
            "max_iter": 500,
            "verbose": False,
        }
    return {"verbose": False}


def _full_odd_coefficients(partial_coefficients: Array, degree: int) -> Array:
    coefficients = np.zeros(degree + 1)
    coefficients[1::2] = partial_coefficients
    return coefficients


def _global_max_abs(partial_coefficients: Array, degree: int) -> tuple[float, float]:
    full_coefficients = _full_odd_coefficients(partial_coefficients, degree)
    dense_grid = np.cos(np.linspace(0.0, np.pi, DENSE_NPTS))
    dense_values = chebval(dense_grid, full_coefficients)
    dense_maximum = float(np.max(np.abs(dense_values)))

    derivative_roots = chebroots(chebder(full_coefficients))
    real_roots = derivative_roots[np.abs(derivative_roots.imag) <= 1.0e-7].real
    real_roots = real_roots[(real_roots >= -1.0) & (real_roots <= 1.0)]
    candidates = np.concatenate((np.asarray([-1.0, 1.0]), real_roots))
    critical_maximum = float(np.max(np.abs(chebval(candidates, full_coefficients))))
    return max(dense_maximum, critical_maximum), dense_maximum


def _restore_strict_feasibility(
    coefficients: Array,
    gram: Array,
    degree: int,
    bound: float,
) -> tuple[Array, Array, dict[str, Any]]:
    symmetric_gram = 0.5 * (gram + gram.T)
    global_before, dense_before = _global_max_abs(coefficients, degree)
    minimum_eigenvalue_before = float(np.linalg.eigvalsh(symmetric_gram)[0])

    scale = 1.0
    if global_before >= bound - STRICT_BOUND_BUFFER:
        scale = min(scale, (bound - STRICT_BOUND_BUFFER) / global_before)

    constant_eigenvalue = bound / (degree + 1)
    if minimum_eigenvalue_before < PSD_FLOOR:
        psd_scale = (constant_eigenvalue - PSD_FLOOR) / (
            constant_eigenvalue - minimum_eigenvalue_before
        )
        scale = min(scale, max(0.0, min(1.0, psd_scale)))

    restored_coefficients = scale * coefficients
    restored_gram = scale * symmetric_gram
    restored_gram += (1.0 - scale) * constant_eigenvalue * np.eye(degree + 1)
    return restored_coefficients, restored_gram, {
        "restoration_applied": bool(scale < 1.0),
        "restoration_scale": float(scale),
        "pre_restoration_global_max_abs": global_before,
        "pre_restoration_dense_max_abs": dense_before,
        "pre_restoration_minimum_gram_eigenvalue": minimum_eigenvalue_before,
    }


def _certificate_diagnostics(
    coefficients: Array,
    gram: Array,
    degree: int,
    bound: float,
) -> tuple[float, float, float, float]:
    expected = np.zeros(degree + 1)
    expected[0] = bound
    expected[1::2] = -0.5 * coefficients
    observed = np.asarray(
        [np.trace(gram) if k == 0 else np.diag(gram, k=k).sum() for k in range(degree + 1)]
    )
    diagonal_residual = float(np.max(np.abs(observed - expected)))
    symmetric_gram = 0.5 * (gram + gram.T)
    minimum_eigenvalue = float(np.linalg.eigvalsh(symmetric_gram)[0])
    global_maximum, dense_maximum = _global_max_abs(coefficients, degree)
    return diagonal_residual, minimum_eigenvalue, global_maximum, dense_maximum


def _base_diagnostics(
    problem: cp.Problem,
    solver: str,
    elapsed: float,
    grid: Array,
    fit_indices: Array,
    design: Array,
    target_values: Array,
    coefficients: Array,
) -> dict[str, Any]:
    fit_values = design[fit_indices] @ coefficients
    fit_error = fit_values - target_values[fit_indices]
    stats = problem.solver_stats
    return {
        "solver": solver.upper(),
        "status": problem.status,
        "objective_value": float(np.max(np.abs(fit_error))),
        "solver_objective_value": float(problem.value),
        "solve_time_seconds": float(elapsed),
        "solver_solve_time_seconds": None if stats.solve_time is None else float(stats.solve_time),
        "num_iters": None if stats.num_iters is None else int(stats.num_iters),
        "n_grid": int(grid.size),
        "n_fit": int(fit_indices.size),
        "grid_max_abs": float(np.max(np.abs(design @ coefficients))),
        "fit_value_max_abs": float(np.max(np.abs(fit_values))),
    }


def solve_odd_bounded_approx(
    target: Target,
    degree: int,
    fit_intervals: Sequence[float] | Sequence[Sequence[float]],
    npts: int = 500,
    bound: float = 1.0,
    solver: str = "SCS",
) -> tuple[Array, dict[str, Any]]:
    """Minimize the grid error subject to an exact continuous odd-polynomial bound."""
    intervals = _validate_inputs(degree, fit_intervals, npts, bound, solver)
    grid, fit_indices, design, target_values = _problem_data(target, degree, intervals, npts)
    n_coefficients = (degree + 1) // 2

    coefficients = cp.Variable(n_coefficients, name="c")
    values = cp.Variable(grid.size, name="f")
    gram = cp.Variable((degree + 1, degree + 1), symmetric=True, name="Q")

    constraints: list[cp.Constraint] = [values == design @ coefficients, gram >> 0]
    constraints.append(cp.trace(gram) == bound)
    for k in range(1, degree + 1):
        diagonal_sum = cp.sum(cp.diag(gram, k=k))
        if k % 2:
            constraints.append(diagonal_sum == -0.5 * coefficients[(k - 1) // 2])
        else:
            constraints.append(diagonal_sum == 0.0)

    objective = cp.Minimize(cp.norm(values[fit_indices] - target_values[fit_indices], "inf"))
    problem = cp.Problem(objective, constraints)
    started = time.perf_counter()
    try:
        problem.solve(solver=solver.upper(), **_solver_options(solver))
    except cp.error.SolverError as error:
        raise RuntimeError(f"SDP solver {solver!r} failed") from error
    elapsed = time.perf_counter() - started

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"SDP solve failed with status {problem.status!r}")
    if coefficients.value is None or gram.value is None:
        raise RuntimeError("SDP solver returned no numerical solution")

    coefficient_values, gram_value, restoration_diagnostics = _restore_strict_feasibility(
        np.asarray(coefficients.value, dtype=float),
        np.asarray(gram.value, dtype=float),
        degree,
        bound,
    )
    diagonal_residual, minimum_eigenvalue, global_maximum, dense_maximum = (
        _certificate_diagnostics(coefficient_values, gram_value, degree, bound)
    )
    diagnostics = _base_diagnostics(
        problem, solver, elapsed, grid, fit_indices, design, target_values, coefficient_values
    )
    diagnostics.update(
        {
            "degree": int(degree),
            "bound": float(bound),
            "diagonal_sum_residual": diagonal_residual,
            "minimum_gram_eigenvalue": minimum_eigenvalue,
            "global_max_abs": global_maximum,
            "dense_max_abs": dense_maximum,
            **restoration_diagnostics,
        }
    )
    certificate_passed = (
        diagonal_residual <= CERTIFICATE_TOL
        and minimum_eigenvalue >= 0.0
        and global_maximum <= bound
        and dense_maximum <= bound
    )
    diagnostics["certificate_passed"] = certificate_passed
    diagnostics["continuous_bound_satisfied"] = global_maximum <= bound
    diagnostics["role"] = "accepted_solution"
    if not certificate_passed:
        raise RuntimeError(f"numerical SDP certificate failed: {json.dumps(diagnostics, sort_keys=True)}")
    return coefficient_values, diagnostics


def solve_sampled_bounded_approx(
    target: Target,
    degree: int,
    fit_intervals: Sequence[float] | Sequence[Sequence[float]],
    npts: int = 500,
    bound: float = 1.0,
    solver: str = "SCS",
) -> tuple[Array, dict[str, Any]]:
    """Solve the original grid-bounded comparison problem with eta equal to zero."""
    intervals = _validate_inputs(degree, fit_intervals, npts, bound, solver)
    grid, fit_indices, design, target_values = _problem_data(target, degree, intervals, npts)
    coefficients = cp.Variable((degree + 1) // 2, name="c_sampled")
    values = cp.Variable(grid.size, name="f_sampled")
    constraints = [values == design @ coefficients, values >= -bound, values <= bound]
    objective = cp.Minimize(cp.norm(values[fit_indices] - target_values[fit_indices], "inf"))
    problem = cp.Problem(objective, constraints)
    started = time.perf_counter()
    try:
        problem.solve(solver=solver.upper(), **_solver_options(solver))
    except cp.error.SolverError as error:
        raise RuntimeError(f"sampled solver {solver!r} failed") from error
    elapsed = time.perf_counter() - started

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"sampled solve failed with status {problem.status!r}")
    if coefficients.value is None:
        raise RuntimeError("sampled solver returned no numerical solution")

    coefficient_values = np.asarray(coefficients.value, dtype=float)
    diagnostics = _base_diagnostics(
        problem, solver, elapsed, grid, fit_indices, design, target_values, coefficient_values
    )
    global_maximum, dense_maximum = _global_max_abs(coefficient_values, degree)
    diagnostics.update(
        {
            "degree": int(degree),
            "bound": float(bound),
            "global_max_abs": global_maximum,
            "dense_max_abs": dense_maximum,
            "continuous_bound_satisfied": global_maximum <= bound,
            "role": "diagnostic_baseline_only",
        }
    )
    return coefficient_values, diagnostics


def run_small_degree_checks(solver: str = "SCS") -> dict[str, Any]:
    """Run deterministic validation checks before the degree-101 benchmark."""
    target = lambda x: 0.5 * x
    coefficients, diagnostics = solve_odd_bounded_approx(
        target, degree=5, fit_intervals=[0.0, 0.5], npts=80, solver=solver
    )
    if abs(coefficients[0] - 0.5) > 1.0e-4 or np.linalg.norm(coefficients[1:], np.inf) > 1.0e-4:
        raise RuntimeError("small-degree exact-target recovery failed")

    test_coefficients = np.asarray([1.0 + 1.0e-7])
    test_gram = np.asarray(
        [[0.5, -0.5 * test_coefficients[0]], [-0.5 * test_coefficients[0], 0.5]]
    )
    restored_coefficients, restored_gram, restoration = _restore_strict_feasibility(
        test_coefficients, test_gram, degree=1, bound=1.0
    )
    restoration_residual, restoration_eigenvalue, restoration_maximum, _ = (
        _certificate_diagnostics(restored_coefficients, restored_gram, degree=1, bound=1.0)
    )
    if not (
        restoration["restoration_applied"]
        and restoration_residual <= CERTIFICATE_TOL
        and restoration_eigenvalue >= 0.0
        and restoration_maximum <= 1.0
    ):
        raise RuntimeError("strict-feasibility restoration check failed")

    rejected: list[str] = []
    for label, kwargs in (
        ("even degree", {"degree": 4, "fit_intervals": [0.0, 0.5]}),
        ("malformed interval", {"degree": 5, "fit_intervals": [0.5, 0.0]}),
        ("missing solver", {"degree": 5, "fit_intervals": [0.0, 0.5], "solver": "MISSING"}),
    ):
        try:
            solve_odd_bounded_approx(target, npts=20, **kwargs)
        except ValueError:
            rejected.append(label)
        else:
            raise RuntimeError(f"validation did not reject {label}")
    return {
        "rejected_inputs": rejected,
        "diagnostics": diagnostics,
        "restoration_check": {
            "restoration_scale": restoration["restoration_scale"],
            "diagonal_sum_residual": restoration_residual,
            "minimum_gram_eigenvalue": restoration_eigenvalue,
            "global_max_abs": restoration_maximum,
        },
    }


def run_uniform_singular_value_amplification(
    degree: int = 101,
    npts: int = 500,
    solver: str = "SCS",
) -> dict[str, Any]:
    """Run the buffered uniform singular value amplification comparison."""
    amplification_endpoint = 0.2
    delta = 0.01
    target = lambda x: 0.9 * x / amplification_endpoint
    fit_intervals = [0.0, amplification_endpoint - delta]
    sdp_coefficients, sdp_diagnostics = solve_odd_bounded_approx(
        target, degree, fit_intervals, npts=npts, bound=1.0, solver=solver
    )
    sampled_solver = "CLARABEL" if "CLARABEL" in cp.installed_solvers() else solver
    _, sampled_diagnostics = solve_sampled_bounded_approx(
        target, degree, fit_intervals, npts=npts, bound=1.0, solver=sampled_solver
    )
    return {
        "parameters": {
            "a": amplification_endpoint,
            "delta": delta,
            "degree": degree,
            "npts": npts,
            "fit_intervals": [fit_intervals],
            "target_scale": 0.9,
            "bound": 1.0,
        },
        "sdp": sdp_diagnostics,
        "sampled_baseline": sampled_diagnostics,
        "sdp_coefficients": sdp_coefficients.tolist(),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot encode object of type {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=101)
    parser.add_argument("--npts", type=int, default=500)
    parser.add_argument("--solver", default="SCS")
    parser.add_argument("--skip-small-checks", action="store_true")
    args = parser.parse_args()

    output: dict[str, Any] = {}
    if not args.skip_small_checks:
        output["small_degree_checks"] = run_small_degree_checks(args.solver)
    output["uniform_singular_value_amplification"] = run_uniform_singular_value_amplification(
        degree=args.degree, npts=args.npts, solver=args.solver
    )
    print(json.dumps(output, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
