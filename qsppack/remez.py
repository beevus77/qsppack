"""Constrained Remez approximation for QSP polynomials.

The public interface uses polynomial coordinates ``x`` on ``[0, 1]``.  The
active-set exchange iteration is performed internally in the angular variable
``omega = arccos(x)``, where Chebyshev polynomials become cosine functions.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np

from .utils import check_feasibility


ArrayFunction = Callable[[np.ndarray], np.ndarray]
Interval = Tuple[float, float]
Bound = Union[float, ArrayFunction]


class RemezConvergenceWarning(UserWarning):
    """Warning emitted when Remez returns a finite, unconverged iterate."""


class RemezConvergenceError(RuntimeError):
    """Raised when constrained Remez cannot produce an acceptable result."""


@dataclass
class RemezOptions:
    """Numerical controls for the constrained Remez iteration."""

    relative_tolerance: float = 1e-3
    max_iterations: int = 40
    fit_grid_size: int = 4001
    constraint_grid_size: int = 4001
    root_grid_size: int = 2001
    root_bisection_iterations: int = 60
    max_exchange_iterations: int = 100
    exchange_tolerance: float = 1e-9
    metrics_grid_size: int = 4000

    def validate(self) -> None:
        if self.relative_tolerance <= 0:
            raise ValueError("relative_tolerance must be positive.")
        for name in (
            "max_iterations",
            "fit_grid_size",
            "constraint_grid_size",
            "root_grid_size",
            "root_bisection_iterations",
            "max_exchange_iterations",
            "metrics_grid_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError("{} must be a positive integer.".format(name))
            setattr(self, name, int(value))
        if self.exchange_tolerance <= 0:
            raise ValueError("exchange_tolerance must be positive.")


@dataclass
class RemezMetrics:
    """Accuracy and feasibility diagnostics for one coefficient variant."""

    max_error: float
    max_work_error: float
    max_magnitude: float
    max_constraint_violation: float
    interval_errors: Tuple[float, ...] = field(default_factory=tuple)


@dataclass
class RemezResult:
    """Result of a constrained Remez approximation.

    All coefficient arrays use ascending Chebyshev order.  Full arrays have
    length ``degree + 1``; parity coefficient arrays contain only orders with
    the same parity as ``degree``.
    """

    degree: int
    coefficients: np.ndarray
    raw_coefficients: np.ndarray
    scaled_coefficients: np.ndarray
    parity_coefficients: np.ndarray
    raw_parity_coefficients: np.ndarray
    scaled_parity_coefficients: np.ndarray
    scale_factor: float
    ripple_amplitude: float
    raw_ripple_amplitude: float
    scaled_ripple_amplitude: float
    extremal_points: np.ndarray
    fit_intervals: Tuple[Interval, ...]
    work_intervals: Tuple[Interval, ...]
    metrics: RemezMetrics
    raw_metrics: RemezMetrics
    scaled_metrics: RemezMetrics
    rescaled: bool
    converged: bool
    outer_iterations: int
    exchange_iterations: int
    message: str

    @property
    def parity(self) -> int:
        """Polynomial parity: zero for even and one for odd."""

        return self.degree % 2

    def evaluate(self, x, variant: str = "selected"):
        """Evaluate the selected, raw, or scaled polynomial at ``x``."""

        if variant == "selected":
            coefficients = self.coefficients
        elif variant == "raw":
            coefficients = self.raw_coefficients
        elif variant == "scaled":
            coefficients = self.scaled_coefficients
        else:
            raise ValueError("variant must be 'selected', 'raw', or 'scaled'.")
        return np.polynomial.chebyshev.chebval(x, coefficients)


def _evaluate(function: ArrayFunction, points: np.ndarray) -> np.ndarray:
    values = np.asarray(function(points), dtype=float)
    if values.ndim == 0:
        values = np.full(points.shape, float(values), dtype=float)
    if values.shape != points.shape:
        try:
            values = np.broadcast_to(values, points.shape).astype(float)
        except ValueError as exc:
            raise ValueError("Callable output must broadcast to the input shape.") from exc
    return values


def _as_function(value: Bound) -> ArrayFunction:
    if callable(value):
        return value
    constant = float(value)
    return lambda x: np.full(np.asarray(x).shape, constant, dtype=float)


def _normalize_intervals(
    intervals: Sequence[Interval], domain: Interval = (0.0, 1.0)
) -> Tuple[Interval, ...]:
    if not intervals:
        return tuple()
    normalized = []
    for interval in intervals:
        if len(interval) != 2:
            raise ValueError("Each interval must contain exactly two endpoints.")
        left, right = float(interval[0]), float(interval[1])
        if not (np.isfinite(left) and np.isfinite(right)):
            raise ValueError("Interval endpoints must be finite.")
        if left > right:
            raise ValueError("Intervals must be ordered from left to right.")
        if left < domain[0] - 1e-14 or right > domain[1] + 1e-14:
            raise ValueError("Intervals must lie in [0, 1].")
        normalized.append((max(domain[0], left), min(domain[1], right)))
    normalized.sort()
    merged = [list(normalized[0])]
    for left, right in normalized[1:]:
        if left <= merged[-1][1] + 1e-14:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return tuple((float(left), float(right)) for left, right in merged)


def _is_subset(subset: Sequence[Interval], superset: Sequence[Interval]) -> bool:
    return all(
        any(left >= outer_left - 1e-14 and right <= outer_right + 1e-14
            for outer_left, outer_right in superset)
        for left, right in subset
    )


def _complement(intervals: Sequence[Interval], domain: Interval) -> Tuple[Interval, ...]:
    result = []
    cursor = domain[0]
    for left, right in intervals:
        if left > cursor + 1e-14:
            result.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < domain[1] - 1e-14:
        result.append((cursor, domain[1]))
    return tuple(result)


def _interval_grid(intervals: Sequence[Interval], size: int) -> np.ndarray:
    if not intervals:
        return np.array([], dtype=float)
    return np.unique(np.concatenate([
        np.linspace(left, right, max(2, int(size)), endpoint=True)
        for left, right in intervals
    ]))


def _interval_edges(intervals: Sequence[Interval]) -> np.ndarray:
    if not intervals:
        return np.array([], dtype=float)
    return np.unique(np.asarray([endpoint for interval in intervals for endpoint in interval]))


def _to_omega_intervals(intervals: Sequence[Interval]) -> Tuple[Interval, ...]:
    converted = [(float(np.arccos(right)), float(np.arccos(left))) for left, right in intervals]
    return tuple(sorted(converted))


def _full_coefficients(compact_descending: np.ndarray, orders: np.ndarray, degree: int) -> np.ndarray:
    coefficients = np.zeros(degree + 1, dtype=float)
    coefficients[orders.astype(int)] = compact_descending
    return coefficients


class ConstrainedRemezFitter:
    """Reusable constrained Remez fitter with an ``x``-space interface.

    Construct a fitter when several degrees will use the same target, fitting
    intervals, and numerical controls.  For a single fit, :func:`remez` is the
    shorter equivalent interface.
    """

    def __init__(
        self,
        target: ArrayFunction,
        fit_intervals: Sequence[Interval],
        work_intervals: Optional[Sequence[Interval]] = None,
        target_scale: float = 1.0,
        lower_bound: Bound = -1.0,
        upper_bound: Bound = 1.0,
        weight: Optional[ArrayFunction] = None,
        target_derivative: Optional[ArrayFunction] = None,
        weight_derivative: Optional[ArrayFunction] = None,
        options: Optional[RemezOptions] = None,
    ) -> None:
        if not callable(target):
            raise TypeError("target must be callable.")
        self.fit_intervals = _normalize_intervals(fit_intervals)
        if not self.fit_intervals:
            raise ValueError("fit_intervals must be nonempty.")
        self.work_intervals = (
            self.fit_intervals
            if work_intervals is None
            else _normalize_intervals(work_intervals)
        )
        if not self.work_intervals:
            raise ValueError("work_intervals must be nonempty.")
        if not _is_subset(self.work_intervals, self.fit_intervals):
            raise ValueError("work_intervals must be subsets of fit_intervals.")
        if not np.isfinite(target_scale) or not (0.0 < target_scale <= 1.0):
            raise ValueError("target_scale must lie in (0, 1].")

        self.target = target
        self.target_scale = float(target_scale)
        self.lower_bound = _as_function(lower_bound)
        self.upper_bound = _as_function(upper_bound)
        self.weight = weight or (lambda x: np.ones(np.asarray(x).shape, dtype=float))
        self.target_derivative = target_derivative
        self.weight_derivative = weight_derivative
        self.options = options or RemezOptions()
        self.options.validate()

        sample = _interval_grid(self.fit_intervals, 5)
        lower = _evaluate(self.lower_bound, sample)
        upper = _evaluate(self.upper_bound, sample)
        if np.any(lower >= upper):
            raise ValueError("lower_bound must be strictly below upper_bound.")

        self._omega_fit = _to_omega_intervals(self.fit_intervals)
        self._omega_work = _to_omega_intervals(self.work_intervals)
        self._omega_constraint = _complement(self._omega_work, (0.0, 0.5 * np.pi))

    @staticmethod
    def _orders(degree: int) -> np.ndarray:
        return np.arange(degree, -1, -2, dtype=int)

    @staticmethod
    def _design(omega: np.ndarray, orders: np.ndarray) -> np.ndarray:
        return np.cos(omega[:, None] * orders[None, :])

    @classmethod
    def _polynomial(cls, coefficients: np.ndarray, omega: np.ndarray, orders: np.ndarray) -> np.ndarray:
        return cls._design(np.asarray(omega), orders) @ coefficients

    @staticmethod
    def _polynomial_derivative(
        coefficients: np.ndarray, omega: np.ndarray, orders: np.ndarray
    ) -> np.ndarray:
        return -np.sum(
            (orders * coefficients)[None, :]
            * np.sin(np.asarray(omega)[:, None] * orders[None, :]),
            axis=1,
        )

    def _target_omega(self, omega: np.ndarray) -> np.ndarray:
        return self.target_scale * _evaluate(self.target, np.cos(omega))

    def _weight_omega(self, omega: np.ndarray) -> np.ndarray:
        return _evaluate(self.weight, np.cos(omega))

    def _target_derivative_omega(self, omega: np.ndarray) -> np.ndarray:
        x = np.cos(omega)
        if self.target_derivative is not None:
            return -np.sin(omega) * _evaluate(self.target_derivative, x) * self.target_scale
        step = 1e-6
        return (self._target_omega(omega + step) - self._target_omega(omega - step)) / (2 * step)

    def _weight_derivative_omega(self, omega: np.ndarray) -> np.ndarray:
        x = np.cos(omega)
        if self.weight_derivative is not None:
            return -np.sin(omega) * _evaluate(self.weight_derivative, x)
        step = 1e-6
        return (self._weight_omega(omega + step) - self._weight_omega(omega - step)) / (2 * step)

    def _weighted_error_derivative(
        self, coefficients: np.ndarray, omega: np.ndarray, orders: np.ndarray
    ) -> np.ndarray:
        polynomial = self._polynomial(coefficients, omega, orders)
        derivative = self._polynomial_derivative(coefficients, omega, orders)
        return (
            self._weight_derivative_omega(omega) * (self._target_omega(omega) - polynomial)
            + self._weight_omega(omega) * (self._target_derivative_omega(omega) - derivative)
        )

    def _roots(self, function: ArrayFunction, intervals: Sequence[Interval]) -> np.ndarray:
        roots: List[float] = []
        for left, right in intervals:
            grid = np.linspace(left, right, max(3, self.options.root_grid_size))
            values = np.asarray(function(grid), dtype=float)
            absolute_values = np.abs(values)
            if np.max(absolute_values) < 1e-10:
                continue
            near_zero = absolute_values < 1e-12
            isolated = near_zero.copy()
            isolated[1:-1] &= (
                (absolute_values[1:-1] <= absolute_values[:-2])
                & (absolute_values[1:-1] <= absolute_values[2:])
            )
            roots.extend(grid[isolated].tolist())
            signs = np.sign(values)
            signs[near_zero] = 0.0
            changes = np.where(signs[:-1] * signs[1:] < 0)[0]
            for index in changes:
                low, high = grid[index], grid[index + 1]
                low_value = float(function(np.array([low]))[0])
                for _ in range(self.options.root_bisection_iterations):
                    middle = 0.5 * (low + high)
                    middle_value = float(function(np.array([middle]))[0])
                    if low_value * middle_value <= 0:
                        high = middle
                    else:
                        low, low_value = middle, middle_value
                roots.append(0.5 * (low + high))
        return np.unique(np.asarray(roots, dtype=float)) if roots else np.array([], dtype=float)

    def _exchange(
        self,
        orders: np.ndarray,
        initial: np.ndarray,
        equality_points: np.ndarray,
        equality_values: np.ndarray,
        work_grid: np.ndarray,
    ):
        points = np.unique(np.concatenate((initial, equality_points)))
        points.sort()
        coefficients = np.zeros(len(orders), dtype=float)
        ripple = 0.0
        rank_deficient = False

        for iteration in range(1, self.options.max_exchange_iterations + 1):
            signs = np.where(np.arange(len(points)) % 2 == 0, 1.0, -1.0)
            is_equality = np.isin(points, equality_points)
            alternating = points[~is_equality]
            alternating_signs = signs[~is_equality]
            matrix_alt = np.hstack((
                self._weight_omega(alternating)[:, None] * self._design(alternating, orders),
                alternating_signs[:, None],
            ))
            rhs_alt = self._weight_omega(alternating) * self._target_omega(alternating)
            if equality_points.size:
                matrix_eq = np.hstack((
                    self._design(equality_points, orders),
                    np.zeros((len(equality_points), 1)),
                ))
                matrix = np.vstack((matrix_alt, matrix_eq))
                rhs = np.concatenate((rhs_alt, equality_values))
            else:
                matrix, rhs = matrix_alt, rhs_alt

            solution, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
            rank_deficient = rank_deficient or rank < min(matrix.shape)
            if not np.all(np.isfinite(solution)):
                raise RemezConvergenceError("The Remez linear system produced nonfinite coefficients.")
            coefficients, ripple = solution[:-1], float(solution[-1])

            extrema = self._roots(
                lambda omega: self._weighted_error_derivative(coefficients, omega, orders),
                self._omega_work,
            )
            candidates = np.unique(np.concatenate((extrema, _interval_edges(self._omega_work))))
            candidate_error = self._weight_omega(candidates) * (
                self._target_omega(candidates)
                - self._polynomial(coefficients, candidates, orders)
            )
            needed = len(orders) + 1 - len(equality_points)
            if needed < 0:
                raise RemezConvergenceError("Too many active constraints for the polynomial degree.")
            selected = (
                candidates[np.argsort(-np.abs(candidate_error))[:needed]]
                if needed else np.array([], dtype=float)
            )
            new_points = np.unique(np.concatenate((selected, equality_points)))
            new_points.sort()
            work_error = self._weight_omega(work_grid) * (
                self._target_omega(work_grid)
                - self._polynomial(coefficients, work_grid, orders)
            )
            stable = len(new_points) == len(points) and np.allclose(new_points, points)
            bounded = np.max(np.abs(work_error)) <= abs(ripple) + self.options.exchange_tolerance
            if stable and bounded:
                return coefficients, ripple, new_points, True, iteration, rank_deficient
            points = new_points
        return (
            coefficients,
            ripple,
            points,
            False,
            self.options.max_exchange_iterations,
            rank_deficient,
        )

    def _metrics(self, coefficients: np.ndarray) -> RemezMetrics:
        fit_errors = []
        for left, right in self.fit_intervals:
            x = np.linspace(left, right, max(2, self.options.metrics_grid_size))
            error = np.abs(np.polynomial.chebyshev.chebval(x, coefficients) - _evaluate(self.target, x))
            fit_errors.append(float(np.max(error)))
        work_x = _interval_grid(self.work_intervals, self.options.metrics_grid_size)
        work_error = np.abs(
            np.polynomial.chebyshev.chebval(work_x, coefficients) - _evaluate(self.target, work_x)
        )
        certificate = check_feasibility(coefficients, interval=(0.0, 1.0))
        critical = certificate.critical_points
        check_x = np.unique(np.concatenate((
            critical,
            np.linspace(0.0, 1.0, max(2, self.options.metrics_grid_size)),
        )))
        values = np.polynomial.chebyshev.chebval(check_x, coefficients)
        lower = _evaluate(self.lower_bound, check_x)
        upper = _evaluate(self.upper_bound, check_x)
        violation = np.maximum(lower - values, values - upper)
        return RemezMetrics(
            max_error=max(fit_errors),
            max_work_error=float(np.max(work_error)),
            max_magnitude=certificate.max_magnitude,
            max_constraint_violation=float(max(0.0, np.max(violation))),
            interval_errors=tuple(fit_errors),
        )

    def fit(
        self,
        degree: int,
        rescale: bool = True,
        strict: bool = False,
        initial_extremals: Optional[np.ndarray] = None,
    ) -> RemezResult:
        """Fit a polynomial of prescribed degree and parity."""

        if isinstance(degree, bool) or int(degree) != degree or degree < 1:
            raise ValueError("degree must be a positive integer.")
        degree = int(degree)
        orders = self._orders(degree)
        work_grid = _interval_grid(self._omega_work, self.options.fit_grid_size)
        constraint_grid = _interval_grid(
            self._omega_constraint, self.options.constraint_grid_size
        )
        if initial_extremals is None:
            previous = np.array([], dtype=float)
        else:
            initial_x = np.asarray(initial_extremals, dtype=float)
            if np.any((initial_x < 0.0) | (initial_x > 1.0)):
                raise ValueError("initial_extremals must lie in [0, 1].")
            previous = np.arccos(initial_x)

        coefficients = np.zeros(len(orders), dtype=float)
        ripple = 0.0
        previous_violation = 0.1
        exchange_converged = False
        total_exchanges = 0
        outer_converged = False
        encountered_rank_deficiency = False

        for outer_iteration in range(1, self.options.max_iterations + 1):
            if constraint_grid.size:
                extrema = self._roots(
                    lambda omega: self._polynomial_derivative(coefficients, omega, orders),
                    self._omega_constraint,
                )
                candidates = np.unique(np.concatenate((
                    extrema, _interval_edges(self._omega_constraint)
                )))
                x_candidates = np.cos(candidates)
                values = self._polynomial(coefficients, candidates, orders)
                lower = _evaluate(self.lower_bound, x_candidates)
                upper = _evaluate(self.upper_bound, x_candidates)
                below, above = values < lower, values > upper
                equality_points = np.concatenate((candidates[below], candidates[above]))
                equality_values = np.concatenate((lower[below], upper[above]))
            else:
                equality_points = np.array([], dtype=float)
                equality_values = np.array([], dtype=float)

            needed = max(0, len(orders) + 1 - len(equality_points))
            if previous.size == 0:
                indexes = np.linspace(0, len(work_grid) - 1, needed, dtype=int)
                initial = work_grid[indexes] if needed else np.array([], dtype=float)
            else:
                initial = previous

            (
                coefficients,
                ripple,
                previous,
                exchange_converged,
                exchanges,
                rank_deficient,
            ) = self._exchange(orders, initial, equality_points, equality_values, work_grid)
            total_exchanges += exchanges
            encountered_rank_deficiency = encountered_rank_deficiency or rank_deficient
            if constraint_grid.size:
                x_constraint = np.cos(constraint_grid)
                values = self._polynomial(coefficients, constraint_grid, orders)
                violation = np.maximum(
                    _evaluate(self.lower_bound, x_constraint) - values,
                    values - _evaluate(self.upper_bound, x_constraint),
                )
                max_violation = float(max(0.0, np.max(violation)))
            else:
                max_violation = 0.0
            outer_converged = abs(max_violation - previous_violation) <= (
                self.options.relative_tolerance * max(1.0, abs(previous_violation))
            )
            previous_violation = max_violation
            if outer_converged and exchange_converged:
                break

        raw = _full_coefficients(coefficients, orders, degree)
        raw_metrics = self._metrics(raw)
        scale_factor = max(1.0, raw_metrics.max_magnitude)
        scaled = raw / scale_factor
        scaled_metrics = self._metrics(scaled)
        converged = bool(
            outer_converged and exchange_converged and not encountered_rank_deficiency
        )
        if encountered_rank_deficiency:
            message = "Remez encountered a rank-deficient exchange system; inspect the returned metrics."
        elif converged:
            message = "Converged."
        else:
            message = "Remez reached an iteration limit before convergence."
        if not converged:
            if strict:
                raise RemezConvergenceError(message)
            warnings.warn(message, RemezConvergenceWarning, stacklevel=2)

        selected = scaled if rescale else raw
        selected_metrics = scaled_metrics if rescale else raw_metrics
        parity = degree % 2
        extrema_x = np.sort(np.cos(previous))
        return RemezResult(
            degree=degree,
            coefficients=selected.copy(),
            raw_coefficients=raw.copy(),
            scaled_coefficients=scaled.copy(),
            parity_coefficients=selected[parity::2].copy(),
            raw_parity_coefficients=raw[parity::2].copy(),
            scaled_parity_coefficients=scaled[parity::2].copy(),
            scale_factor=scale_factor,
            ripple_amplitude=float(ripple / scale_factor if rescale else ripple),
            raw_ripple_amplitude=float(ripple),
            scaled_ripple_amplitude=float(ripple / scale_factor),
            extremal_points=extrema_x,
            fit_intervals=self.fit_intervals,
            work_intervals=self.work_intervals,
            metrics=selected_metrics,
            raw_metrics=raw_metrics,
            scaled_metrics=scaled_metrics,
            rescaled=bool(rescale),
            converged=converged,
            outer_iterations=outer_iteration,
            exchange_iterations=total_exchanges,
            message=message,
        )


def remez(
    target: ArrayFunction,
    degree: int,
    fit_intervals: Sequence[Interval],
    work_intervals: Optional[Sequence[Interval]] = None,
    target_scale: float = 1.0,
    rescale: bool = True,
    strict: bool = False,
    lower_bound: Bound = -1.0,
    upper_bound: Bound = 1.0,
    weight: Optional[ArrayFunction] = None,
    target_derivative: Optional[ArrayFunction] = None,
    weight_derivative: Optional[ArrayFunction] = None,
    options: Optional[RemezOptions] = None,
) -> RemezResult:
    """Compute a constrained minimax polynomial approximation on ``[0, 1]``.

    Parameters
    ----------
    target
        Vectorized target function in polynomial coordinates.
    degree
        Maximum degree of the approximation.  Its parity determines the
        Chebyshev basis used by the fit.
    fit_intervals
        Closed intervals on which the uniform approximation error is minimized.
    work_intervals
        Optional subsets of ``fit_intervals`` used by the exchange iteration.
        Shrinking these intervals can provide a transition buffer near an
        endpoint.  Errors are still reported on ``fit_intervals``.
    target_scale
        Positive factor, at most one, applied only during optimization.  Metrics
        always compare with the original unscaled target.
    rescale
        If true, select the raw polynomial divided by
        ``max(1, max(abs(P)))``.  Both raw and scaled coefficients are retained
        in the result regardless of this choice.
    strict
        If true, raise :class:`RemezConvergenceError` instead of returning a
        finite unconverged iterate with :class:`RemezConvergenceWarning`.
    lower_bound, upper_bound
        Scalar bounds or vectorized bound functions on ``[0, 1]``.
    weight
        Optional positive error-weighting function in polynomial coordinates.
    target_derivative, weight_derivative
        Optional derivatives with respect to ``x``.  Centered finite
        differences are used when they are omitted.
    options
        Advanced numerical controls.

    Returns
    -------
    RemezResult
        Full ascending Chebyshev coefficients, raw and scaled variants, and
        convergence, approximation, and feasibility diagnostics.
    """

    fitter = ConstrainedRemezFitter(
        target=target,
        fit_intervals=fit_intervals,
        work_intervals=work_intervals,
        target_scale=target_scale,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        weight=weight,
        target_derivative=target_derivative,
        weight_derivative=weight_derivative,
        options=options,
    )
    return fitter.fit(degree=degree, rescale=rescale, strict=strict)


def plot_remez_result(
    result: RemezResult,
    target: ArrayFunction,
    variant: str = "selected",
    axes=None,
    error_scale: str = "linear",
    points: int = 2000,
):
    """Plot one Remez approximation and its pointwise fitting error."""

    import matplotlib.pyplot as plt

    if error_scale not in ("linear", "log"):
        raise ValueError("error_scale must be 'linear' or 'log'.")
    if int(points) < 2:
        raise ValueError("points must be at least two.")
    if axes is None:
        figure, axes_array = plt.subplots(1, 2, figsize=(11, 4))
        approximation_axis, error_axis = axes_array
    else:
        if len(axes) != 2:
            raise ValueError("axes must contain exactly two Matplotlib axes.")
        approximation_axis, error_axis = axes
        figure = approximation_axis.figure

    x_domain = np.linspace(0.0, 1.0, int(points))
    approximation_axis.plot(x_domain, result.evaluate(x_domain, variant), label="Remez polynomial")
    approximation_axis.axhline(1.0, color="0.6", linestyle=":", linewidth=1)
    approximation_axis.axhline(-1.0, color="0.6", linestyle=":", linewidth=1)
    for index, (left, right) in enumerate(result.fit_intervals):
        x = np.linspace(left, right, int(points))
        target_values = _evaluate(target, x)
        approximation_axis.plot(
            x, target_values, color="black", linestyle="--",
            label="Target" if index == 0 else None,
        )
        error_axis.plot(x, np.abs(result.evaluate(x, variant) - target_values))
        approximation_axis.axvspan(left, right, color="0.9", zorder=-1)
        error_axis.axvspan(left, right, color="0.9", zorder=-1)
    approximation_axis.set_xlabel("x")
    approximation_axis.set_ylabel("Value")
    approximation_axis.legend()
    error_axis.set_xlabel("x")
    error_axis.set_ylabel("Absolute error")
    error_axis.set_yscale(error_scale)
    figure.tight_layout()
    return figure, (approximation_axis, error_axis)


__all__ = [
    "ConstrainedRemezFitter",
    "RemezConvergenceError",
    "RemezConvergenceWarning",
    "RemezMetrics",
    "RemezOptions",
    "RemezResult",
    "plot_remez_result",
    "remez",
]
