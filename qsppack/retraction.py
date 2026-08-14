"""Nonlinear Fourier retraction for bounded QSP polynomials.

This module packages the Weiss factorization and nonlinear Fourier transforms
used by :mod:`qsppack.nlfa` into a polynomial-level operation.  Public inputs
and outputs are full Chebyshev coefficient arrays in ascending order.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .nlfa import (
    b_from_cheb,
    forward_nonlinear_FFT,
    inverse_nonlinear_FFT,
    weiss,
)


@dataclass
class RetractionMetrics:
    """Feasibility diagnostics evaluated at endpoints and critical points."""

    max_magnitude: float
    max_constraint_violation: float
    maximizer: float
    critical_points: np.ndarray


@dataclass
class RetractionResult:
    """Result of nonlinear Fourier retraction.

    Coefficient arrays use ascending Chebyshev order.  The intermediate
    nonlinear Fourier data are retained so numerical reconstruction can be
    audited without repeating the factorization.
    """

    coefficients: np.ndarray
    original_coefficients: np.ndarray
    parity: int
    degree: int
    n_weiss: int
    original_metrics: RetractionMetrics
    metrics: RetractionMetrics
    b_coefficients: np.ndarray
    a_coefficients: np.ndarray
    retracted_b_coefficients: np.ndarray
    gammas: np.ndarray
    reconstruction_residual: float

    def evaluate(self, x, variant: str = "retracted"):
        """Evaluate the retracted or original polynomial at ``x``."""

        if variant == "retracted":
            coefficients = self.coefficients
        elif variant == "original":
            coefficients = self.original_coefficients
        else:
            raise ValueError("variant must be 'retracted' or 'original'.")
        return np.polynomial.chebyshev.chebval(x, coefficients)


def _validate_coefficients(coefficients, parity: Optional[int], tolerance: float):
    values = np.asarray(coefficients)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("coefficients must be a nonempty one-dimensional array.")
    values = np.real_if_close(values, tol=1000)
    if np.iscomplexobj(values):
        raise ValueError("coefficients must be real up to numerical roundoff.")
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("coefficients must be finite.")

    if parity is None:
        nonzero = np.flatnonzero(np.abs(values) > tolerance)
        parity = int(nonzero[-1] % 2) if nonzero.size else int((values.size - 1) % 2)
    if parity not in (0, 1):
        raise ValueError("parity must be zero (even) or one (odd).")
    if np.any(np.abs(values[1 - parity :: 2]) > tolerance):
        raise ValueError("coefficients contain terms inconsistent with parity.")

    active = np.flatnonzero(np.abs(values) > tolerance)
    degree = int(active[-1]) if active.size else int(parity)
    return values.copy(), int(parity), degree


def _critical_metrics(coefficients: np.ndarray) -> RetractionMetrics:
    derivative = np.polynomial.chebyshev.chebder(coefficients)
    if derivative.size <= 1 or np.all(np.abs(derivative) < 1e-15):
        roots = np.empty(0, dtype=float)
    else:
        roots = np.polynomial.chebyshev.chebroots(derivative)
        roots = np.real(roots[np.abs(np.imag(roots)) <= 1e-10])
        roots = roots[(roots > -1.0) & (roots < 1.0)]
        roots = np.unique(np.clip(roots, -1.0, 1.0))
    points = np.concatenate(([-1.0], roots, [1.0]))
    magnitudes = np.abs(np.polynomial.chebyshev.chebval(points, coefficients))
    index = int(np.argmax(magnitudes))
    maximum = float(magnitudes[index])
    return RetractionMetrics(
        max_magnitude=maximum,
        max_constraint_violation=max(0.0, maximum - 1.0),
        maximizer=float(points[index]),
        critical_points=points,
    )


def _chebyshev_from_b(
    b_coefficients: np.ndarray, coefficient_count: int, parity: int
) -> np.ndarray:
    partial_count = (coefficient_count + (1 - parity)) // 2
    if parity:
        partial = (
            b_coefficients[partial_count - 1 :: -1]
            + b_coefficients[partial_count:]
        )
    elif partial_count == 1:
        partial = np.asarray([b_coefficients[0]])
    else:
        center = partial_count - 1
        partial = np.concatenate(
            ([b_coefficients[center]],
             b_coefficients[center - 1 :: -1] + b_coefficients[center + 1 :])
        )
    result = np.zeros(coefficient_count, dtype=np.result_type(partial, float))
    result[parity::2] = partial
    result = np.real_if_close(result, tol=1000)
    if np.iscomplexobj(result):
        raise RuntimeError("retraction produced materially complex coefficients.")
    return np.asarray(result, dtype=float)


def retract(
    coefficients,
    *,
    n_weiss: int = 2**16,
    parity: Optional[int] = None,
    parity_tolerance: float = 1e-12,
) -> RetractionResult:
    """Retract a definite-parity Chebyshev polynomial into the QSP feasible set.

    Parameters
    ----------
    coefficients
        Full Chebyshev coefficients in ascending order.  Coefficients of the
        opposite parity must vanish up to ``parity_tolerance``.
    n_weiss
        Even number of roots of unity used by the Weiss factorization.  Larger
        values reduce discretization error at increased runtime and memory cost.
    parity
        Optional explicit parity: zero for even or one for odd.  By default it
        is inferred from the highest nonzero coefficient.
    parity_tolerance
        Absolute threshold used for parity inference and validation.

    Returns
    -------
    RetractionResult
        Retracted coefficients, critical-point feasibility metrics, and NLFA
        reconstruction diagnostics.

    Notes
    -----
    Feasibility is checked globally on ``[-1, 1]`` at the endpoints and every
    real root of the derivative.  This certifies the maximum magnitude of the
    returned real polynomial up to numerical root-finding accuracy.
    """

    if parity_tolerance < 0:
        raise ValueError("parity_tolerance must be nonnegative.")
    if isinstance(n_weiss, bool) or int(n_weiss) != n_weiss:
        raise ValueError("n_weiss must be an even positive integer.")
    n_weiss = int(n_weiss)
    if n_weiss < 2 or n_weiss % 2:
        raise ValueError("n_weiss must be an even positive integer.")

    original, parity, degree = _validate_coefficients(
        coefficients, parity, parity_tolerance
    )
    partial = original[parity::2]
    b_coefficients = b_from_cheb(partial, parity)
    if n_weiss < len(b_coefficients):
        raise ValueError("n_weiss must be at least the length of the NLFA polynomial.")

    if len(b_coefficients) == 1:
        magnitude_squared = float(np.abs(b_coefficients[0]) ** 2)
        if magnitude_squared > 1.0 + 1e-14:
            raise ValueError("a constant polynomial outside [-1, 1] cannot be retracted.")
        a_coefficients = np.asarray([np.sqrt(max(0.0, 1.0 - magnitude_squared))])
        if a_coefficients[0] == 0.0:
            gammas = np.asarray([np.copysign(np.inf, b_coefficients[0])])
        else:
            gammas = np.asarray([b_coefficients[0] / a_coefficients[0]])
        retracted_b = b_coefficients.copy()
    else:
        a_coefficients = weiss(b_coefficients, n_weiss)
        gammas, _, _ = inverse_nonlinear_FFT(a_coefficients, b_coefficients)
        _, retracted_b = forward_nonlinear_FFT(gammas)
    retracted = _chebyshev_from_b(retracted_b, original.size, parity)
    denominator = max(float(np.linalg.norm(b_coefficients)), np.finfo(float).eps)
    residual = float(np.linalg.norm(retracted_b - b_coefficients) / denominator)

    return RetractionResult(
        coefficients=retracted,
        original_coefficients=original,
        parity=parity,
        degree=degree,
        n_weiss=n_weiss,
        original_metrics=_critical_metrics(original),
        metrics=_critical_metrics(retracted),
        b_coefficients=b_coefficients,
        a_coefficients=a_coefficients,
        retracted_b_coefficients=retracted_b,
        gammas=gammas,
        reconstruction_residual=residual,
    )


__all__ = ["RetractionMetrics", "RetractionResult", "retract"]
