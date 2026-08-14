import numpy as np
import pytest

from qsppack import RetractionResult, retract
from qsppack.nlfa import weiss


def test_fft_weiss_preserves_reference_values():
    result = weiss(np.array([0.38157934, 0.05342111, 0.45789521]), 8)
    assert result == pytest.approx([0.76099391, -0.08783997, -0.23742773])

    longer = weiss(
        np.array([
            0.237136846,
            0.106711580,
            0.432585034,
            -0.304180174,
            -0.000474273691,
            0.521701060,
        ]),
        64,
    )
    assert longer == pytest.approx([
        0.47379318,
        0.09424218,
        0.11612456,
        -0.24713393,
        -0.06540336,
        -0.26132533,
    ])


def test_odd_retraction_enforces_bound_and_preserves_parity():
    coefficients = np.array([0.0, 0.2, 0.0, 0.9])
    result = retract(coefficients, n_weiss=4096)

    assert isinstance(result, RetractionResult)
    assert result.parity == 1
    assert result.degree == 3
    assert result.n_weiss == 4096
    assert result.original_metrics.max_magnitude > 1.0
    assert result.metrics.max_magnitude <= 1.0 + 1e-10
    assert result.metrics.max_constraint_violation <= 1e-10
    assert np.array_equal(result.coefficients[::2], np.zeros(2))
    assert result.reconstruction_residual > 0.0


def test_even_retraction_and_constant_edge_case():
    result = retract([0.2, 0.0, 0.9], n_weiss=4096, parity=0)
    constant = retract([0.5], n_weiss=64, parity=0)

    assert result.metrics.max_magnitude <= 1.0 + 1e-10
    assert np.array_equal(result.coefficients[1::2], np.zeros(1))
    assert constant.coefficients == pytest.approx([0.5], abs=2e-4)
    assert constant.metrics.max_constraint_violation == 0.0
    assert retract([1.0], n_weiss=64).coefficients == pytest.approx([1.0])


def test_metrics_use_critical_points_not_only_a_uniform_grid():
    coefficients = np.zeros(13)
    coefficients[0] = 0.2
    coefficients[2] = -0.35
    coefficients[6] = 0.55
    coefficients[12] = 0.75
    result = retract(coefficients, n_weiss=4096, parity=0)

    points = result.original_metrics.critical_points
    derivative = np.polynomial.chebyshev.chebder(coefficients)
    assert len(points) > 2
    assert np.max(np.abs(np.polynomial.chebyshev.chebval(points[1:-1], derivative))) < 1e-7
    assert result.original_metrics.max_magnitude == pytest.approx(
        np.max(np.abs(np.polynomial.chebyshev.chebval(points, coefficients)))
    )


@pytest.mark.parametrize(
    "coefficients, kwargs, message",
    [
        ([0.1, 0.2], {}, "inconsistent with parity"),
        ([0.0, 0.5], {"n_weiss": 63}, "even positive integer"),
        ([0.0, 0.5, 0.0, 0.1], {"n_weiss": 2}, "at least the length"),
        ([], {}, "nonempty one-dimensional"),
        ([[0.5]], {}, "nonempty one-dimensional"),
        ([0.5 + 0.2j], {}, "real"),
        ([np.nan], {}, "finite"),
        ([0.5], {"parity": 2}, "parity"),
        ([0.5], {"n_weiss": True}, "even positive integer"),
        ([0.5], {"n_weiss": 4.5}, "even positive integer"),
        ([0.5], {"parity_tolerance": -1}, "nonnegative"),
        ([1.1], {"n_weiss": 64}, "outside"),
    ],
)
def test_validation(coefficients, kwargs, message):
    with pytest.raises(ValueError, match=message):
        retract(coefficients, **kwargs)


def test_evaluate_variants():
    result = retract([0.0, 1.01], n_weiss=4096)
    x = np.array([-1.0, 0.0, 1.0])

    assert result.evaluate(x, "original") == pytest.approx([-1.01, 0.0, 1.01])
    assert np.max(np.abs(result.evaluate(x))) <= 1.0
    with pytest.raises(ValueError, match="variant"):
        result.evaluate(x, "unknown")
