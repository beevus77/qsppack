"""Regression tests for nonlinear Fourier analysis helpers."""

import numpy as np
import pytest

from qsppack.nlfa import (
    b_from_cheb,
    forward_nlft,
    forward_nonlinear_FFT,
    inverse_nonlinear_FFT,
    weiss,
)


@pytest.mark.parametrize(
    ("coefficients", "parity", "expected"),
    [
        ([2, -1, 6, -7, 1], 0, [0.5, -3.5, 3, -0.5, 2, -0.5, 3, -3.5, 0.5]),
        ([1, 2, 3], 1, [1.5, 1, 0.5, 0.5, 1, 1.5]),
    ],
)
def test_b_from_cheb_accepts_documented_array_like(coefficients, parity, expected):
    np.testing.assert_allclose(b_from_cheb(coefficients, parity), expected)


@pytest.mark.parametrize(
    ("coefficients", "parity", "message"),
    [
        ([], 0, "nonempty"),
        ([[1.0]], 0, "one-dimensional"),
        ([1.0], -1, "parity"),
        ([1.0], True, "parity"),
    ],
)
def test_b_from_cheb_validation(coefficients, parity, message):
    with pytest.raises(ValueError, match=message):
        b_from_cheb(coefficients, parity)


@pytest.mark.parametrize(
    "gammas",
    [
        np.array([0.2]),
        np.array([0.1, -0.5, 0.3]),
        np.array([0.1 + 0.2j, -0.3 + 0.1j, 0.2 - 0.4j]),
    ],
)
def test_forward_nlft_matches_recursive_transform(gammas):
    _, expected = forward_nonlinear_FFT(gammas)

    result = forward_nlft(gammas)

    np.testing.assert_allclose(result, expected, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("function", [forward_nlft, forward_nonlinear_FFT])
def test_forward_transforms_reject_empty_input(function):
    with pytest.raises(ValueError, match="nonempty"):
        function([])


def test_forward_nonlinear_fft_validates_offset():
    with pytest.raises(ValueError, match="nonnegative integer"):
        forward_nonlinear_FFT([0.1], m=-1)


@pytest.mark.parametrize(
    ("a", "b", "message"),
    [
        ([], [], "nonempty"),
        ([1.0, 2.0], [1.0], "same length"),
        ([[1.0]], [[1.0]], "one-dimensional"),
    ],
)
def test_inverse_nonlinear_fft_validation(a, b, message):
    with pytest.raises(ValueError, match=message):
        inverse_nonlinear_FFT(a, b)


def test_weiss_single_coefficient_and_validation():
    result = weiss([0.25], 8)

    assert result.shape == (1,)
    assert np.all(np.isfinite(result))
    with pytest.raises(ValueError, match="nonempty"):
        weiss([], 8)
    with pytest.raises(ValueError, match="even integer"):
        weiss([0.25], 7)
