"""Tests for the utils module."""

import numpy as np
import pytest

from qsppack import (
    chebyshev_to_func,
    cvx_poly_coef,
    get_entry,
    get_unitary,
    reduced_to_full,
)
from qsppack.utils import (
    F,
    F_Jacobian,
    get_pim_sym,
    get_pim_sym_real,
    get_unitary_sym,
)


def test_get_unitary_matches_simple_analytic_value():
    phase = np.array([0.0, np.pi / 4])

    result = get_unitary(phase, x=0.5)

    assert result == pytest.approx(0.5 / np.sqrt(2))


@pytest.mark.parametrize(
    ("parity", "target_pre", "expected"),
    [
        (0, False, [0.2, 0.2, 0.2]),
        (0, True, [0.2 + np.pi / 4, 0.2, 0.2 + np.pi / 4]),
        (1, False, [0.2, 0.1, 0.1, 0.2]),
        (1, True, [0.2 + np.pi / 4, 0.1, 0.1, 0.2 + np.pi / 4]),
    ],
)
def test_reduced_to_full_exact_conventions(parity, target_pre, expected):
    reduced = np.array([0.1, 0.2])
    original = reduced.copy()

    result = reduced_to_full(reduced, parity, target_pre)

    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(reduced, original)


@pytest.mark.parametrize(
    ("phases", "parity", "target_pre", "exception", "message"),
    [
        ([], 0, True, ValueError, "nonempty one-dimensional"),
        ([[0.1, 0.2]], 0, True, ValueError, "nonempty one-dimensional"),
        ([np.nan], 0, True, ValueError, "finite"),
        ([0.1 + 0.2j], 0, True, ValueError, "real"),
        ([0.1], 2, True, ValueError, "parity"),
        ([0.1], 0, 1, TypeError, "boolean"),
        ([0.1], 0, None, TypeError, "boolean"),
    ],
)
def test_reduced_to_full_validation(phases, parity, target_pre, exception, message):
    with pytest.raises(exception, match=message):
        reduced_to_full(phases, parity, target_pre)


def test_chebyshev_to_func_partial_coefficients():
    x = np.array([0.0, 0.5, 1.0])

    result = chebyshev_to_func(x, coef=np.array([1.0]), parity=1, partialcoef=True)

    np.testing.assert_allclose(result, x, atol=1e-15)


def test_chebyshev_to_func_accepts_scalar_input():
    result = chebyshev_to_func(0.5, np.array([1.0]), 1, True)

    assert isinstance(result, float)
    assert result == pytest.approx(0.5)


@pytest.mark.parametrize("parity", [0, 1])
def test_chebyshev_to_func_full_coefficients(parity):
    x = np.linspace(-1.0, 1.0, 21)
    coefficients = np.array([0.2, 0.1, -0.3, 0.4])
    parity_coefficients = coefficients.copy()
    parity_coefficients[1 - parity :: 2] = 0.0

    result = chebyshev_to_func(x, coefficients, parity, partialcoef=False)

    expected = np.polynomial.chebyshev.chebval(x, parity_coefficients)
    np.testing.assert_allclose(result, expected, atol=1e-15)


@pytest.mark.parametrize("method", ["SLSQP", "cvxpy", "linprog"])
def test_cvx_poly_coef_backends_approximate_linear_target(method, capsys):
    options = {
        "method": method,
        "intervals": [0.0, 1.0],
        "npts": 40,
        "epsil": 0.01,
        "fscale": 1.0,
        "isplot": False,
        "objnorm": np.inf,
        "verbose": False,
    }
    original = options.copy()

    coefficients = cvx_poly_coef(lambda x: 0.5 * x, 3, options)

    grid = np.linspace(-1.0, 1.0, 201)
    approximation = np.polynomial.chebyshev.chebval(grid, coefficients)
    np.testing.assert_allclose(approximation, 0.5 * grid, atol=1e-5)
    assert np.max(np.abs(approximation)) <= 0.99 + 1e-10
    assert options == original
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("degree", "options", "exception", "message"),
    [
        (True, {}, ValueError, "nonnegative integer"),
        (2, {"intervals": [0.0, 0.5, 0.75]}, ValueError, "endpoint pairs"),
        (2, {"intervals": [0.5, 0.0]}, ValueError, "ordered"),
        (2, {"intervals": [-0.1, 0.5]}, ValueError, r"\[0, 1\]"),
        (2, {"npts": 1}, ValueError, "at least two"),
        (2, {"method": "unknown"}, ValueError, "not supported"),
        (2, {"method": "linprog", "objnorm": 2}, ValueError, "np.inf"),
    ],
)
def test_cvx_poly_coef_validation(degree, options, exception, message):
    with pytest.raises(exception, match=message):
        cvx_poly_coef(lambda x: x, degree, options)


def test_cvx_poly_coef_validates_target_output():
    with pytest.raises(ValueError, match="broadcast"):
        cvx_poly_coef(lambda x: [1.0, 2.0], 2, {"npts": 20})

    coefficients = cvx_poly_coef(
        lambda x: 0.5 * x.astype(complex) + 1e-16j,
        1,
        {"npts": 20, "fscale": 1.0},
    )
    np.testing.assert_allclose(coefficients, [0.0, 0.5], atol=1e-5)

    with pytest.raises(ValueError, match="real target"):
        cvx_poly_coef(lambda x: x + 0.1j, 1, {"npts": 20})


@pytest.mark.parametrize("parity", [0, 1])
@pytest.mark.parametrize("use_real", [False, True])
def test_phase_map_matches_jacobian_value(parity, use_real):
    """F and F_Jacobian must evaluate the same symmetric QSP map."""
    phi = np.array([0.12, -0.08, 0.03])
    opts = {"useReal": use_real}

    value = F(phi, parity, opts)
    jacobian_value, _ = F_Jacobian(phi, parity, opts)

    np.testing.assert_allclose(value, jacobian_value, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("parity", [0, 1])
def test_real_and_complex_phase_maps_agree(parity):
    """Real and complex implementations should encode the same map."""
    phi = np.array([0.12, -0.08, 0.03])
    x = 0.37

    assert get_pim_sym_real(phi, x, parity) == pytest.approx(
        get_pim_sym(phi, x, parity), abs=1e-12
    )


@pytest.mark.parametrize("parity", [0, 1])
def test_symmetric_unitary_matches_full_phase_evaluation(parity):
    """The reduced L-BFGS representation should match the public full one."""
    reduced_phi = np.array([0.12, -0.08, 0.03])
    lbfgs_phi = reduced_phi.copy()
    if parity == 0:
        lbfgs_phi[0] *= 2
    x = 0.37

    symmetric_value = np.real(get_unitary_sym(lbfgs_phi, x, parity)[0, 0])
    full_phi = reduced_to_full(reduced_phi, parity, True)
    full_value = get_unitary(full_phi, x)

    assert symmetric_value == pytest.approx(full_value, abs=1e-12)


def test_get_entry_does_not_mutate_full_phases():
    phases = np.array([0.1, 0.2, 0.1])
    original = phases.copy()

    get_entry(
        np.linspace(-1.0, 1.0, 5),
        phases,
        {"typePhi": "full", "targetPre": False, "parity": 0},
    )

    np.testing.assert_array_equal(phases, original)


@pytest.mark.parametrize("parity", [0, 1])
def test_phase_map_jacobian_matches_finite_difference(parity):
    """F_Jacobian should differentiate the coefficient map returned by F."""
    phi = np.array([0.12, -0.08, 0.03])
    opts = {"useReal": True}
    _, jacobian = F_Jacobian(phi, parity, opts)
    step = 1e-7
    finite_difference = np.empty_like(jacobian)

    for column in range(len(phi)):
        perturbation = np.zeros_like(phi)
        perturbation[column] = step
        finite_difference[:, column] = (
            F(phi + perturbation, parity, opts)
            - F(phi - perturbation, parity, opts)
        ) / (2 * step)

    np.testing.assert_allclose(jacobian, finite_difference, atol=1e-8, rtol=1e-8)
