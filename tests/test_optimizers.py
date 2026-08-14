"""Tests for the optimizers module."""

import numpy as np
import pytest
from qsppack import lbfgs, coordinate_minimization, newton
from qsppack.objective import grad_sym, grad_sym_real, obj_sym

def test_lbfgs_basic():
    """Test basic functionality of L-BFGS optimizer."""
    # Simple objective function: f(x) = x^2
    def obj(x, delta, opts):
        return np.array([x[0]**2])
    
    # Gradient of objective function: f'(x) = 2x
    def grad(x, delta, opts):
        return np.array([[2*x[0]]]), np.array([x[0]**2])
    
    # Test parameters
    delta = np.array([1.0])
    x0 = np.array([2.0])
    opts = {
        'maxiter': 100,
        'criteria': 1e-6,
        'print': False
    }
    
    x, obj_value, iter = lbfgs(obj, grad, delta, x0, opts)
    
    assert isinstance(x, np.ndarray)
    assert isinstance(obj_value, float)
    assert isinstance(iter, int)
    assert abs(x[0]) < 1e-3  # Should be close to 0
    assert obj_value < 1e-6  # Objective should be small


def test_lbfgs_uses_its_correction_history():
    """L-BFGS should converge rapidly on an anisotropic quadratic."""
    hessian = np.diag([1.0, 100.0])

    def obj(x, delta, opts):
        return np.array([0.5 * x @ hessian @ x])

    def grad(x, delta, opts):
        return np.array([hessian @ x]), obj(x, delta, opts)

    x, obj_value, iterations = lbfgs(
        obj,
        grad,
        np.array([0.0]),
        np.array([2.0, 2.0]),
        {
            'maxiter': 30,
            'criteria': 1e-8,
            'lmem': 3,
            'print': False,
        },
    )

    assert iterations < 30
    assert np.linalg.norm(x, np.inf) < 1e-6
    assert obj_value < 1e-12


@pytest.mark.parametrize("option", ["maxiter", "lmem"])
def test_lbfgs_rejects_nonpositive_iteration_options(option):
    def obj(x, delta, opts):
        return np.array([x[0] ** 2])

    def grad(x, delta, opts):
        return np.array([[2 * x[0]]]), obj(x, delta, opts)

    with pytest.raises(ValueError, match=option):
        lbfgs(
            obj,
            grad,
            np.array([0.0]),
            np.array([1.0]),
            {option: 0, "print": False},
        )


def test_real_gradient_does_not_mutate_phase_input():
    """Even-parity scaling must not modify the caller's phase vector."""
    phi = np.array([0.2, -0.1])
    original = phi.copy()
    opts = {
        'parity': 0,
        'target': lambda x: 0.0,
    }

    grad_sym_real(phi, np.array([0.25]), opts)

    np.testing.assert_array_equal(phi, original)


@pytest.mark.parametrize('parity', [0, 1])
@pytest.mark.parametrize('gradient', [grad_sym, grad_sym_real])
def test_qsp_gradient_matches_objective_finite_difference(parity, gradient):
    """Both gradient implementations should differentiate obj_sym."""
    phi = np.array([0.12, -0.08, 0.03])
    if parity == 0:
        phi[0] *= 2
    delta = np.array([-0.8, -0.1, 0.5, 0.9])
    opts = {
        'parity': parity,
        'target': lambda x: 0.2 * x ** (parity + 1),
    }
    calculated, gradient_objective = gradient(phi.copy(), delta, opts)
    step = 1e-7
    finite_difference = np.empty_like(phi)

    for column in range(len(phi)):
        perturbation = np.zeros_like(phi)
        perturbation[column] = step
        finite_difference[column] = (
            np.mean(obj_sym(phi + perturbation, delta, opts))
            - np.mean(obj_sym(phi - perturbation, delta, opts))
        ) / (2 * step)

    np.testing.assert_allclose(gradient_objective, obj_sym(phi, delta, opts))
    np.testing.assert_allclose(
        np.mean(calculated, axis=0), finite_difference, atol=1e-8, rtol=1e-8
    )

def test_coordinate_minimization_basic():
    """Test basic functionality of coordinate minimization."""
    # Keep the target inside the fixed-point method's contraction regime.
    coef = np.array([0.5])
    parity = 1
    opts = {
        'maxiter': 100,
        'criteria': 1e-6,
        'print': False
    }
    
    phi, err, iter, runtime = coordinate_minimization(coef, parity, opts)
    
    assert isinstance(phi, np.ndarray)
    assert isinstance(err, float)
    assert isinstance(iter, int)
    assert isinstance(runtime, float)
    assert err < 1e-6  # Error should be small


def test_coordinate_minimization_boundary_target():
    """A unit-amplitude boundary target converges, but needs more iterations."""
    _, err, iterations, _ = coordinate_minimization(
        np.array([1.0]),
        1,
        {
            'maxiter': 2000,
            'criteria': 1e-6,
            'print': False,
        },
    )

    assert iterations < 2000
    assert err < 1e-6

def test_newton_basic():
    """Test basic functionality of Newton's method."""
    coef = np.array([1.0])
    parity = 1
    opts = {
        'maxiter': 100,
        'criteria': 1e-6,
        'print': False
    }
    
    phi, err, iter, runtime = newton(coef, parity, opts)
    
    assert isinstance(phi, np.ndarray)
    assert isinstance(err, float)
    assert isinstance(iter, int)
    assert isinstance(runtime, float)
    assert err < 1e-6  # Error should be small
