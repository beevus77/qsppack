"""Tests for the solver module."""

import numpy as np
import pytest
from qsppack.solver import solve
from qsppack.utils import cvx_poly_coef, get_entry

def test_solve_basic():
    """Test basic functionality of solve function."""
    # Test case: simple polynomial P(x) = x
    coef = np.array([1.0])
    parity = 1  # odd polynomial
    opts = {
        'criteria': 1e-12,
        'useReal': True,
        'targetPre': True,
        'method': 'FPI',
        'typePhi': 'full'
    }
    
    phi, out = solve(coef, parity, opts)
    
    # Basic assertions
    assert phi is not None
    assert out is not None
    assert isinstance(phi, np.ndarray)
    assert isinstance(out, dict)
    assert 'iter' in out
    assert 'time' in out
    assert 'value' in out
    assert out['parity'] == parity
    assert out['targetPre'] == opts['targetPre']
    assert out['typePhi'] == opts['typePhi']

def test_solve_invalid_method():
    """Test solve function with invalid method."""
    coef = np.array([1.0])
    parity = 1
    opts = {
        'method': 'INVALID_METHOD'
    }
    
    phi, out = solve(coef, parity, opts)
    assert phi is None
    assert out is None

def test_solve_different_parity():
    """Test solve function with even parity polynomial."""
    # Test case: simple polynomial P(x) = x^2
    coef = np.array([1.0])
    parity = 0  # even polynomial
    opts = {
        'criteria': 1e-12,
        'useReal': True,
        'targetPre': True,
        'method': 'FPI',
        'typePhi': 'full'
    }
    
    phi, out = solve(coef, parity, opts)
    
    assert phi is not None
    assert out is not None
    assert out['parity'] == parity 


@pytest.mark.parametrize('method', ['FPI', 'Newton', 'LBFGS', 'NLFT'])
@pytest.mark.parametrize('use_real', [False, True])
@pytest.mark.parametrize('phase_type', ['full', 'reduced'])
def test_solve_hamiltonian_simulation_example(method, use_real, phase_type):
    """Each solver method should reproduce a documented QSP instance."""
    tau = 10
    degree = 60
    target = lambda x: 0.5 * np.cos(tau * x)
    coefficients = np.polynomial.chebyshev.chebinterpolate(target, degree)
    reduced_coefficients = coefficients[::2]
    opts = {
        'maxiter': 200,
        'criteria': 1e-10,
        'useReal': use_real,
        'targetPre': True,
        'method': method,
        'typePhi': phase_type,
        'print': False,
    }

    phases, out = solve(reduced_coefficients, 0, opts)
    grid = np.linspace(-1.0, 1.0, 501)
    expected = np.polynomial.chebyshev.chebval(grid, coefficients)
    actual = get_entry(grid, phases, out)

    assert np.all(np.isfinite(phases))
    assert np.linalg.norm(actual - expected, np.inf) < 1e-8


@pytest.mark.parametrize('method', ['FPI', 'Newton', 'LBFGS', 'NLFT'])
@pytest.mark.parametrize('parity', [0, 1])
@pytest.mark.parametrize('target_pre', [False, True])
@pytest.mark.parametrize('phase_type', ['full', 'reduced'])
def test_solve_phase_conventions(method, parity, target_pre, phase_type):
    """Every solver should obey the public phase and target conventions."""
    reduced_coefficients = np.array([0.2, 0.1])
    phases, out = solve(reduced_coefficients, parity, {
        'method': method,
        'N': 256,
        'maxiter': 200,
        'criteria': 1e-10,
        'useReal': True,
        'targetPre': target_pre,
        'typePhi': phase_type,
        'print': False,
    })
    grid = np.linspace(-1.0, 1.0, 101)
    full_coefficients = np.zeros(2 * len(reduced_coefficients) - 1 + parity)
    full_coefficients[parity::2] = reduced_coefficients

    actual = get_entry(grid, phases, out)
    expected = np.polynomial.chebyshev.chebval(grid, full_coefficients)

    expected_phase_count = (
        len(full_coefficients) if phase_type == 'full'
        else len(reduced_coefficients)
    )
    assert len(phases) == expected_phase_count
    assert not np.iscomplexobj(phases)
    if method == 'NLFT':
        assert out['iter'] == 1
        assert out['value'] < 1e-12
    assert np.linalg.norm(actual - expected, np.inf) < 1e-8

def test_solve_gibbs():
    """Every solver should reproduce the documented Gibbs polynomial."""
    # set parameters for polynomial approximation
    beta = 2
    targ = lambda x: np.exp(-beta * x)
    deg = 151
    parity = deg % 2
    delta = 0.2

    # options for cvx_poly_coef
    opts = {
        'intervals': [delta, 1],
        'objnorm': 2,
        'epsil': 0.2,
        'npts': 500,
        'fscale': 1,
        'isplot': False,
        'method': 'cvxpy',
        'maxiter': 100
    }

    coef_full = cvx_poly_coef(targ, deg, opts)
    coef = coef_full[parity::2]
    grid = np.linspace(-1.0, 1.0, 301)
    expected = np.polynomial.chebyshev.chebval(grid, coef_full)

    for method in ('FPI', 'Newton', 'LBFGS', 'NLFT'):
        phases, out = solve(coef, parity, {
            'method': method,
            'maxiter': 200,
            'criteria': 1e-10,
            'useReal': True,
            'targetPre': True,
            'typePhi': 'full',
            'print': False,
            'N': 1024,
        })
        actual = get_entry(grid, phases, out)

        assert np.all(np.isfinite(phases)), method
        assert np.linalg.norm(actual - expected, np.inf) < 1e-8, method
