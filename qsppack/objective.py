"""Objective and gradient functions for QSP optimization.

This module provides functions for computing objective values and gradients
needed in QSP optimization problems.
"""

import numpy as np
from .utils import get_unitary_sym, get_pim_deri_sym, get_pim_deri_sym_real

def obj_sym(phi, delta, opts):
    """Compute objective function value for QSP optimization.

    Parameters
    ----------
    phi : array_like
        Phase factors for QSP circuit
    delta : array_like
        Samples
    opts : dict
        Options dictionary containing target function and parameters

    Returns
    -------
    ndarray
        Pointwise half-squared objective values, one per sample
    """
    m = len(delta)
    obj = np.zeros(m)
    for i in range(m):
        qspmat = get_unitary_sym(phi, delta[i], opts['parity'])
        target_value = np.asarray(opts['target'](delta[i])).item()
        obj[i] = 0.5 * (np.real(qspmat[0, 0]) - target_value)**2

    return obj

def _grad_sym(phi, delta, opts, derivative):
    """Evaluate the symmetric-QSP objective gradient with ``derivative``."""
    m = len(delta)
    d = len(phi)
    obj = np.zeros(m)
    grad = np.zeros((m, d))
    targetx = opts['target']
    parity = opts['parity']

    # MATLAB uses copy-on-write when converting the even-parity L-BFGS
    # representation to reduced phases. Make that copy explicit for NumPy.
    phi_eval = np.asarray(phi).copy()
    if parity == 0:
        phi_eval[0] /= 2

    for i, x in enumerate(delta):
        value_and_derivative = derivative(phi_eval, x, parity)
        if parity == 0:
            value_and_derivative[0] /= 2
        value_and_derivative = -value_and_derivative
        gap = value_and_derivative[-1] - np.asarray(targetx(x)).item()
        obj[i] = 0.5 * gap**2
        grad[i, :] = value_and_derivative[:-1] * gap

    return grad, obj


def grad_sym(phi, delta, opts):
    """Compute the symmetric-QSP gradient using complex arithmetic.

    Parameters
    ----------
    phi : array_like
        Phase factors for QSP circuit
    delta : array_like
        Samples
    opts : dict
        Options dictionary containing target function and parameters

    Returns
    -------
    grad : ndarray
        Gradient of objective function
    obj : ndarray
        Objective function value
    """
    return _grad_sym(phi, delta, opts, get_pim_deri_sym)

def grad_sym_real(phi, delta, opts):
    """Compute gradient using real arithmetic.

    Similar to grad_sym but uses only real arithmetic for efficiency.

    Parameters
    ----------
    phi : array_like
        Phase factors for QSP circuit
    delta : array_like
        Samples
    opts : dict
        Options dictionary containing target function and parameters

    Returns
    -------
    grad : ndarray
        Gradient of objective function
    obj : ndarray
        Objective function value
    """
    return _grad_sym(phi, delta, opts, get_pim_deri_sym_real)
