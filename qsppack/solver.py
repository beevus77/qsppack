"""Main solver interface for Quantum Signal Processing optimization.

This module provides the main interface for solving QSP optimization problems,
coordinating the various optimization methods and utility functions.
"""

import numpy as np
from time import time
from .utils import chebyshev_to_func, reduced_to_full
from .objective import obj_sym, grad_sym, grad_sym_real
from .optimizers import lbfgs, coordinate_minimization, newton, nlft

def solve(coef, parity, opts=None):
    """Given coefficients of a polynomial P, yield corresponding phase factors.

    The reference chose the first half of the phase factors as the 
    optimization variables, while in the code we used the second half of the 
    phase factors. These two formulations are equivalent.

    To simplify the representation, a constant pi/4 is added to both sides of 
    the phase factors when evaluating the objective and the gradient. In the
    output, the FULL phase factors with pi/4 are given.

    Parameters
    ----------
    coef : array_like
        Coefficients of polynomial P under Chebyshev basis. P should be even/odd,
        only provide non-zero coefficients. Coefficients should be ranked from
        low order term to high order term.
    parity : int
        Parity of polynomial P (0 -- even, 1 -- odd)
    opts : dict, optional
        Options dictionary with fields:
        
        - criteria : float
            Stop criteria
        - useReal : bool
            Use only real arithmetics if true
        - targetPre : bool
            Want Pre to be target function if true
        - method : {'LBFGS', 'FPI', 'Newton', 'NLFT'}
            Optimization method to use
        - typePhi : {'full', 'reduced'}
            Type of phase factors to return

    Returns
    -------
    phi_proc : ndarray
        Solution of optimization problem, FULL phase factors
    out : dict
        Information of solving process containing:
        
        - iter : int
            Number of iterations
        - time : float
            Runtime in seconds
        - value : float
            Final error value
        - parity : int
            Input parity value
        - targetPre : bool
            Whether Pre was target function
        - typePhi : str
            Type of phase factors returned
        - method : str
            Optimization method used
        - converged : bool
            Whether the method satisfied its numerical stopping criterion
    """
    if opts is None:
        opts = {}
    elif not isinstance(opts, dict):
        raise TypeError("opts must be a dictionary or None")
    else:
        # Low-level solvers add method-specific defaults, so isolate those
        # updates from the caller's dictionary.
        opts = opts.copy()

    coef = np.asarray(coef)
    if coef.ndim != 1 or coef.size == 0:
        raise ValueError("coef must be a nonempty one-dimensional array")
    if np.iscomplexobj(coef) and np.any(np.imag(coef) != 0):
        raise ValueError("coef must contain real Chebyshev coefficients")
    coef = np.asarray(np.real(coef), dtype=float)
    if not np.all(np.isfinite(coef)):
        raise ValueError("coef must contain only finite values")
    if parity not in (0, 1) or isinstance(parity, (bool, np.bool_)):
        raise ValueError("parity must be zero (even) or one (odd)")

    # Setup options for L-BFGS solver
    opts.setdefault('maxiter', 50000)
    opts.setdefault('criteria', 1e-12)
    opts.setdefault('useReal', True)
    opts.setdefault('targetPre', True)
    opts.setdefault('method', 'FPI')
    opts.setdefault('typePhi', 'full')

    if (
        isinstance(opts['maxiter'], (bool, np.bool_))
        or int(opts['maxiter']) != opts['maxiter']
        or int(opts['maxiter']) < 1
    ):
        raise ValueError("maxiter must be a positive integer")
    opts['maxiter'] = int(opts['maxiter'])
    if not np.isfinite(opts['criteria']) or opts['criteria'] <= 0:
        raise ValueError("criteria must be positive and finite")
    for name in ('useReal', 'targetPre'):
        if not isinstance(opts[name], (bool, np.bool_)):
            raise TypeError(f"{name} must be a boolean")
    methods = ('LBFGS', 'FPI', 'Newton', 'NLFT')
    if opts['method'] not in methods:
        raise ValueError(f"method must be one of {methods}")
    if opts['typePhi'] not in ('full', 'reduced'):
        raise ValueError("typePhi must be 'full' or 'reduced'")

    if opts['method'] == 'LBFGS':
        # Initial preparation
        tot_len = len(coef)
        # Roots of T_{2 * tot_len}. The MATLAB expression is
        # ``(1:2:2*d-1) * (pi/2/(2*d))``; preserving both divisions gives
        # pi/(4*d), not pi/(2*d).
        delta = np.cos(np.arange(1, 2 * tot_len, 2) * np.pi / (4 * tot_len))
        if not opts['targetPre']:
            opts['target'] = lambda x: -chebyshev_to_func(x, coef, parity, True)
        else:
            opts['target'] = lambda x: chebyshev_to_func(x, coef, parity, True)
        opts['parity'] = parity
        obj = obj_sym
        grad = grad_sym_real if opts['useReal'] else grad_sym

        # Solve by L-BFGS with selected initial point
        start_time = time()
        phi, err, iter = lbfgs(obj, grad, delta, np.zeros(tot_len), opts)
        # Convert phi to reduced phase factors
        if parity == 0:
            phi[0] = phi[0] / 2
        runtime = time() - start_time

    elif opts['method'] == 'FPI':
        phi, err, iter, runtime = coordinate_minimization(coef, parity, opts)

    elif opts['method'] == 'Newton':
        phi, err, iter, runtime = newton(coef, parity, opts)

    elif opts['method'] == 'NLFT':
        phi, err, iter, runtime = nlft(coef, parity, opts)

    # Output information
    threshold = opts['criteria'] ** 2 if opts['method'] == 'LBFGS' else opts['criteria']
    out = {
        'iter': iter,
        'time': runtime,
        'value': err,
        'parity': parity,
        'targetPre': opts['targetPre'],
        'method': opts['method'],
        'converged': bool(err < threshold),
    }

    if opts['typePhi'] == 'full':
        phi_proc = reduced_to_full(phi, parity, opts['targetPre'])
        out['typePhi'] = 'full'
    elif opts['typePhi'] == 'reduced':
        phi_proc = phi
        out['typePhi'] = 'reduced'

    return phi_proc, out
