"""Optimization methods for Quantum Signal Processing.

This module provides various optimization algorithms for finding phase factors
in Quantum Signal Processing problems.
"""

import numpy as np
import time
from .utils import F, F_Jacobian
from .nlfa import (
    b_from_cheb,
    weiss,
    inverse_nonlinear_FFT,
    forward_nonlinear_FFT,
)

def lbfgs(obj, grad, delta, phi, opts):
    """L-BFGS optimization for QSP phase factors.

    This function implements the Limited-memory BFGS optimization algorithm
    for finding optimal phase factors in QSP problems.

    Parameters
    ----------
    obj : callable
        Objective function to minimize
    grad : callable
        Gradient function of the objective
    delta : array_like
        Sample points passed to ``obj`` and ``grad``
    phi : array_like
        Initial phase factors
    opts : dict
        Options dictionary containing:
        
        - maxiter : int
            Maximum number of iterations
        - criteria : float
            Convergence criteria
        - gamma : float
            Line search retraction rate (default 0.5)
        - accrate : float
            Line search accept ratio (default 1e-3)
        - minstep : float
            Minimal step size (default 1e-5)
        - lmem : int
            L-BFGS memory size (default 200)
        - print : bool
            Whether to print progress (default True)
        - itprint : int
            Print frequency (default 1)
        - parity : int, optional
            Parity of polynomial (0 for even, 1 for odd). When omitted, the
            generic odd-parity initial inverse-Hessian scaling is used.

    Returns
    -------
    phi : ndarray
        Optimized phase factors
    obj_value : float
        Objective value at optimal point
    iter : int
        Number of iterations performed
    """
    # Options for L-BFGS solver
    opts.setdefault('maxiter', 50000)
    opts.setdefault('gamma', 0.5)
    opts.setdefault('accrate', 1e-3)
    opts.setdefault('minstep', 1e-5)
    opts.setdefault('criteria', 1e-12)
    opts.setdefault('lmem', 200)
    opts.setdefault('print', 1)
    opts.setdefault('itprint', 1)

    # Copy value to parameters
    maxiter = int(opts['maxiter'])
    gamma = opts['gamma']
    accrate = opts['accrate']
    lmem = int(opts['lmem'])
    minstep = opts['minstep']
    pri = opts['print']
    itprint = opts['itprint']
    crit = opts['criteria']

    # Setup print format
    str_head = "{:4s} {:13s} {:10s} {:10s}\n".format('iter', 'obj', 'stepsize', 'des_ratio')
    str_num = "{:4d}  {:+5.4e} {:+3.2e} {:+3.2e}\n"

    if maxiter < 1:
        raise ValueError("maxiter must be positive")
    if lmem < 1:
        raise ValueError("lmem must be positive")

    # Initial computation
    phi = np.asarray(phi, dtype=float).copy()
    iter = 0
    d = len(phi)
    mem_size = 0
    # Zero-based index of the most recently stored correction pair. The
    # MATLAB reference starts at zero and increments before its first write;
    # therefore the faithful zero-based translation starts at -1.
    mem_now = -1
    mem_grad = np.zeros((lmem, d))
    mem_obj = np.zeros((lmem, d))
    mem_dot = np.zeros(lmem)
    grad_s, obj_s = grad(phi, delta, opts)
    obj_value = float(np.mean(obj_s))
    GRAD = np.mean(grad_s, axis=0)

    # Start L-BFGS algorithm
    if pri:
        print('L-BFGS solver started')

    while True:
        iter += 1
        theta_d = GRAD.copy()
        alpha = np.zeros(mem_size)
        for i in range(mem_size):
            # Traverse correction pairs from newest to oldest.
            subsc = (mem_now - i) % lmem
            alpha[i] = mem_dot[subsc] * np.dot(mem_obj[subsc, :], theta_d)
            theta_d -= alpha[i] * mem_grad[subsc, :]

        theta_d *= 0.5
        if opts.get('parity') == 0:
            theta_d[0] *= 2

        for i in range(mem_size):
            # Complete the two-loop recursion from oldest to newest.
            subsc = (mem_now - (mem_size - 1 - i)) % lmem
            beta = mem_dot[subsc] * np.dot(mem_grad[subsc, :], theta_d)
            theta_d += (alpha[mem_size - i - 1] - beta) * mem_obj[subsc, :]

        step = 1
        exp_des = np.dot(GRAD, theta_d)
        if not np.isfinite(exp_des) or exp_des <= 0:
            # Discard unusable history and fall back to the reference's
            # initial inverse-Hessian scaling.
            mem_size = 0
            theta_d = 0.5 * GRAD
            if opts.get('parity') == 0:
                theta_d[0] *= 2
            exp_des = np.dot(GRAD, theta_d)

        phi_old = phi.copy()
        while True:
            theta_new = phi - step * theta_d
            obj_snew = obj(theta_new, delta, opts)
            obj_valuenew = float(np.mean(obj_snew))
            ad = obj_value - obj_valuenew
            if ad > exp_des * accrate * step or step < minstep:
                break
            step *= gamma

        phi = theta_new
        obj_value = obj_valuenew
        obj_max = np.max(obj_snew)
        grad_s, _ = grad(phi, delta, opts)
        GRAD_new = np.mean(grad_s, axis=0)
        grad_delta = GRAD_new - GRAD
        iterate_delta = phi - phi_old
        curvature = np.dot(grad_delta, iterate_delta)
        if np.isfinite(curvature) and curvature > np.finfo(float).eps:
            mem_now = (mem_now + 1) % lmem
            mem_grad[mem_now, :] = grad_delta
            mem_obj[mem_now, :] = iterate_delta
            mem_dot[mem_now] = 1 / curvature
            mem_size = min(lmem, mem_size + 1)
        GRAD = GRAD_new

        if pri and iter % itprint == 0:
            if iter == 1 or (iter - itprint) % (itprint * 10) == 0:
                print(str_head, end='')
            descent_ratio = ad / (exp_des * step) if exp_des else np.nan
            print(str_num.format(iter, obj_max, step, descent_ratio), end='')

        if obj_max < crit**2:
            if pri:
                print("Stop criteria satisfied.")
            break
        if iter >= maxiter:
            if pri:
                print("Max iteration reached.")
            break

    return phi, obj_value, iter

def coordinate_minimization(coef, parity, opts):
    """Fixed-point iteration for symmetric QSP phase factors.

    The historical public name is retained for compatibility. The algorithm
    is the contraction mapping used by QSPPACK's ``FPI`` solver,
    ``phi <- phi - (F(phi) - coef) / 2``; it is not coordinate descent.

    Parameters
    ----------
    coef : array_like
        Coefficients of polynomial P under Chebyshev basis
    parity : int
        Parity of polynomial P (0 for even, 1 for odd)
    opts : dict
        Options dictionary containing optimization parameters

    Returns
    -------
    phi : ndarray
        Optimized phase factors
    err : float
        Final error value
    iter : int
        Number of iterations performed
    runtime : float
        Total runtime in seconds
    """
    # Setup options for CM solver
    opts.setdefault('maxiter', int(1e5))
    opts.setdefault('criteria', 1e-12)
    opts.setdefault('targetPre', True)
    opts.setdefault('useReal', True)
    opts.setdefault('print', 1)
    opts.setdefault('itprint', 1)

    start_time = time.time()

    # Copy value to parameters
    maxiter = int(opts['maxiter'])
    crit = opts['criteria']
    pri = opts['print']
    itprint = opts['itprint']

    # Setup print format
    str_head = "{:4s} {:13s}\n".format('iter', 'err')
    str_num = "{:4d}  {:+5.4e}\n"

    # Initial preparation
    coef = np.asarray(coef, dtype=float).copy()
    if opts['targetPre']:
        coef = -coef  # inverse is necessary
    phi = coef / 2
    iter = 0

    # Solve by contraction mapping algorithm
    while True:
        Fval = F(phi, parity, opts)
        res = Fval - coef
        err = float(np.linalg.norm(res, 1))
        iter += 1
        if err < crit:
            if pri:
                print("Stop criteria satisfied.")
            break
        if iter >= maxiter:
            if pri:
                print("Max iteration reached.")
            break
        phi = phi - res / 2
        if pri and iter % itprint == 0:
            if iter == 1 or (iter - itprint) % (itprint * 10) == 0:
                print(str_head, end='')
            print(str_num.format(iter, err), end='')

    runtime = time.time() - start_time
    return phi, err, iter, runtime

def newton(coef, parity, opts):
    """Newton's method optimization for QSP phase factors.

    This function implements Newton's method for finding optimal phase
    factors in QSP problems.

    Parameters
    ----------
    coef : array_like
        Coefficients of polynomial P under Chebyshev basis
    parity : int
        Parity of polynomial P (0 for even, 1 for odd)
    opts : dict
        Options dictionary containing optimization parameters

    Returns
    -------
    phi : ndarray
        Optimized phase factors
    err : float
        Final error value
    iter : int
        Number of iterations performed
    runtime : float
        Total runtime in seconds
    """
    # Setup options for Newton solver
    opts.setdefault('maxiter', int(1e5))
    opts.setdefault('criteria', 1e-12)
    opts.setdefault('targetPre', True)
    opts.setdefault('useReal', True)
    opts.setdefault('print', 1)
    opts.setdefault('itprint', 1)

    start_time = time.time()

    # Copy value to parameters
    maxiter = int(opts['maxiter'])
    crit = opts['criteria']
    pri = opts['print']
    itprint = opts['itprint']

    # Setup print format
    str_head = "{:4s} {:13s}\n".format('iter', 'err')
    str_num = "{:4d}  {:+5.4e}\n"

    # Initial preparation
    coef = np.asarray(coef, dtype=float).copy()
    if opts['targetPre']:
        coef = -coef  # inverse is necessary
    phi = coef / 2
    iter = 0

    # Solve by Newton's method
    while True:
        Fval, DFval = F_Jacobian(phi, parity, opts)
        res = Fval - coef
        err = float(np.linalg.norm(res, 1))
        iter += 1
        if err < crit:
            if pri:
                print("Stop criteria satisfied.")
            break
        if iter >= maxiter:
            if pri:
                print("Max iteration reached.")
            break
        phi = phi - np.linalg.solve(DFval, res)
        if pri and iter % itprint == 0:
            if iter == 1 or (iter - itprint) % (itprint * 10) == 0:
                print(str_head, end='')
            print(str_num.format(iter, err), end='')

    runtime = time.time() - start_time
    return phi, err, iter, runtime

def nlft(coef, parity, opts):
    """Direct phase synthesis through the inverse nonlinear Fourier transform.

    This function computes a complementary polynomial with the Weiss
    factorization and recovers phase factors with the inverse NLFT. Unlike
    the iterative solvers, it performs one direct synthesis pass.

    Parameters
    ----------
    coef : array_like
        Coefficients of polynomial P under Chebyshev basis
    parity : int
        Parity of polynomial P (0 for even, 1 for odd)
    opts : dict
        Options dictionary. ``N`` is the even FFT length used by the Weiss
        factorization (default 256), and ``targetPre`` selects a real-part
        target when true (default true).

    Returns
    -------
    phi : ndarray
        Reduced symmetric phase factors, in the convention expected by
        :func:`qsppack.utils.reduced_to_full`.
    err : float
        One-norm reconstruction residual of the NLFT polynomial coefficients.
    iter : int
        One, because NLFT is a direct method.
    runtime : float
        Total runtime in seconds.
    """
    opts.setdefault('N', 256)
    opts.setdefault('targetPre', True)

    start_time = time.time()
    coef = np.asarray(coef, dtype=float).copy()
    if opts['targetPre']:
        # The NLFT correspondence produces the target in Im(U[0, 0]).
        # Negating it here and applying reduced_to_full's +pi/4 endpoint
        # shifts rotates that component into Re(U[0, 0]).
        coef = -coef
    b_coeffs = b_from_cheb(coef, parity)
    a_coeffs = weiss(b_coeffs, opts['N'])
    gammas, _, _ = inverse_nonlinear_FFT(a_coeffs, b_coeffs)
    _, reconstructed_b = forward_nonlinear_FFT(gammas)
    err = float(np.linalg.norm(reconstructed_b - b_coeffs, 1))

    gammas = np.real_if_close(gammas, tol=1000)
    if np.iscomplexobj(gammas):
        raise RuntimeError("NLFT produced materially complex phase parameters")
    full_phases = np.arctan(np.asarray(gammas, dtype=float))

    # inverse_nonlinear_FFT returns the complete symmetric phase sequence,
    # whereas solve() and the other optimizers exchange reduced phases.
    phis = full_phases[-len(coef):].copy()
    if parity == 0:
        phis[0] /= 2
    runtime = time.time() - start_time
    return phis, err, 1, runtime
