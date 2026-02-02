"""Convex polynomial coefficient helper using CVXPY only."""

import numpy as np
import cvxpy as cp
from scipy.special import chebyt


def cvx_poly_coef2(func, deg, opts):
    """Compute Chebyshev coefficients via convex optimization (CVXPY only).

    Parameters
    ----------
    func : callable
        Target function to approximate.
    deg : int
        Polynomial degree.
    opts : dict
        Options dictionary with fields:
        - intervals : list
            [min, max] interval(s) for x values.
        - constraint_intervals : list
            [min, max] interval(s) where constraints are enforced.
        - npts : int
            Number of points to evaluate the function at.
        - objnorm : float
            Norm to use for error reporting (np.inf typical).
        - epsil : float
            Epsilon for numerical stability.
        - fscale : float
            Scale factor for function values.

    Returns
    -------
    ndarray
        Chebyshev coefficients of the best-fit polynomial.
    """
    opts.setdefault("npts", 200)
    opts.setdefault("epsil", 0.01)
    opts.setdefault("fscale", 1 - opts["epsil"])
    opts.setdefault("intervals", [0, 1])
    opts.setdefault("constraint_intervals", opts["intervals"])
    opts.setdefault("objnorm", np.inf)

    assert len(opts["intervals"]) % 2 == 0
    assert len(opts["constraint_intervals"]) % 2 == 0
    parity = deg % 2
    epsil = opts["epsil"]
    npts = opts["npts"]

    # Generate Chebyshev points
    xpts = np.cos(np.pi * np.arange(2 * npts) / (2 * npts - 1))
    xpts = np.union1d(xpts, opts["intervals"])
    xpts = xpts[xpts >= 0]
    npts = len(xpts)

    n_interval = len(opts["intervals"]) // 2
    n_constraint_interval = len(opts["constraint_intervals"]) // 2
    ind_union = np.array([], dtype=int)
    ind_constraint = np.array([], dtype=int)

    for i in range(n_interval):
        ind = np.where(
            (xpts >= opts["intervals"][2 * i])
            & (xpts <= opts["intervals"][2 * i + 1])
        )[0]
        ind_union = np.union1d(ind_union, ind)

    for i in range(n_constraint_interval):
        ind = np.where(
            (xpts >= opts["constraint_intervals"][2 * i])
            & (xpts <= opts["constraint_intervals"][2 * i + 1])
        )[0]
        ind_constraint = np.union1d(ind_constraint, ind)

    # Evaluate the target function
    fx = np.zeros(npts)
    fx[ind_union] = opts["fscale"] * func(xpts[ind_union])

    # Prepare the Chebyshev polynomials
    n_coef = deg // 2 + 1 if parity == 0 else (deg + 1) // 2
    Ax = np.zeros((npts, n_coef))

    for k in range(1, n_coef + 1):
        Tcheb = chebyt(2 * (k - 1)) if parity == 0 else chebyt(2 * k - 1)
        Ax[:, k - 1] = Tcheb(xpts)

    # CVXPY optimization
    c = cp.Variable(n_coef)
    y = Ax @ c
    residual = y[ind_union] - fx[ind_union]
    objective = cp.Minimize(cp.norm_inf(residual))
    constraints = [
        y[ind_constraint] <= 1 - epsil,
        y[ind_constraint] >= -(1 - epsil),
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    coef = c.value

    err_inf = np.linalg.norm((Ax @ coef)[ind_union] - fx[ind_union], opts["objnorm"])
    print(f"norm error = {err_inf}")

    # Make sure the maximum is less than 1
    coef_full = np.zeros(deg + 1)
    if parity == 0:
        coef_full[::2] = coef
    else:
        coef_full[1::2] = coef

    max_sol = np.max(np.abs(np.polynomial.chebyshev.chebval(xpts, coef_full)))
    print(f"max of solution = {max_sol}")

    return coef_full