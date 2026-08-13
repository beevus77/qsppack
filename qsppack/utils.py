"""Utility functions for Quantum Signal Processing.

This module provides utility functions for working with Chebyshev polynomials,
unitary matrix construction, and other mathematical operations needed in QSP.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, linprog
from scipy.special import chebyt
import cvxpy as cp

def get_unitary(phase, x):
    """Compute QSP unitary matrix for given phase factors.

    This function constructs the full QSP unitary matrix for a given set of phase
    factors at a specific point. The unitary is built by alternating W(x) gates
    and phase rotations.

    Parameters
    ----------
    phase : array_like
        Phase factors for the QSP circuit. These are used to construct the
        phase rotation gates in the circuit.
    x : float
        Point at which to evaluate the unitary, must be in [-1, 1].

    Returns
    -------
    float
        Real part of the (1,1) element of the QSP unitary matrix, which
        represents the QSP approximation of the target function.

    Notes
    -----
    The unitary is constructed as:
    U = R(phi_0) * W(x) * R(phi_1) * W(x) * ... * R(phi_n)
    where R(phi) is a phase rotation gate and W(x) is the signal processing gate.
    """
    Wx = np.array([[x, 1j * np.sqrt(1 - x**2)], [1j * np.sqrt(1 - x**2), x]])
    expphi = np.exp(1j * phase)

    ret = np.array([[expphi[0], 0], [0, np.conj(expphi[0])]])

    for k in range(1, len(expphi)):
        temp = np.array([[expphi[k], 0], [0, np.conj(expphi[k])]])
        ret = np.dot(np.dot(ret, Wx), temp)

    targ = np.real(ret[0, 0])

    return targ

def get_entry(xlist, phase, opts):
    """Compute QSP unitary matrix entries for multiple points.

    This function evaluates the QSP unitary matrix at multiple points, handling
    both full and reduced phase factors. It can compute either the real or
    imaginary part of the (1,1) element based on the options.

    Parameters
    ----------
    xlist : array_like
        Points at which to evaluate the QSP unitary, must be in [-1, 1].
    phase : array_like
        Phase factors for the QSP circuit. Can be either full or reduced
        phase factors depending on opts['typePhi'].
    opts : dict
        Options dictionary containing:
        
        - targetPre : bool
            If True, compute real part (Pre), otherwise compute imaginary part (Pim)
        - parity : int
            Parity of the polynomial (0 for even, 1 for odd)
        - typePhi : {'full', 'reduced'}
            Type of phase factors provided:
            
            - 'full' : complete set of phase factors
            - 'reduced' : reduced set of phase factors (will be expanded)

    Returns
    -------
    ndarray
        QSP approximation values at each point in xlist. For targetPre=True,
        returns real part of (1,1) element; for targetPre=False, returns
        imaginary part.

    Notes
    -----
    When typePhi='reduced', the function expands the reduced phase factors to
    full phase factors using symmetry. For targetPre=False, it also adjusts
    the first and last phase factors by -π/4 to compute the imaginary part.
    """
    typePhi = opts['typePhi']
    targetPre = opts['targetPre']
    parity = opts['parity']

    d = len(xlist)
    ret = np.zeros(d)

    if typePhi == 'reduced':
        dd = 2 * len(phase) - 1 + parity
        phi = np.zeros(dd)
        phi[(dd - len(phase)):] = phase
        phi[:len(phase)] += phase[::-1]
    else:
        phi = phase

    if not targetPre:
        phi[0] -= np.pi / 4
        phi[-1] -= np.pi / 4

    for i in range(d):
        x = xlist[i]
        ret[i] = get_unitary(phi, x)

    return ret

def reduced_to_full(phi_cm, parity, targetPre):
    """Convert reduced phase factors to full phase factors for QSP.

    This function constructs the full set of phase factors required for the
    Quantum Signal Processing (QSP) unitary matrix from a reduced set. The
    conversion uses symmetry properties of the phase factors and handles
    both even and odd parity cases.

    Parameters
    ----------
    phi_cm : array_like
        Reduced phase factors. For even parity, these represent half the
        total phase factors; for odd parity, they represent the unique
        phase factors.
    parity : int
        Parity of the phase factors:
        
        - 0 : even parity (full length = 2*len(phi_cm) - 1)
        - 1 : odd parity (full length = 2*len(phi_cm))
    targetPre : bool
        Whether to adjust for target preparation:
        
        - True : add π/4 to the last phase factor
        - False : use phase factors as is

    Returns
    -------
    ndarray
        Full phase factors constructed by mirroring the reduced factors.
        The length depends on parity:
        
        - For even parity: 2*len(phi_cm) - 1
        - For odd parity: 2*len(phi_cm)

    Notes
    -----
    The full phase factors are constructed by:
    1. Copying the reduced factors to the right half
    2. Mirroring them to the left half
    3. Adjusting the last factor if targetPre is True
    """
    phi_right = phi_cm.copy()
    if targetPre:
        phi_right[-1] += np.pi / 4

    dd = 2 * len(phi_right)
    if parity == 0:
        dd -= 1

    phi_full = np.zeros(dd)
    phi_full[(dd - len(phi_right)):] = phi_right
    phi_full[:len(phi_right)] += phi_right[::-1]

    return phi_full

def chebyshev_to_func(x, coef, parity, partialcoef):
    """Convert Chebyshev coefficients to function values.

    This function evaluates a polynomial represented in the Chebyshev basis
    at given points.

    Parameters
    ----------
    x : array_like or float
        Points at which to evaluate the polynomial. Can be a single point
        or an array of points.
    coef : array_like
        Coefficients in Chebyshev basis, ordered from lowest to highest degree
    parity : int
        Parity of the polynomial (0 for even, 1 for odd)
    partialcoef : bool, optional
        Whether to return only coefficients of odd/even order

    Returns
    -------
    ndarray or float
        Function values at the given points. Returns a scalar if input is
        scalar, array otherwise.
    """
    ret = np.zeros(len(x))
    y = np.arccos(x)
    if partialcoef:
        if parity == 0:
            for k in range(len(coef)):
                ret += coef[k] * np.cos(2 * k * y)
        else:
            for k in range(len(coef)):
                ret += coef[k] * np.cos((2 * k + 1) * y)
    else:
        if parity == 0:
            for k in range(0, len(coef), 2):
                ret += coef[k] * np.cos(k * y)
        else:
            for k in range(1, len(coef), 2):
                ret += coef[k] * np.cos(k * y)
    return ret

def cvx_poly_coef(func, deg, opts):
    """Compute coefficients for a polynomial approximation using convex optimization.

    This function computes the coefficients of a polynomial that best approximates
    a target function over specified intervals in a least-squares sense. The function
    is called with a target function, the degree of the polynomial, and an options
    dictionary.

    Parameters
    ----------
    func : callable
        The target function to approximate.
    deg : int
        The degree of the polynomial.
    opts : dict
        Options dictionary with the following fields:
        
        - intervals : list
            [min, max] interval for x values.
        - npts : int
            Number of points to evaluate the function at.
        - objnorm : float
            Norm to use for optimization.
        - epsil : float
            Epsilon for numerical stability.
        - fscale : float
            Scale factor for function values.
        - isplot : bool
            Whether to plot results.

    Returns
    -------
    ndarray
        Coefficients of the best-fit polynomial in the Chebyshev basis.
    """
    # Set default options if not provided
    opts.setdefault('npts', 200)
    opts.setdefault('epsil', 0.01)
    opts.setdefault('fscale', 1 - opts['epsil'])
    opts.setdefault('intervals', [0, 1])
    opts.setdefault('isplot', False)
    opts.setdefault('objnorm', np.inf)
    opts.setdefault('method', 'SLSQP')
    opts.setdefault('solver', 'CLARABEL')
    opts.setdefault('verbose', False)

    # Check variables and assign local variables
    assert len(opts['intervals']) % 2 == 0
    parity = deg % 2
    epsil = opts['epsil']
    npts = opts['npts']

    # Generate Chebyshev points
    xpts = np.cos(np.pi * np.arange(2 * npts) / (2 * npts - 1))
    xpts = np.union1d(xpts, opts['intervals'])
    xpts = xpts[xpts >= 0]
    npts = len(xpts)

    n_interval = len(opts['intervals']) // 2
    ind_union = np.array([], dtype=int)
    ind_set = []

    for i in range(n_interval):
        ind = np.where((xpts >= opts['intervals'][2 * i]) & (xpts <= opts['intervals'][2 * i + 1]))[0]
        ind_set.append(ind)
        ind_union = np.union1d(ind_union, ind)

    # Evaluate the target function
    fx = np.zeros(npts)
    fx[ind_union] = opts['fscale'] * func(xpts[ind_union])

    # Prepare the Chebyshev polynomials
    n_coef = deg // 2 + 1 if parity == 0 else (deg + 1) // 2
    Ax = np.zeros((npts, n_coef))

    for k in range(1, n_coef + 1):
        Tcheb = chebyt(2 * (k - 1)) if parity == 0 else chebyt(2 * k - 1)
        Ax[:, k-1] = Tcheb(xpts)

    # Use optimization to find the Chebyshev coefficients
    coef = np.zeros(n_coef)
    if opts['method'] == 'SLSQP':
        def objective(coef):
            y = Ax @ coef
            return np.linalg.norm(y[ind_union] - fx[ind_union], opts['objnorm'])

        constraints = [{'type': 'ineq', 'fun': lambda coef: 1 - epsil - Ax @ coef},
                    {'type': 'ineq', 'fun': lambda coef: Ax @ coef + (1 - epsil)}]

        result = minimize(objective, np.zeros(n_coef), constraints=constraints)
        coef = result.x
    
    elif opts['method'] == 'cvxpy':
        c = cp.Variable(n_coef)
        y = Ax @ c
        residual = y[ind_union] - fx[ind_union]
        objective = cp.Minimize(cp.norm_inf(residual))
        constraints = [
            y <= 1 - epsil,
            y >= -(1-epsil)
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=opts['solver'], verbose=opts['verbose'])
        coef = c.value

    elif opts['method'] == 'linprog':
        e0 = np.zeros(n_coef+1)
        e0[0] = 1
        A_prime = Ax[ind_union, :]

        # Build A_ub and b_ub
        neg_one = -np.ones((A_prime.shape[0], 1))
        zero_one = np.zeros((Ax.shape[0], 1))

        A_ub = np.vstack([
            np.hstack([neg_one, A_prime]),       # A'c - t <= f
            np.hstack([neg_one, -A_prime]),      # -A'c - t <= -f
            np.hstack([zero_one, Ax]),            # A c <= 1 - epsil
            np.hstack([zero_one, -Ax])            # -A c <= 1 - epsil
        ])

        b_ub = np.concatenate([
            fx[ind_union],
            -fx[ind_union],
            (1 - epsil) * np.ones(Ax.shape[0]),
            (1 - epsil) * np.ones(Ax.shape[0])
        ])

        # Solve
        result = linprog(c=e0, A_ub=A_ub, b_ub=b_ub, method='highs')

        if result.success:
            coef = result.x[1:n_coef+1]
        else:
            raise ValueError(f'Linear programming failed to find an optimal solution, status: {result.status}')

    else:
        raise ValueError(f'Method {opts["method"]} not supported')

    err_inf = np.linalg.norm((Ax @ coef)[ind_union] - fx[ind_union], opts['objnorm'])
    print(f'norm error = {err_inf}')

    # Make sure the maximum is less than 1
    coef_full = np.zeros(deg + 1)
    if parity == 0:
        coef_full[::2] = coef
    else:
        coef_full[1::2] = coef

    max_sol = np.max(np.abs(np.polynomial.chebyshev.chebval(xpts, coef_full)))
    print(f'max of solution = {max_sol}')
    # if max_sol > 1.0 - 1e-10:
    #     raise ValueError('Solution is not bounded by 1. Increase npts')

    if opts['isplot']:
        plt.figure(1)
        plt.clf()
        plt.plot(xpts, Ax @ coef, 'ro', linewidth=1.5)
        for ind in ind_set:
            plt.plot(xpts[ind], fx[ind], 'b-', linewidth=2)
        plt.xlabel('$x$', fontsize=15)
        plt.ylabel('$f(x)$', fontsize=15)
        plt.legend(['polynomial', 'target'], fontsize=15)

        plt.figure(2)
        plt.clf()
        for ind in ind_set:
            plt.plot(xpts[ind], np.abs(Ax[ind] @ coef - fx[ind]), 'k-', linewidth=1.5)
        plt.xlabel('$x$', fontsize=15)
        plt.ylabel('$|f_\\mathrm{poly}(x)-f(x)|$', fontsize=15)
        plt.show()

    return coef_full

def get_unitary_sym(phi, x, parity):
    """Get the QSP unitary matrix based on given phase vector and point x.

    This function constructs the full QSP unitary matrix for a given set of
    phase factors at a specific point, handling both even and odd parity cases.

    Parameters
    ----------
    phi : array_like
        Phase factors for the QSP circuit:
        
        - For parity=1: reduced phase factors
        - For parity=0: phi[0] differs from reduced phase factors by factor of 2
    x : float
        Point at which to evaluate the unitary, must be in [-1, 1].
    parity : int
        Parity of the phase factors:
        
        - 0 : even parity
        - 1 : odd parity

    Returns
    -------
    ndarray
        The QSP unitary matrix at point x.
    """
    Wx = np.array([[x, 1j * np.sqrt(1 - x**2)], [1j * np.sqrt(1 - x**2), x]])
    expphi = np.exp(1j * phi)

    ret = np.array([[expphi[0], 0], [0, np.conj(expphi[0])]])

    for k in range(1, len(expphi)):
        temp = np.array([[expphi[k], 0], [0, np.conj(expphi[k])]])
        ret = np.dot(np.dot(ret, Wx), temp)

    return ret

def get_pim_sym(phi, x, parity):
    """Get the imaginary part of the QSP unitary matrix.

    Parameters
    ----------
    phi : array_like
        Phase factors for the QSP circuit
    x : float
        Point at which to evaluate the unitary
    parity : int
        Parity of the phase factors (0 for even, 1 for odd)

    Returns
    -------
    float
        Imaginary part of the (1,1) element of the QSP unitary matrix
    """
    U = get_unitary_sym(phi, x, parity)
    return np.imag(U[0, 0])

def get_pim_sym_real(phi, x, parity):
    """Get the imaginary part of the QSP unitary matrix using real arithmetic.

    Parameters
    ----------
    phi : array_like
        Phase factors for the QSP circuit
    x : float
        Point at which to evaluate the unitary
    parity : int
        Parity of the phase factors (0 for even, 1 for odd)

    Returns
    -------
    float
        Imaginary part of the (1,1) element of the QSP unitary matrix
    """
    n = len(phi)
    theta = np.arccos(x)
    
    # Define the 3D rotation matrix B
    B = np.array([
        [np.cos(2*theta), 0, -np.sin(2*theta)],
        [0, 1, 0],
        [np.sin(2*theta), 0, np.cos(2*theta)]
    ])
    
    # Initialize R based on parity
    if parity == 0:
        R = np.array([1, 0, 0])
    else:
        R = np.array([np.cos(theta), 0, np.sin(theta)])
    
    # Apply rotations
    for k in range(1, n):
        R_phi = np.array([
            [np.cos(2*phi[k]), -np.sin(2*phi[k]), 0],
            [np.sin(2*phi[k]), np.cos(2*phi[k]), 0],
            [0, 0, 1]
        ])
        R = B @ R_phi @ R
    
    # Final projection
    return np.array([np.sin(2*phi[-1]), np.cos(2*phi[-1]), 0]) @ R

def get_pim_deri_sym(phi, x, parity):
    """Get the derivative of the imaginary part of the QSP unitary matrix.

    Parameters
    ----------
    phi : array_like
        Phase factors for the QSP circuit
    x : float
        Point at which to evaluate the unitary
    parity : int
        Parity of the phase factors (0 for even, 1 for odd)

    Returns
    -------
    ndarray
        Derivatives of the imaginary part with respect to each phase factor
    """
    n = len(phi)
    theta = np.arccos(x)
    B = np.array([[np.cos(2 * theta), 0, -np.sin(2 * theta)],
                  [0, 1, 0],
                  [np.sin(2 * theta), 0, np.cos(2 * theta)]])
    
    L = np.zeros((n, 3))
    L[n-1, :] = [0, 1, 0]
    
    for k in range(n-2, -1, -1):
        L[k, :] = np.dot(L[k+1, :], np.dot(np.array([[np.cos(2 * phi[k+1]), -np.sin(2 * phi[k+1]), 0],
                                                     [np.sin(2 * phi[k+1]), np.cos(2 * phi[k+1]), 0],
                                                     [0, 0, 1]]), B))
    
    R = np.zeros((3, n))
    if parity == 0:
        R[:, 0] = [1, 0, 0]
    else:
        R[:, 0] = [np.cos(theta), 0, np.sin(theta)]
    
    for k in range(1, n):
        R[:, k] = np.dot(B, np.dot(np.array([[np.cos(2 * phi[k-1]), -np.sin(2 * phi[k-1]), 0],
                                             [np.sin(2 * phi[k-1]), np.cos(2 * phi[k-1]), 0],
                                             [0, 0, 1]]), R[:, k-1]))
    
    y = np.zeros(n+1)
    for k in range(n):
        y[k] = 2 * np.dot(L[k, :], np.dot(np.array([[-np.sin(2 * phi[k]), -np.cos(2 * phi[k]), 0],
                                                    [np.cos(2 * phi[k]), -np.sin(2 * phi[k]), 0],
                                                    [0, 0, 0]]), R[:, k]))
    y[n] = np.dot(L[n-1, :], np.dot(np.array([[np.cos(2 * phi[n-1]), -np.sin(2 * phi[n-1]), 0],
                                              [np.sin(2 * phi[n-1]), np.cos(2 * phi[n-1]), 0],
                                              [0, 0, 1]]), R[:, n-1]))
    
    return y

def get_pim_deri_sym_real(phi, x, parity):
    """Get the derivative of the imaginary part using real arithmetic.

    Parameters
    ----------
    phi : array_like
        Phase factors for the QSP circuit
    x : float
        Point at which to evaluate the unitary
    parity : int
        Parity of the phase factors (0 for even, 1 for odd)

    Returns
    -------
    ndarray
        Derivatives of the imaginary part with respect to each phase factor,
        plus one additional derivative term
    """
    n = len(phi)
    theta = np.arccos(x)
    
    # Define the 3D rotation matrix B
    B = np.array([
        [np.cos(2*theta), 0, -np.sin(2*theta)],
        [0, 1, 0],
        [np.sin(2*theta), 0, np.cos(2*theta)]
    ])
    
    # Initialize L matrix (n x 3)
    L = np.zeros((n, 3))
    L[n-1, :] = np.array([0, 1, 0])
    
    # Compute L matrix
    for k in range(n-2, -1, -1):
        R_phi = np.array([
            [np.cos(2*phi[k+1]), -np.sin(2*phi[k+1]), 0],
            [np.sin(2*phi[k+1]), np.cos(2*phi[k+1]), 0],
            [0, 0, 1]
        ])
        L[k, :] = L[k+1, :] @ R_phi @ B
    
    # Initialize R matrix (3 x n)
    R = np.zeros((3, n))
    if parity == 0:
        R[:, 0] = np.array([1, 0, 0])
    else:
        R[:, 0] = np.array([np.cos(theta), 0, np.sin(theta)])
    
    # Compute R matrix
    for k in range(1, n):
        R_phi = np.array([
            [np.cos(2*phi[k-1]), -np.sin(2*phi[k-1]), 0],
            [np.sin(2*phi[k-1]), np.cos(2*phi[k-1]), 0],
            [0, 0, 1]
        ])
        R[:, k] = B @ (R_phi @ R[:, k-1])
    
    # Compute derivatives
    y = np.zeros(n + 1)  # Note: n+1 size to match MATLAB
    
    for k in range(n):
        D_phi = np.array([
            [-np.sin(2*phi[k]), -np.cos(2*phi[k]), 0],
            [np.cos(2*phi[k]), -np.sin(2*phi[k]), 0],
            [0, 0, 0]
        ])
        y[k] = 2 * L[k, :] @ D_phi @ R[:, k]
    
    # Compute final derivative (n+1)th term
    R_phi = np.array([
        [np.cos(2*phi[n-1]), -np.sin(2*phi[n-1]), 0],
        [np.sin(2*phi[n-1]), np.cos(2*phi[n-1]), 0],
        [0, 0, 1]
    ])
    y[n] = L[n-1, :] @ R_phi @ R[:, n-1]
    
    return y

def F(phi, parity, opts):
    """Compute the Chebyshev coefficients of P_im.
    
    P_im is the imaginary part of the (1,1) element of the QSP unitary matrix.

    Parameters
    ----------
    phi : array_like
        Reduced phase factors
    parity : int
        Parity of phi (0 for even, 1 for odd)
    opts : dict
        Options dictionary with fields:
            - useReal : bool
                Whether to use real matrix multiplication

    Returns
    -------
    ndarray
        Chebyshev coefficients of P_im w.r.t. 
        T_(2k) for even parity or T_(2k-1) for odd parity
    """
    # Setup options for CM solver
    opts.setdefault('useReal', True)

    # Initial preparation
    d = len(phi)
    dd = 2 * d
    theta = np.arange(d + 1) * np.pi / dd
    M = np.zeros(2 * dd)

    if opts['useReal']:
        f = lambda x: [get_pim_sym_real(phi, xval, parity) for xval in x]
    else:
        f = lambda x: [get_pim_sym(phi, xval, parity) for xval in x]

    # Start Chebyshev coefficients evaluation
    M[:d+1] = f(np.cos(theta))
    M[d+1:dd+1] = (-1)**parity * M[d-1::-1]
    M[dd+1:] = M[dd-1:0:-1]
    M = np.fft.fft(M)  # FFT w.r.t. columns.
    M = np.real(M)
    M /= (2 * dd)
    M[1:-1] *= 2
    coe = M[parity:2*d:2]

    return coe

def F_Jacobian(phi, parity, opts):
    """Compute the Jacobian matrix of Chebyshev coefficients.

    Parameters
    ----------
    phi : array_like
        Reduced phase factors
    parity : int
        Parity of phi (0 for even, 1 for odd)
    opts : dict
        Options dictionary with fields:
            - useReal : bool
                Whether to use real matrix multiplication

    Returns
    -------
    tuple
        (f, df) where:
        - f is the function values (Chebyshev coefficients)
        - df is the Jacobian matrix (square matrix)
    """
    # Setup options
    opts.setdefault('useReal', True)

    # Initial preparation
    if opts['useReal']:
        f = lambda x: get_pim_deri_sym_real(phi, x, parity)
    else:
        f = lambda x: get_pim_deri_sym(phi, x, parity)

    d = len(phi)
    dd = 2 * d
    theta = np.arange(d + 1) * np.pi / dd
    M = np.zeros((2 * dd, d + 1))

    for n in range(d + 1):
        M[n, :] = f(np.cos(theta[n]))

    M[d + 1:dd + 1, :] = (-1) ** parity * M[d - 1::-1, :]
    M[dd + 1:, :] = M[dd - 1:0:-1, :]

    M = np.fft.fft(M, axis=0)  # FFT w.r.t. columns.
    M = np.real(M[:dd + 1, :])
    M[1:-1, :] *= 2
    M /= (2 * dd)

    f = M[parity::2, -1][:d]
    df = M[parity::2, :-1][:d]

    return f, df