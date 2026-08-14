Nonlinear Fourier Analysis Module
=================================

The nonlinear Fourier analysis (NLFA) module provides functions for working with nonlinear Fourier transforms and related operations.

.. currentmodule:: qsppack.nlfa

.. autofunction:: b_from_cheb
.. autofunction:: weiss
.. autofunction:: inverse_nonlinear_FFT
.. autofunction:: forward_nlft
.. autofunction:: forward_nonlinear_FFT

These functions provide essential operations for nonlinear Fourier analysis,
including:

- Converting Chebyshev coefficients to complex polynomial coefficients
- Computing the Weiss algorithm for polynomial coefficients
- Performing inverse nonlinear FFT operations
- Computing forward nonlinear Fourier transforms
- Computing forward nonlinear FFT with recursive algorithm

Example
-------

This example starts with a small feasible odd polynomial, constructs its
complement with the Weiss factorization, recovers the nonlinear Fourier
parameters, and verifies the forward transform.

.. testcode::

    import numpy as np
    from qsppack.nlfa import (
        b_from_cheb,
        forward_nlft,
        forward_nonlinear_FFT,
        inverse_nonlinear_FFT,
        weiss,
    )

    partial_chebyshev_coefficients = np.array([0.2, 0.1])
    b_coefficients = b_from_cheb(
        partial_chebyshev_coefficients,
        parity=1,
    )
    a_coefficients = weiss(b_coefficients, N=32)

    gammas, _, _ = inverse_nonlinear_FFT(
        a_coefficients,
        b_coefficients,
    )
    _, reconstructed_b = forward_nonlinear_FFT(gammas)

    np.testing.assert_allclose(reconstructed_b, b_coefficients, atol=1e-14)
    np.testing.assert_allclose(forward_nlft(gammas), b_coefficients, atol=1e-14)
