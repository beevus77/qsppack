Solver Module
=============

The solver module provides the main functionality for solving Quantum Signal Processing (QSP) problems.

.. currentmodule:: qsppack.solver

.. autofunction:: solve

The :func:`solve` function is the main entry point for QSP phase synthesis. It
takes the nonzero-parity Chebyshev coefficients, the polynomial parity, and an
options dictionary. It returns the phase factors together with convergence and
timing information.

Example usage:

.. code-block:: python

    import numpy as np
    from qsppack.solver import solve

    # P(x) = 0.5*x is odd, so its partial coefficient vector is [0.5].
    coefficients = np.array([0.5])
    parity = 1
    options = {
        'method': 'Newton',
        'criteria': 1e-12,
        'targetPre': True,
        'typePhi': 'full',
        'print': False,
    }

    phases, info = solve(coefficients, parity, options)
