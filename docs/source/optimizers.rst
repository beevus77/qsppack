Optimizers Module
=================

The optimizers module provides iterative and direct methods for QSP phase
factor synthesis. The recommended interface is :func:`qsppack.solve`, which
prepares the method-specific inputs and returns phase factors in a consistent
format.

The available solver methods are:

* ``FPI`` -- fixed-point iteration. Its historical low-level function name is
  :func:`coordinate_minimization`.
* ``Newton`` -- Newton iteration on the symmetric phase-to-coefficient map.
* ``LBFGS`` -- limited-memory BFGS minimization on Chebyshev sample points.
* ``NLFT`` -- direct synthesis using the Weiss factorization and inverse
  nonlinear Fourier transform.

.. currentmodule:: qsppack.optimizers

.. autofunction:: lbfgs
.. autofunction:: coordinate_minimization
.. autofunction:: newton
.. autofunction:: nlft

The low-level functions are exposed for specialized use, but their signatures
are method-specific. For normal use, select a method through
:func:`qsppack.solve` as follows.

Example
-------

.. code-block:: python

    import numpy as np
    from qsppack import solve

    # Approximate 0.5*cos(10*x) by an even degree-60 polynomial. solve()
    # accepts only the coefficients of the polynomial's nonzero parity.
    full_coefficients = np.polynomial.chebyshev.chebinterpolate(
        lambda x: 0.5 * np.cos(10 * x),
        60,
    )
    coefficients = full_coefficients[::2]
    parity = 0

    common_options = {
        'criteria': 1e-10,
        'targetPre': True,
        'typePhi': 'full',
        'print': False,
    }
    for method in ('FPI', 'Newton', 'LBFGS', 'NLFT'):
        phases, info = solve(
            coefficients,
            parity,
            {**common_options, 'method': method},
        )
        print(method, info['value'])
