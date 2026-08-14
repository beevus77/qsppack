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

.. testcode::

    import numpy as np
    from qsppack import get_entry, solve

    coefficients = np.array([0.2, 0.1])
    parity = 1
    x = np.linspace(-1.0, 1.0, 51)
    expected = np.polynomial.chebyshev.chebval(x, [0.0, 0.2, 0.0, 0.1])

    common_options = {
        'criteria': 1e-10,
        'targetPre': True,
        'typePhi': 'full',
        'print': False,
        'N': 256,
    }
    for method in ('FPI', 'Newton', 'LBFGS', 'NLFT'):
        phases, info = solve(
            coefficients,
            parity,
            {**common_options, 'method': method},
        )
        assert info['converged'], method
        np.testing.assert_allclose(
            get_entry(x, phases, info),
            expected,
            atol=1e-8,
        )
