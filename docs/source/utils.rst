Utilities Module
================

The utils module provides various utility functions for QSP operations.

.. currentmodule:: qsppack.utils

.. autofunction:: get_unitary
.. autofunction:: reduced_to_full
.. autofunction:: chebyshev_to_func
.. autofunction:: cvx_poly_coef
.. autofunction:: F
.. autofunction:: F_Jacobian

These utility functions provide operations for QSP evaluation, phase-factor
conversion, and polynomial coefficient manipulation.

Example usage:

.. code-block:: python

    import numpy as np
    from qsppack.utils import (
        F,
        F_Jacobian,
        chebyshev_to_func,
        get_unitary,
        reduced_to_full,
    )

    # Evaluate the real part of the (0, 0) QSP unitary entry.
    phase_factors = np.array([0.1, 0.2, 0.3])
    value = get_unitary(phase_factors, x=0.5)

    # Convert reduced symmetric phase factors to a full even sequence.
    reduced_phases = np.array([0.1, 0.2])
    full_phases = reduced_to_full(
        reduced_phases,
        parity=0,
        targetPre=True,
    )

    # Evaluate 0.5*T_1(x) from its odd partial coefficient vector.
    x = np.linspace(-1, 1, 100)
    func_values = chebyshev_to_func(
        x,
        coef=np.array([0.5]),
        parity=1,
        partialcoef=True,
    )

    # Evaluate the reduced phase-to-coefficient map and its Jacobian.
    options = {'useReal': True}
    coefficients = F(reduced_phases, parity=0, opts=options)
    coefficients, jacobian = F_Jacobian(
        reduced_phases,
        parity=0,
        opts=options,
    )
