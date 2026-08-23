Utilities Module
================

The utils module provides various utility functions for QSP operations.

.. currentmodule:: qsppack.utils

Public evaluation and conversion helpers
----------------------------------------

.. autoclass:: FeasibilityCertificate
   :members:
.. autofunction:: check_feasibility
.. autofunction:: get_unitary
.. autofunction:: get_unitary_sym
.. autofunction:: get_entry
.. autofunction:: reduced_to_full
.. autofunction:: chebyshev_to_func
.. autofunction:: cvx_poly_coef

Low-level phase-map helpers
---------------------------

.. autofunction:: F
.. autofunction:: F_Jacobian

These utility functions provide operations for QSP evaluation, phase-factor
conversion, and polynomial coefficient manipulation.

Example
-------

.. testcode::

    import numpy as np
    from qsppack.utils import (
        F,
        F_Jacobian,
        chebyshev_to_func,
        check_feasibility,
        get_unitary,
        reduced_to_full,
    )

    # Evaluate the real part of the (0, 0) QSP unitary entry.
    phase_factors = np.array([0.1, 0.2, 0.3])
    value = get_unitary(phase_factors, x=0.5)
    assert np.isfinite(value)

    # Convert reduced symmetric phase factors to a full even sequence.
    reduced_phases = np.array([0.1, 0.2])
    full_phases = reduced_to_full(
        reduced_phases,
        parity=0,
        targetPre=True,
    )
    assert len(full_phases) == 3

    # Evaluate 0.5*T_1(x) from its odd partial coefficient vector.
    x = np.linspace(-1, 1, 100)
    func_values = chebyshev_to_func(
        x,
        coef=np.array([0.5]),
        parity=1,
        partialcoef=True,
    )
    np.testing.assert_allclose(func_values, 0.5 * x, atol=1e-14)

    # Certify |T_2(x)| <= 1 from its endpoints and derivative roots.
    certificate = check_feasibility(np.array([0.0, 0.0, 1.0]))
    assert certificate.is_feasible
    assert certificate.max_magnitude == 1.0

    # Evaluate the reduced phase-to-coefficient map and its Jacobian.
    options = {'useReal': True}
    coefficients = F(reduced_phases, parity=0, opts=options)
    coefficients, jacobian = F_Jacobian(
        reduced_phases,
        parity=0,
        opts=options,
    )
    assert jacobian.shape == (len(reduced_phases), len(reduced_phases))
