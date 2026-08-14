Objective Module
================

The objective module provides functions for computing objective functions and their gradients in QSP optimization.

.. currentmodule:: qsppack.objective

.. autofunction:: obj_sym
.. autofunction:: grad_sym
.. autofunction:: grad_sym_real

These functions are used internally by the optimization methods to evaluate the quality of phase factor solutions and compute gradients for optimization.

Example
-------

The low-level objective functions receive sample points and an options
dictionary containing the target callable and polynomial parity.

.. testcode::

    import numpy as np
    from qsppack.objective import obj_sym, grad_sym

    phase_factors = np.array([0.1, 0.2, 0.3])
    samples = np.linspace(-0.8, 0.8, 5)
    options = {
        "parity": 1,
        "target": lambda x: 0.2 * x,
    }

    objective = obj_sym(phase_factors, samples, options)
    gradient, gradient_objective = grad_sym(
        phase_factors,
        samples,
        options,
    )

    assert objective.shape == samples.shape
    assert gradient.shape == (len(samples), len(phase_factors))
    np.testing.assert_allclose(gradient_objective, objective)
