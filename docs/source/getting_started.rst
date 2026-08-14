Getting started
===============

Installation
------------

Install the released package from PyPI:

.. code-block:: bash

   pip install qsppack

For development, clone the repository and install the package with its test
dependencies:

.. code-block:: bash

   git clone https://github.com/qsppack/pyqsppack.git
   cd pyqsppack
   pip install -e ".[test]"

The QSP workflow
----------------

A typical calculation has three stages:

1. **Construct a bounded definite-parity polynomial.** Use
   :func:`qsppack.utils.cvx_poly_coef`, :func:`qsppack.remez.remez`, or
   :func:`qsppack.retraction.retract`, depending on how the target polynomial
   is obtained.
2. **Synthesize phase factors.** Pass the polynomial's nonzero-parity
   Chebyshev coefficients to :func:`qsppack.solve`.
3. **Verify the result.** Evaluate the synthesized sequence with
   :func:`qsppack.utils.get_entry` and inspect ``info["converged"]`` and
   ``info["value"]``.

Minimal example
---------------

The odd polynomial
:math:`P(x)=0.2T_1(x)+0.1T_3(x)` is represented by the partial coefficient
vector ``[0.2, 0.1]``.

.. testcode::

   import numpy as np
   from qsppack import get_entry, solve

   coefficients = np.array([0.2, 0.1])
   phases, info = solve(
       coefficients,
       parity=1,
       opts={
           "method": "Newton",
           "criteria": 1e-10,
           "targetPre": True,
           "typePhi": "full",
           "print": False,
       },
   )

   x = np.linspace(-1.0, 1.0, 101)
   expected = np.polynomial.chebyshev.chebval(x, [0.0, 0.2, 0.0, 0.1])
   actual = get_entry(x, phases, info)

   assert info["converged"]
   np.testing.assert_allclose(actual, expected, atol=1e-8)

Coefficient and phase conventions
---------------------------------

``solve`` accepts only coefficients of the polynomial's active parity, ordered
from lowest to highest degree. For parity zero, ``[c0, c2, ...]`` represents
:math:`c_0T_0+c_2T_2+\cdots`; for parity one, ``[c1, c3, ...]`` represents
:math:`c_1T_1+c_3T_3+\cdots`. Functions that return full coefficient arrays
place zeros in the opposite-parity positions.

The legacy option names encode two independent choices:

``targetPre``
   ``True`` uses the real-part target convention. ``False`` uses the
   imaginary-part convention.

``typePhi``
   ``"full"`` returns the complete symmetric phase sequence.
   ``"reduced"`` returns only its independent half.

Pass the returned ``info`` dictionary directly to ``get_entry`` so evaluation
uses the same conventions as synthesis.

Choosing a synthesis method
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 16 20 64

   * - Method
     - Type
     - When to use it
   * - ``FPI``
     - Iterative
     - Simple fixed-point iteration for targets in its contraction regime.
   * - ``Newton``
     - Iterative
     - Fast local convergence using the phase-to-coefficient Jacobian.
   * - ``LBFGS``
     - Iterative
     - Optimization-based alternative that avoids a dense Newton solve.
   * - ``NLFT``
     - Direct
     - Weiss factorization followed by the inverse nonlinear Fourier transform;
       configure its even FFT length with ``N``.

Every method returns the same phase convention through ``solve``. A result may
still be returned after an iteration limit, so always check
``info["converged"]`` before using it downstream.
