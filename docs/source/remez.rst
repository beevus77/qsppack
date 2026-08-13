Constrained Remez Approximation
===============================

The constrained Remez interface constructs fixed-parity Chebyshev
approximations on fitting intervals in :math:`[0,1]`, while enforcing QSP
polynomial bounds on the full domain.

.. currentmodule:: qsppack.remez

.. autofunction:: remez
.. autoclass:: RemezResult
   :members:
.. autoclass:: RemezMetrics
   :members:
.. autoclass:: RemezOptions
   :members:
.. autoclass:: ConstrainedRemezFitter
   :members:
.. autofunction:: plot_remez_result
.. autoexception:: RemezConvergenceError
.. autoclass:: RemezConvergenceWarning
