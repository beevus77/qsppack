Constrained Remez Approximation
===============================

The constrained Remez interface constructs fixed-parity Chebyshev
approximations on fitting intervals in :math:`[0,1]`, while enforcing QSP
polynomial bounds on the full domain.

See :doc:`remez_tutorial` for a task-oriented walkthrough that constructs and
diagnoses an approximation.

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
