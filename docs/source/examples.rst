Example applications
====================

These notebooks apply the full workflow: construct a target polynomial,
synthesize QSP phases, and compare the resulting unitary entry with the target.

Core algorithms
---------------

Polynomial inversion and Hamiltonian simulation illustrate the most common
QSP transformations and the associated approximation errors.

.. toctree::
   :maxdepth: 1

   examples/linear_systems
   examples/negative_power_function
   examples/hamiltonian_simulation

Filters and singular-value transformations
------------------------------------------

These examples construct filters, projectors, and amplitude transformations.

.. toctree::
   :maxdepth: 1

   examples/gaussian_filter
   examples/singular_value_threshold_projector
   examples/singular_vector_transformation
   examples/uniform_singular_value_amplification

State preparation
-----------------

These notebooks use QSP polynomials to prepare Gaussian, Kaiser-window, and
Gibbs distributions.

.. toctree::
   :maxdepth: 1

   examples/gaussian_state
   examples/kaiser_window_state
   examples/gibbs_state
