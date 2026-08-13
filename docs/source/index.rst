.. qsppack documentation master file, created by
   sphinx-quickstart on Thu May  8 15:36:23 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to qsppack's documentation!
===================================

qsppack is a Python package for Quantum Signal Processing optimization and analysis.
This package is based on a MATLAB package by the same name. The MATLAB code can be found at https://github.com/qsppack/QSPPACK, with an earlier version of the tutorials and examples at https://qsppack.gitbook.io/qsppack/ (not actively maintained).


Installation
------------

You can install qsppack using pip::

    pip install qsppack

Or install from source::

    git clone https://github.com/qsppack/pyqsppack.git
    cd pyqsppack
    pip install -e .

For development and testing, install with test dependencies::

    pip install -e ".[test]"


Contents
========
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   solver
   objective
   optimizers
   utils
   nlfa
   remez
   retraction

Example Applications
====================

.. toctree::
   :maxdepth: 2
   :caption: Example Applications:

   examples

Polynomial Approximation Examples
=================================

.. toctree::
   :maxdepth: 2
   :caption: Polynomial Approximation Examples:

   polynomial_approximation_examples

* :ref:`genindex`
