# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import tomli
sys.path.insert(0, os.path.abspath('../..'))

# Read version from pyproject.toml
with open("../../pyproject.toml", "rb") as f:
    pyproject = tomli.load(f)
    version = pyproject["project"]["version"]
    release = version

project = 'qsppack'
copyright = '2025-2026, James Larsen'
author = 'James Larsen'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'nbsphinx',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_title = 'qsppack Documentation'
html_short_title = 'qsppack'

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# nbsphinx settings
nbsphinx_execute = 'auto'
nbsphinx_allow_errors = False
nbsphinx_timeout = 600

# Shared namespace for executable examples in both API docstrings and RST pages.
doctest_global_setup = """
import numpy as np
from qsppack import *
"""

# Autodoc settings
autodoc_default_options = {
    'member-order': 'bysource',
    'show-inheritance': True,
}

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = False

# Intersphinx settings
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}
