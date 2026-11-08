"""Marks this directory as a package so ``import evals`` finds the real package.

This file is load-bearing, not boilerplate -- the other test directories in the
workspace deliberately have none.

Under ``--import-mode=importlib`` (set in the root pytest config) pytest derives
a test module's dotted name from its path below the rootdir and *fabricates*
namespace modules for the missing parents. For ``evals/tests/test_trec.py`` that
means it registers a module named ``evals`` pointing at ``<repo>/evals/`` --
the project directory, not the package in ``evals/src/evals/``. That shadow has
no ``__file__``, so ``from evals import trec`` dies with the distinctive
"cannot import name 'trec' from 'evals' (unknown location)".

No other workspace member hits this, because only here does the project
directory share a name with the package it ships (``services/gateway/`` ships
``gateway``). With this ``__init__.py``, pytest stops walking up at ``evals/``
(which has no ``__init__.py``), names the module ``tests.test_trec``, and never
invents an ``evals`` module -- so the editable install resolves normally.

The alternative, a ``conftest.py`` that patches ``sys.path``, collides with the
root ``tests/conftest.py`` under mypy ("Duplicate module named conftest") and
would need an edit to the orchestrator-owned root ``pyproject.toml``.
"""
