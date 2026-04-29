"""Smoke tests for placeholder shared utility modules."""

import importlib


def test_oracle_module_imports() -> None:
    module = importlib.import_module("ola.utils.oracle")

    assert module.__doc__ is not None
    assert "Oracle" in module.__doc__


def test_seeding_module_imports() -> None:
    module = importlib.import_module("ola.utils.seeding")

    assert module.__doc__ is not None
    assert "Random seed" in module.__doc__