import importlib

import pytest


@pytest.mark.parametrize(
    "module", ["numpy", "pandas", "pydicom", "torch", "torchvision"]
)
def test_core_imports(module):
    assert importlib.import_module(module) is not None
