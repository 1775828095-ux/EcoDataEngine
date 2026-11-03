import builtins
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_platform_import_and_registry_without_golem():
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'golem' or name.startswith('golem.'):
        raise AssertionError('GOLEM must not be imported')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from ecodata.algorithms.platform_search import platform_registry
assert platform_registry().specs()
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_retired_modes_fail_explicitly(tmp_path):
    from ecodata.algorithms.platform_search import submit_search
    for mode in ('A', 'B', 'C', 'C-gap', 'C-local', 'C-directed', 'R'):
        folder = tmp_path / mode
        with pytest.raises(ValueError, match='GOLEM'):
            submit_search([], {}, None, folder, mode=mode)
        assert json.loads((folder / 'failure.json').read_text(encoding='utf-8'))['status'] == 'BLOCKED'


def test_backend_cannot_start():
    from ecodata.algorithms.numerical_search import run_numerical_search
    from ecodata.algorithms.golem_generation import TypedGolemFactory
    for entry in (run_numerical_search, TypedGolemFactory):
        with pytest.raises(ValueError, match='GOLEM'):
            entry()
