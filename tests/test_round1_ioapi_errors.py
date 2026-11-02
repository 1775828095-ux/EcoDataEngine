from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from ecodata.core.exceptions import SchemaError
from ecodata.models.adapters.ioapi_reader import CmaqIOAPIReader

def test_native_non_utf8_diagnostic_is_reported_without_typeerror():
    failed=SimpleNamespace(returncode=1,stdout=b'',stderr='无法读取中文路径'.encode('gb18030'))
    with patch('subprocess.run',return_value=failed):
        with pytest.raises(SchemaError,match='读取失败'):
            CmaqIOAPIReader(__import__('sys').executable).inspect('not_ioapi.nc')

def test_reader_rejects_invalid_json_even_with_zero_exit_code():
    failed=SimpleNamespace(returncode=0,stdout=b'not json',stderr=b'')
    with patch('subprocess.run',return_value=failed):
        with pytest.raises(SchemaError,match='JSON'):
            CmaqIOAPIReader(__import__('sys').executable).inspect('not_ioapi.nc')

def test_actual_auxiliary_file_inspects_on_chinese_path_but_is_not_ioapi():
    case=Path(__file__).resolve().parents[1]/'examples/ai_round1/cmaq/B/model_input.nc'
    reader=CmaqIOAPIReader()
    if not reader.python.is_file():reader=CmaqIOAPIReader(__import__('sys').executable)
    try:report=reader.inspect(case,format='ioapi')
    except SchemaError as exc:
        if 'PseudoNetCDF' in str(exc) and 'No module named' in str(exc):pytest.skip('optional PseudoNetCDF is absent')
        raise
    assert not report['tflag_present'] and 'FTYPE' in report['missing_ioapi_attributes']
    assert not report['native_model_input_verified']
