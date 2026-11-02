"""Actual converter wiring and ambiguous metadata rejection."""
from pathlib import Path
import json,io,zipfile
import pytest
from ecodata.interface.engineering_conversion import convert_input


def test_real_cmaq_source_produces_unique_report_and_original_hash():
    p=Path('D:/EcoDataEngine/cases/CMAQ_20261024/official_12NE3/platform_no')
    if not p.is_dir():pytest.skip('Local official input not installed')
    raw=(p/'A_official_finished_NO_decoded.nc').read_bytes()
    payload,report=convert_input('cmaq',raw,json.loads((p/'config.json').read_text()))
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        assert len(z.namelist())==len(set(z.namelist()))
        assert z.namelist().count('conversion_report.json')==1
        assert json.loads(z.read('conversion_report.json'))==report
        assert 'emissions.CMAQ55.nc' in z.namelist()
    assert report['source_sha256']==__import__('hashlib').sha256(raw).hexdigest()


@pytest.mark.parametrize('config',[[],{'mode':'bcon_profile'},{'mode':'unsupported'}])
def test_unsupported_engineering_update_never_falls_back(config):
    with pytest.raises(ValueError):convert_input('cmaq',b'not executed',config)
