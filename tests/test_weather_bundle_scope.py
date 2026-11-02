import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from ecodata.core.exceptions import SchemaError
from ecodata.interface.round1_page import INPUT_SCOPE
from ecodata.models.adapters.swat_plus_upload import _validate_weather_references
from ecodata.round1 import convert

ROOT = Path(__file__).resolve().parents[1]


def test_bundle_references_and_manifest():
    case = ROOT / 'examples/ai_round1/swat_plus/power_2021'
    files, report = convert('swat_plus', (case/'A_power_2021.nc').read_bytes(),
                            json.loads((case/'config_ui.json').read_text()))
    assert 'file.cio' not in files and 'weather-wgn.cli' not in files
    assert report['index_references_verified'] is True
    _validate_weather_references(files)
    manifest = json.loads(files['manifest.json'])
    assert 'file.cio' not in manifest['files']
    assert 'file.cio' not in manifest['validation']['generated_files']
    for name, meta in manifest['files'].items():
        assert meta['size'] == len(files[name])
        assert meta['sha256'] == hashlib.sha256(files[name]).hexdigest()
    files['tmp.cli'] = b'tmp.cli\nFILENAME\nabsent.tem\n'
    with pytest.raises(SchemaError, match='不存在的文件：absent.tem'):
        _validate_weather_references(files)
    files['tmp.cli'] = b'tmp.cli\nFILENAME\nstation.tem\n'
    files['weather-sta.cli'] = files['weather-sta.cli'].replace(b'null', b'absent.wgn', 1)
    with pytest.raises(SchemaError, match='不能引用未提供的WGN'):
        _validate_weather_references(files)


@pytest.mark.parametrize('model', list(INPUT_SCOPE))
def test_scope_visible_before_upload(model):
    with patch('streamlit.file_uploader', return_value=None):
        app = AppTest.from_string(
            'from ecodata.interface.round1_page import render_round1_page\nrender_round1_page()',
            default_timeout=30).run()
        app.selectbox[0].set_value(model).run()
        assert not app.exception
        assert any(INPUT_SCOPE[model] in notice.value for notice in app.info)
        assert not app.get('download_button')
        assert app.button[0].disabled
