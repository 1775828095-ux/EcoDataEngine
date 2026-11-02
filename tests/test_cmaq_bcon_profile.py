"""Independent BCON/profile ASCII readback and conversion checks."""
import csv
from decimal import Decimal
import io
import json
from pathlib import Path

import pytest
from ecodata.core.exceptions import SchemaError
from ecodata.round1 import convert

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/ai_round1/cmaq/bcon_profile'
OFFICIAL = Path(__file__).resolve().parents[1] / 'examples/ai_round1/cmaq/icon_profile/reference_official.csv'


def independent_read(content):
    lines = content.decode('ascii').splitlines()
    assert lines[0].startswith('#')
    reader = csv.reader(io.StringIO('\n'.join(line for line in lines if not line.startswith('#'))))
    header = next(reader)
    assert header[:2] == ['VNAME', 'UNITS']
    rows = {}
    for row in reader:
        assert len(row) == len(header)
        assert row[0] not in rows and len(row[0]) <= 16 and len(row[1]) <= 16
        rows[row[0]] = (row[1], tuple(Decimal(v) for v in row[2:]))
    return header, rows


def test_official_long_matches_independent_official_profile():
    cfg = json.loads((EXAMPLE / 'config_official_long.json').read_text())
    files, report = convert('cmaq', (EXAMPLE / 'A_official_decoded_long.csv').read_bytes(), cfg)
    hdr, rows = independent_read(files['BC_PROFILE_CMAQ55.csv'])
    reference_hdr, reference_rows = independent_read(OFFICIAL.read_bytes())
    assert hdr == reference_hdr and rows == reference_rows
    assert len(rows) == 251 and len(hdr) - 2 == 44 and report['values'] == 11044
    assert report['BCON_run_verified'] is False


def test_synthetic_wide_and_unit_conversions():
    cfg = json.loads((EXAMPLE / 'config_synthetic_wide.json').read_text())
    files, report = convert('cmaq', (EXAMPLE / 'A_synthetic_wide.csv').read_bytes(), cfg)
    hdr, rows = independent_read(files['BC_PROFILE_CMAQ55.csv'])
    assert hdr == ['VNAME', 'UNITS', 'LAYER01', 'LAYER02', 'LAYER03']
    assert rows == {'ZH': ('m', (Decimal('50'), Decimal('150'), Decimal('300'))),
                    'PRES': ('Pa', (Decimal('100000'), Decimal('90000'), Decimal('80000'))),
                    'O3': ('ppmV', (Decimal('0.020'), Decimal('0.030'), Decimal('0.040')))}
    assert report['values'] == 9 and report['coordinates'].startswith('ZH metres')


@pytest.mark.parametrize('bad,reason', [
    (b'O3,ppbV,20,30,NA', '缺测'),
    (b'O3,ppbV,20,30,-1', '负值'),
    (b'O3,ppbV,20,30,nan', '有限值'),
    (b'O3,ppbV,20,30,40\nO3,ppbV,20,30,40', '重复'),
])
def test_invalid_data_rejected(bad, reason):
    cfg = json.loads((EXAMPLE / 'config_synthetic_wide.json').read_text())
    source = (EXAMPLE / 'A_synthetic_wide.csv').read_bytes()
    source = source.replace(b'O3,ppbV,20,30,40', bad)
    with pytest.raises(SchemaError, match=reason):
        convert('cmaq', source, cfg)
