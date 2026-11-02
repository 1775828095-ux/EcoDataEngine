"""Independent SWAT+ aquifer.aqu token readback against official and synthetic sources."""
import csv
from decimal import Decimal
import io
import json
from pathlib import Path
import zipfile

import pytest
from ecodata.core.exceptions import SchemaError
from ecodata.round1 import convert

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / 'examples/ai_round1/swat_plus/aquifer'
FIELDS = 'id name init gw_flo dep_bot dep_wt no3_n sol_p carbon flo_dist bf_max alpha_bf revap rchg_dp spec_yld hl_no3n flo_min revap_min'.split()


def read_output(content):
    lines = content.decode('ascii').splitlines()
    assert lines[0].startswith('aquifer.aqu') and lines[1].split() == FIELDS
    rows = [line.split() for line in lines[2:]]
    assert all(len(row) == 18 for row in rows)
    return rows


def test_official_reference_values_and_init_links():
    cfg = json.loads((SAMPLES / 'config_official.json').read_text())
    files, report = convert('swat_plus', (SAMPLES / 'A_official_decoded_bundle.zip').read_bytes(), cfg)
    output = read_output(files['aquifer.aqu'])
    with zipfile.ZipFile(SAMPLES / 'A_official_decoded_bundle.zip') as z:
        source = list(csv.DictReader(io.TextIOWrapper(z.open('source_aquifers.csv'))))
        valid_names = {line.split()[0] for line in z.read('initial.aqu').decode().splitlines()[2:]}
    assert len(output) == len(source) == 21
    b2_lines = (SAMPLES / 'B2_official_aquifer.aqu').read_text(encoding='ascii').splitlines()
    b2_rows = [line.split() for line in b2_lines[2:]]
    assert len(b2_rows) == len(output)
    for line, row in zip(output, source):
        assert line[:3] == [row[field] for field in FIELDS[:3]]
        assert line[2] in valid_names
        assert all(Decimal(value) == Decimal(row[field]) for field, value in zip(FIELDS[3:], line[3:]))
    for line, b2 in zip(output, b2_rows):
        assert line[:3] == b2[:3]
        assert all(Decimal(a) == Decimal(b) for a, b in zip(line[3:], b2[3:]))
    assert report['aquifer_records'] == 21 and set(files) == {'aquifer.aqu', 'README_aquifer.txt', 'conversion_report.json'}
    assert 'aquifer.con' not in files and 'file.cio' not in files


def test_synthetic_mapped_units_and_order():
    cfg = json.loads((SAMPLES / 'config_synthetic.json').read_text())
    files, report = convert('swat_plus', (SAMPLES / 'A_synthetic_mapped_bundle.zip').read_bytes(), cfg)
    rows = read_output(files['aquifer.aqu'])
    assert len(rows) == 2
    assert [row[:3] for row in rows] == [['1', 'aqA', 'initaqu1'], ['2', 'aqB', 'initaqu1']]
    assert [Decimal(rows[0][FIELDS.index(field)]) for field in ('gw_flo', 'dep_bot', 'carbon')] == [Decimal('5'), Decimal('10'), Decimal('0.5')]
    assert [Decimal(rows[1][FIELDS.index(field)]) for field in ('gw_flo', 'dep_bot', 'carbon')] == [Decimal('2.5'), Decimal('20'), Decimal('0.8')]
    assert report['aquifer_records'] == 2


def test_missing_field_and_invalid_init_rejected():
    cfg = json.loads((SAMPLES / 'config_synthetic.json').read_text())
    content = (SAMPLES / 'A_synthetic_mapped_bundle.zip').read_bytes()
    bad = dict(cfg)
    bad['fields'] = {key: value for key, value in cfg['fields'].items() if key != 'dep_wt'}
    with pytest.raises(SchemaError, match='18个'):
        convert('swat_plus', content, bad)
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        csv_data = z.read('mapped_source.csv').replace(b'initaqu1', b'missing1')
        initial = z.read('initial.aqu')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as z:
        z.writestr('mapped_source.csv', csv_data)
        z.writestr('initial.aqu', initial)
    with pytest.raises(SchemaError, match='不在所给initial.aqu'):
        convert('swat_plus', buffer.getvalue(), cfg)
