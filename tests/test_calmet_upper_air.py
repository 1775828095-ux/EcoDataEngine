import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecodata.core.exceptions import SchemaError
from ecodata.models.adapters.calmet_upper_air import export_upper_air


def read_up_independently(data):
    """Fixed-column header and list-directed payload per native RDHDU/RDUP.

    Does not import or call converter helpers, and checks record boundaries.
    """
    lines = data.decode('ascii').splitlines()
    assert lines[0][:16].strip() == 'UP.DAT'
    assert lines[0][16:32].strip() == '2.0'
    i = 2 + int(lines[1])
    projection = lines[i].strip(); i += 1
    if projection == 'LL':
        datum = lines[i][:8].strip(); assert datum in {'WGS-84', 'NAD-83', 'NAD-27'}
        assert lines[i + 1].strip() == 'DEG'; i += 2
    else:
        assert projection == 'NONE'
    header = lines[i]
    dates = [int(header[1 + 5*j:6 + 5*j]) for j in range(6)]
    top = float(header[31:36]); assert int(header[36:41]) == 3
    assert int(header[41:46]) == 2
    i += 1
    assert [lines[i][j] for j in (5, 10, 15, 20)] == ['F']*4
    i += 1
    station_location = None
    if projection == 'LL':
        station_location = lines[i].split(); assert len(station_location) == 5
        i += 1
    rows = []
    while i < len(lines):
        header = lines[i]
        assert len(header) == 70
        station = int(header[9:17]); yy = int(header[22:24])
        century = dates[0] - dates[0] % 100
        year = century + yy
        if year < dates[0] - 50: year += 100
        if year >= dates[0] + 50: year -= 100
        dt = datetime(year, int(header[24:26]), int(header[26:28]), int(header[28:30]), tzinfo=timezone.utc)
        count = int(header[65:70]); i += 1
        for level in range(1, count + 1):
            values = [float(v) for v in lines[i].split(',')]
            assert len(values) == 5
            rows.append([station, dt, level] + values); i += 1
    return pd.DataFrame(rows, columns=['station', 'time', 'level', 'pressure', 'height', 'temperature', 'wind_direction', 'wind_speed']), dates, top, station_location


def test_boolean_measurement_cannot_be_silently_numeric(case):
    frame,config=case
    frame[config['columns']['wind_speed']]=True
    with pytest.raises(SchemaError,match='布尔'):export_upper_air(frame,config)


@pytest.fixture
def case():
    base = Path(__file__).resolve().parents[1]/'examples/ai_round1/calpuff/upper_air'
    return pd.read_csv(base/'A_NOAA_IGRA_72357_20260101.csv'), json.loads((base/'config_NOAA_IGRA.json').read_text(encoding='utf-8'))


def test_noaa_raw_independent_numeric(case):
    frame, config = case
    files, report = export_upper_air(frame, config)
    parsed, dates, top, loc = read_up_independently(files['UP1.DAT'])
    assert dates == [2026, 1, 0, 2026, 1, 12]
    assert top == 500 and loc is None
    assert len(parsed) == 76 and report['stations'][0]['levels'] == [38, 38]
    np.testing.assert_array_equal(parsed.station, frame.station_id)
    np.testing.assert_array_equal(parsed.level, frame.level)
    np.testing.assert_allclose(parsed.pressure, frame.pressure_pa/100, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(parsed.height, frame.height_m_msl)
    np.testing.assert_allclose(parsed.temperature, frame.temperature_c+273.15, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(parsed.wind_direction, frame.wind_from_degree)
    np.testing.assert_array_equal(parsed.wind_speed, frame.wind_speed_ms)
    assert list(parsed.time) == list(pd.to_datetime(frame.time, utc=True))
    assert report['interpolation'] == 'none' and not report['native_model_input_verified']


def test_preserve_variable_specific_missing_without_unit_conversion_of_codes(case):
    frame, cfg = case
    frame.loc[1, 'temperature_c'] = -9999
    frame.loc[2, 'wind_from_degree'] = np.nan
    frame.loc[3, 'wind_speed_ms'] = np.nan
    cfg['missing_policy'] = 'preserve'
    files, report = export_upper_air(frame, cfg)
    parsed, _, _, _ = read_up_independently(files['UP1.DAT'])
    assert len(parsed) == len(frame)
    assert parsed.temperature.iloc[1] == 999.9
    assert parsed.wind_direction.iloc[2] == 999
    assert parsed.wind_speed.iloc[3] == 999
    assert parsed.wind_speed.iloc[2] == frame.wind_speed_ms.iloc[2]
    assert report['missing_counts']['temperature'] == 1
    assert report['missing_counts']['wind_direction'] == 1
    assert report['missing_counts']['wind_speed'] == 1
    assert report['missing_policy'] == 'preserve' and report['interpolation'] == 'none'


@pytest.mark.parametrize('change', ['pressure','height','all_temperature','all_wind','surface','top','infinity','unknown_marker','invalid_policy'])
def test_preserve_does_not_hide_required_data_or_invalid_values(case, change):
    frame, cfg = case
    cfg['missing_policy'] = 'preserve'
    if change == 'pressure': frame.loc[1,'pressure_pa'] = np.nan
    elif change == 'height': frame.loc[1,'height_m_msl'] = np.nan
    elif change == 'all_temperature': frame['temperature_c'] = np.nan
    elif change == 'all_wind': frame['wind_speed_ms'] = np.nan
    elif change == 'surface': frame.loc[0,'temperature_c'] = np.nan
    elif change == 'top': frame.loc[37,'wind_speed_ms'] = np.nan
    elif change == 'infinity': frame.loc[1,'temperature_c'] = np.inf
    elif change == 'unknown_marker': frame.loc[1,'temperature_c'] = 32767
    elif change == 'invalid_policy': cfg['missing_policy'] = 'fill'
    with pytest.raises(SchemaError): export_upper_air(frame,cfg)


def test_multiple_stations_units_columns_ll(case):
    frame, cfg = case
    second = frame[(frame.time == frame.time.iloc[0]) & frame.level.isin([1, 38])].copy()
    second['level'] = [1, 2]; second['station_id'] = 12345678
    frame = pd.concat([frame, second], ignore_index=True)
    frame['wind_speed_ms'] *= 3.6
    frame['height_m_msl'] /= 1000
    frame['pressure_pa'] /= 100
    frame = frame.rename(columns={'temperature_c': '温度', 'wind_speed_ms': '风速'})
    cfg['columns']['temperature'] = '温度'; cfg['columns']['wind_speed'] = '风速'
    cfg['units'].update(height='km', pressure='hPa', wind_speed='km/h')
    cfg['projection'] = 'LL'; cfg['datum'] = 'WGS-84'
    cfg['stations'].append(dict(id=12345678, name='TEST', latitude=-35.1, longitude=149.2, elevation_m=345))
    files, report = export_upper_air(frame.sample(frac=1, random_state=42), cfg)
    data, _, _, location = read_up_independently(files['UP2.DAT'])
    assert len(data) == 2 and data.level.tolist() == [1, 2]
    assert location[0] == '12345678' and location[2].endswith('S') and location[3].endswith('E')
    assert data.height.iloc[0] == 345 and abs(data.wind_speed.iloc[0]-1.5) < 1e-10
    assert report['stations'][1]['soundings'] == 1


@pytest.mark.parametrize('field,value,match', [
    ('pressure_pa', -9999, '缺失'), ('temperature_c', np.nan, '缺失'),
    ('height_m_msl', 10000, '范围'), ('wind_speed_ms', -1, '范围'),
    ('wind_from_degree', 999, '范围'), ('station_id', 72469, '站点'),
    ('level', 0, '正整数'), ('time', '2026-01-01 00:00:00', 'UTC'),
    ('time', '2026-01-01T00:01:00Z', '整点'),
])
def test_reject_invalid_rows(case, field, value, match):
    frame, cfg = case
    frame.loc[0, field] = value
    with pytest.raises(SchemaError, match=match): export_upper_air(frame, cfg)


@pytest.mark.parametrize('change,match', [
    ({'version':'CALMET-7'}, 'version'), ({'height_reference':'AGL'}, 'MSL'),
    ({'wind_direction_reference':'towards'}, 'wind_direction_reference'),
    ({'projection':'UTM'}, '投影'), ({'projection':'LL'}, 'datum'),
    ({'missing_values':None}, 'missing_values'), ({'top_pressure_hpa':499}, '顶层'),
    ({'top_pressure_hpa':500.5}, '整数'), ({'sounding_interval_hours':6}, '不连续'),
    ({'units':{}}, '源单位'),
])
def test_reject_bad_config(case, change, match):
    frame, cfg = case
    cfg.update(change)
    with pytest.raises(SchemaError, match=match): export_upper_air(frame, cfg)


def test_duplicate_and_missing_levels(case):
    frame, cfg = case
    with pytest.raises(SchemaError, match='重复'): export_upper_air(pd.concat([frame, frame.iloc[:1]]), cfg)
    with pytest.raises(SchemaError, match='缺层'): export_upper_air(frame.drop(index=3), cfg)


def test_missing_column_and_wrong_dimension(case):
    frame, cfg = case
    with pytest.raises(SchemaError, match='缺少源字段'): export_upper_air(frame.drop(columns='height_m_msl'), cfg)
    cfg['units']['height'] = 'kg'
    with pytest.raises(SchemaError): export_upper_air(frame, cfg)


def test_leap_day_and_year_boundary(case):
    frame, cfg = case
    frame['time'] = frame['time'].str.replace('2026-01-01', '2024-02-29')
    files, _ = export_upper_air(frame, cfg)
    read, dates, _, _ = read_up_independently(files['UP1.DAT'])
    assert dates == [2024, 60, 0, 2024, 60, 12]
    frame.loc[frame.time.str.contains('T00'), 'time'] = '1999-12-31T12:00:00Z'
    frame.loc[frame.time.str.contains('2024'), 'time'] = '2000-01-01T00:00:00Z'
    files, _ = export_upper_air(frame, cfg)
    read, dates, _, _ = read_up_independently(files['UP1.DAT'])
    assert set(read.time.dt.year) == {1999, 2000}


def test_no_silent_grid_or_missing_sounding_fill(case):
    frame, cfg = case
    frame.loc[frame.time.str.contains('T12'), 'time'] = '2026-01-02T00:00:00Z'
    with pytest.raises(SchemaError, match='不连续'): export_upper_air(frame, cfg)
    cfg['sounding_interval_hours'] = None
    files, report = export_upper_air(frame, cfg)
    parsed, _, _, _ = read_up_independently(files['UP1.DAT'])
    assert len(parsed) == 76 and len(parsed.time.unique()) == 2
    assert report['sounding_interval_hours'] is None
