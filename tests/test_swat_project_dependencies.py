"""Synthetic dependency fixtures only; these are not simulated basins."""
from ecodata.models.swat_project_dependencies import weather_dependencies
from ecodata.interface.swat_runner import check_project


def fixture():
    return {'file.cio': b'fixture\nclimate stations.cli generators.cli null null null null null null null\nconnect custom-network.dat\nsimulation period.sim null null\n',
            'period.sim': b'fixture\nfields\n1 2021 2 2021 0\n',
            'stations.cli': b'fixture\nname wgn pcp tmp slr hmd wnd pet atmo_dep\na g sim sim sim sim sim null null\nb g sim sim sim sim sim null null\n',
            'generators.cli': ('fixture\ng 1 2 3 10\n' + ' '.join('field'+str(i) for i in range(14))+'\n'+(' '.join(['1']*14)+'\n')*12).encode(),
            'custom-network.dat': b'fixture\nid name wst\n1 first a\n2 second b\n3 third a\n'}


def test_explicit_object_station_links_follow_referenced_file():
    files = fixture()
    original = dict(files)
    report = weather_dependencies(files, 'a')
    assert not report['errors']
    assert report['stations']['a']['object_count'] == 2
    assert report['stations']['b']['object_count'] == 1
    assert report['wgn_stations']['g']['months'] == 12
    assert files == original


def test_missing_object_weather_name_is_rejected_by_runner():
    files = fixture()
    files['custom-network.dat'] += b'4 bad unknown\n'
    assert any('不存在的天气站' in e for e in check_project(files)['errors'])


def test_truncated_wgn_previously_missed_is_rejected():
    files = fixture()
    files['generators.cli'] = b'\n'.join(files['generators.cli'].splitlines()[:4])+b'\n'
    report = weather_dependencies(files)
    assert any('12个月统计，实际只有1行' in e for e in report['errors'])
    assert any('12个月统计' in e for e in check_project(files)['errors'])


def test_unknown_wgn_name_is_not_a_filename_reference():
    files = fixture()
    files['stations.cli'] = files['stations.cli'].replace(b'a g ', b'a other ')
    assert any('WGN统计站不存在：other' in e for e in weather_dependencies(files)['errors'])


def test_unrecognised_connection_format_is_reported_not_guessed():
    files = fixture()
    files['custom-network.dat'] = b'fixture\nunknown headers\nsome records\n'
    report = weather_dependencies(files)
    assert not report['errors']
    assert any('未识别' in w for w in report['warnings'])
    assert report['object_station_links'] == []


def test_special_wst_and_unused_station_are_explicit_warnings():
    files = fixture()
    files['custom-network.dat'] = b'fixture\nid name wst\n1 first 0\n'
    report = weather_dependencies(files, 'a')
    assert not report['errors']
    assert any('特殊wst' in w for w in report['warnings'])
    assert any('没有已识别的显式对象引用' in w for w in report['warnings'])


def test_duplicate_ids_and_nonfinite_months_rejected():
    files = fixture()
    files['custom-network.dat'] += b'1 duplicate a\n'
    files['generators.cli'] = files['generators.cli'].replace(b'1 1 1', b'nan 1 1', 1)
    report = weather_dependencies(files)
    assert any('对象ID重复' in e for e in report['errors'])
    assert any('有限数值' in e for e in report['errors'])


def test_multiple_generators_and_missing_last_month():
    files = fixture()
    files['generators.cli'] += files['generators.cli'].split(b'\n', 1)[1].replace(b'g 1', b'h 1', 1)
    files['stations.cli'] = files['stations.cli'].replace(b'b g ', b'b h ')
    assert set(weather_dependencies(files)['wgn_stations']) == {'g', 'h'}
    files['generators.cli'] = files['generators.cli'].rsplit(b'\n', 2)[0] + b'\n'
    assert any('实际只有11行' in e for e in weather_dependencies(files)['errors'])


def test_editor_header_declares_target_model_not_editor_version():
    files = fixture()
    files['file.cio'] = files['file.cio'].replace(b'fixture', b'written by editor v4.0.2 for SWAT+ 62.0.0', 1)
    report = check_project(files, engine_id='official61')
    assert report['declared_major'] == 62
    assert any('版本不匹配' in e for e in report['errors'])


def test_editor_csv_contract_preserves_months_and_each_named_value():
    import csv
    import io
    from ecodata.interface.swat_build import editor_wgn_csv, WGN_FIELDS
    # Distinct values expose a positional shift; not scientific simulation data.
    source = io.StringIO()
    writer = csv.writer(source)
    writer.writerow(['id', 'month', *WGN_FIELDS, 'wgn_id'])
    for month in range(1, 13):
        writer.writerow([100+month, month, *[month*100+i for i in range(14)], 24688])
    stations, months = editor_wgn_csv(b'id,name,lat,lon,elev,rain_yrs\n24688,g,1,2,3,10\n', source.getvalue().encode())
    rows = list(csv.reader(io.StringIO(months.decode())))
    assert rows[0] == ['id', 'weather_wgn_cli_id', 'month', *WGN_FIELDS]
    for month, row in enumerate(rows[1:], 1):
        assert row[1:3] == ['24688', str(month)]
        assert list(map(float, row[3:])) == [month*100+i for i in range(14)]
