"""Dependency behavior tests; fixtures are synthetic, not environmental evidence."""
import io
import json
import zipfile
import pytest

from ecodata.models.project_inventory import inspect_projects,prepare_project_inventory
from ecodata.algorithms.automatic_preparation import prepare_uploads


def uploaded(files):
    blob=io.BytesIO();blob.name='renamable.zip'
    with zipfile.ZipFile(blob,'w') as z:
        for name,data in files.items():z.writestr(name,data)
    return blob


def swat(root='watershed'):
    names=('simulation basin climate connect channel reservoir routing_unit hru exco recall dr aquifer '
           'herd water_rights link hydrology structural hru_parm_db ops lum chg init soils decision_table regions').split()
    values={n:['null'] for n in names}
    values.update(simulation=['clock.dat'],basin=['null']*2,
                  climate=['stations.dat','null','null','rain.cli','null','null','null','null','null'],connect=['objects.con'])
    # Labels deliberately differ: official readcio_read consumes slot positions.
    cio='SWAT+ rev.60.5.7\n'+'\n'.join('label_'+str(i)+' '+' '.join(values[n]) for i,n in enumerate(names))
    return {root+'/file.cio':cio,root+'/clock.dat':'time\na b c d e\n0 2017 0 2017 0',
            root+'/stations.dat':'stations\nname wgn pcp tmp slr hmd wnd\nS null rain.pcp null null null null',
            root+'/objects.con':'objects\nid name wst\n1 hru S',
            root+'/rain.cli':'rain\nfilename\nrain.pcp',root+'/rain.pcp':'real input placeholder for format-only test'}


def test_swat_position_identity_and_nested_weather_edges():
    result=inspect_projects([uploaded(swat())],'swat_plus')
    p=result['projects'][0];edges=p['bindings']
    assert any(r['parameter']=='simulation.0' and r['target']=='watershed/clock.dat' for r in edges)
    assert any(r['parameter']=='weather_index:rain.cli' and r['target']=='watershed/rain.pcp' for r in edges)
    assert result['native_schema_verified'] is False


def test_missing_weather_and_changed_resource_not_silent():
    a=swat();b=dict(a);b.pop('watershed/rain.pcp')
    x=inspect_projects([uploaded(a)],'swat_plus')['projects'][0]
    y=inspect_projects([uploaded(b)],'swat_plus')['projects'][0]
    assert sum(r['status']=='UNRESOLVED' for r in y['bindings'])==1
    a['watershed/rain.pcp']='changed input'
    z=inspect_projects([uploaded(a)],'swat_plus')['projects'][0]
    get=lambda p:next(r['sha256'] for r in p['bindings'] if r['parameter']=='weather_index:rain.cli')
    assert get(x)!=get(z)


def test_multiple_complete_roots_are_reported_without_guessing_winner():
    files={**swat('one'),**swat('two')}
    r=inspect_projects([uploaded(files)],'swat_plus')
    assert len(r['projects'])==2
    assert {e['target'].split('/')[0] for e in r['projects'][0]['bindings']}=={'one'}


def test_unsupported_swat_version_is_unknown_not_pass():
    f=swat();f['watershed/file.cio']=f['watershed/file.cio'].replace('60.5.7','99.0')
    p=inspect_projects([uploaded(f)],'swat_plus')['projects'][0]
    assert p['unknown'] and not p['bindings']


def test_cmaq_never_reads_host_environment_or_executes_script(tmp_path,monkeypatch):
    monkeypatch.setenv('PRIVATE_INPUT','/private/do-not-read')
    script='#!/bin/csh\nsetenv MET_CRO_2D "$PRIVATE_INPUT/met.nc"\nsetenv GRID_CRO_2D "grid.nc"\necho run > sentinel\n'
    f={'job.csh':script,'grid.nc':b'not native; binding evidence only'}
    r=prepare_uploads([uploaded(f)],'cmaq',tmp_path)
    assert r['manifest']['status']=='ENGINEERING_INSPECTED' and not r['manifest']['native_ready']
    with zipfile.ZipFile(r['archive']) as z:p=json.loads(z.read('project_inventory.json'))['projects'][0]
    assert p['bindings'][0]['status']=='UNKNOWN' and '$PRIVATE_INPUT' in p['bindings'][0]['original']
    assert p['bindings'][1]['status']=='BOUND' and p['bindings'][1]['activation']=='UNKNOWN'
    assert not (tmp_path/'sentinel').exists()


def test_public_swat_entry_reports_partial_dependency_scope(tmp_path):
    r=prepare_uploads([uploaded(swat())],'swat_plus',tmp_path)
    assert r['manifest']['status']=='ENGINEERING_INSPECTED'
    with zipfile.ZipFile(r['archive']) as z:
        assert z.testzip() is None
        assert json.loads(z.read('project_inventory.json'))['status']=='PARTIAL'


def test_unrelated_names_order_and_zip_name_do_not_change_scientific_edges():
    f=swat();a=uploaded(f);b=uploaded(dict(reversed(list(f.items()))));b.name='other-name.zip'
    x=inspect_projects([a],'swat_plus')['projects'][0]
    y=inspect_projects([b],'swat_plus')['projects'][0]
    assert x['bindings']==y['bindings']


def test_no_catchall_claim_for_unknown_project():
    assert inspect_projects([uploaded({'arbitrary.project':'unknown syntax'})],'swat_plus') is None
    with pytest.raises(ValueError,match='不安全'):
        inspect_projects([uploaded({'../file.cio':'unsafe'})],'swat_plus')


def test_cmaq_quoted_hash_not_mistaken_for_comment():
    r=inspect_projects([uploaded({'job.csh':'setenv GRID_CRO_2D "grid#actual.nc" # comment',
                                 'grid#actual.nc':'actual bytes','grid':'wrong bytes'})],'cmaq')
    assert r['projects'][0]['bindings'][0]['target']=='grid#actual.nc'


def test_page_partial_state_is_visible_and_downloadable(tmp_path):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    p=tmp_path/'input';p.mkdir();(p/'job.csh').write_text('setenv MET_CRO_2D "$UNDEFINED/data.nc"')
    app=AppTest.from_file(Path(__file__).parents[1]/'src/ecodata/interface/app.py').run(timeout=30)
    app.selectbox[0].select('CMAQ').run()
    app.text_input[0].set_value(str(p)).run()
    app.button[0].click().run(timeout=30)
    assert not app.exception and app.warning
    assert app.get('download_button')[0].label=='下载工程解析与缺项记录'
    assert not app.session_state['workbench_result']['data']['manifest']['native_ready']
