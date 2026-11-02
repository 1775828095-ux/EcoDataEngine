"""Missing-input mutations are test fixtures, never replacement scientific data."""
from pathlib import Path
import json
import pytest
from ecodata.interface import engineering_completeness as c


def test_cmaq_collects_all_missing_streams_and_does_not_guess_counts():
    r=c.cmaq({'environment':{'N_EMIS_GR':'2','N_EMIS_PT':'1'},'inputs':{}})
    missing={i['item'] for i in r['findings'] if i['status']=='missing'}
    assert {'GR_EMIS_001','GR_EMIS_002','STK_GRPS_001','STK_EMIS_001','MET_CRO_2D'}<=missing
    with pytest.raises(c.IncompleteProject) as exc:c.require(r)
    assert exc.value.completeness is r
    r=c.cmaq({'environment':{},'inputs':{}})
    assert any(i['item']=='N_EMIS_PT' and i['status']=='unknown' and i['blocks_run'] for i in r['findings'])


def test_cmaq_disabled_is_based_on_explicit_flag_only():
    m={'environment':{'N_EMIS_GR':'0','N_EMIS_PT':'0','CTM_DDM3D':'N'},
       'unavailable_inputs':{'INIT_SENS_1':'/'},'inputs':{}}
    assert next(i for i in c.cmaq(m)['findings'] if i['item']=='INIT_SENS_1')['status']=='not_required'
    m['environment']['CTM_DDM3D']='Y'
    assert next(i for i in c.cmaq(m)['findings'] if i['item']=='INIT_SENS_1')['blocks_run']


def test_unknown_unavailable_input_is_not_assumed_optional():
    r=c.cmaq({'environment':{},'inputs':{},'unavailable_inputs':{'UNREVIEWED':'/x'}})
    item=next(i for i in r['findings'] if i['item']=='UNREVIEWED')
    assert item['status']=='unknown' and item['blocks_run']


def test_camx_every_nested_grid_requires_weather_and_enabled_emissions(tmp_path):
    v={'Number_of_Grids':'2','Gridded_Emissions':'.true.','Point_Emissions':'.true.',
       'Restart':'.false.','Master_Grid_Restart':"'absent.old'"}
    r=c.camx(tmp_path/'control',v)
    blocked={i['item'] for i in r['findings'] if i['blocks_run']}
    assert {'Met3D_Grid(1)','Met3D_Grid(2)','Emiss_Grid(2,*)','Point_Sources(*)'}<=blocked
    assert next(i for i in r['findings'] if i['item']=='Master_Grid_Restart')['status']=='not_required'


def test_camx_empty_file_does_not_count_as_available(tmp_path):
    (tmp_path/'empty').write_bytes(b'')
    r=c.camx(tmp_path/'control',{'Initial_Conditions':"'empty'",'Number_of_Grids':'1'})
    assert next(i for i in r['findings'] if i['item']=='Initial_Conditions')['status']=='missing'


def test_wrf_wps_dependencies_only_required_when_updating():
    v={'native_inputs':{'wrfinput_d01':{},'wrfbdy_d01':{}},'errors':[],
       'replacement_missing':['geo_em.d01.nc'],'replacement_checked':False}
    assert c.wrf(v,False)['can_proceed']
    assert not c.wrf(v,True)['can_proceed']


def test_swat_all_existing_validation_failures_are_preserved():
    r=c.swat({}, {'errors':['缺file.cio','缺天气站'],'warnings':['未检查深层拓扑']})
    assert not r['can_proceed'] and len([i for i in r['findings'] if i['blocks_run']])==3
    assert r['limitations']==['未检查深层拓扑']


def test_calpuff_reports_both_missing_weather_files_without_raw_gis_requirement():
    raw=b'! METDAT = met.dat !\n! AUXEXT = AUX !'
    r=c.calpuff({'case/puff.inp':raw},'case/puff.inp')
    missing={i['item'] for i in r['findings'] if i['status']=='missing'}
    assert missing=={'case/met.dat','case/met.dat.AUX'}
    assert not any('DEM' in i['item'] for i in r['findings'])


def test_real_cmaq_dependency_report_read_only():
    p=Path(__file__).resolve().parents[3]/'CMAQ已有工程接入_20261101/platform_exported_project.json'
    if not p.exists():pytest.skip('local acceptance data unavailable')
    raw=p.read_bytes();r=c.cmaq(json.loads(raw))
    assert r['can_proceed'],[i for i in r['findings'] if i['blocks_run']]
    assert p.read_bytes()==raw and not r['native_read_verified']


def test_render_exposes_download_and_blockers():
    from streamlit.testing.v1 import AppTest
    script='''from ecodata.interface.engineering_completeness import report,finding,render
r=report('CMAQ',[finding('MET_CRO_3D','missing','工程当前引用','补充气象文件')])
render(r,'test')
import streamlit as st
st.button('运行',disabled=not r['can_proceed'])
'''
    app=AppTest.from_string(script).run()
    assert not app.exception and app.button[0].disabled
    assert len(app.dataframe)==1 and len(app.error)==1
