import csv,io
from unittest.mock import patch
import pytest
from ecodata.interface import cmaq_chemical_profiles as chemistry
from ecodata.interface import independent_cases as cases
from streamlit.testing.v1 import AppTest


def profile(change=None):
    rows=[['VNAME','UNITS','LAYER01','LAYER02'],['PRES','Pa',100000,90000],['ZH','m',10,1000],['O3','ppmV',.04,.05],['NO','ppmV',.001,.002]]
    if change:change(rows)
    out=io.StringIO();csv.writer(out).writerows(rows);return out.getvalue().encode()


def test_complete_profile_preserves_declared_input_without_fill():
    report=chemistry.validate_profile(profile(),{'O3':'ppmV','NO':'ppmV'})
    assert report['layers']==2 and report['required_species']==['NO','O3']


def test_reference_defaults_reject_arbitrary_incomplete_upload():
    with pytest.raises(ValueError,match='逐字节'):
        chemistry.profile_contract(profile(),{'APOCI':'ug/m3'},True)


def test_reference_defaults_require_explicit_profile():
    with pytest.raises(ValueError,match='显式提供'):
        cases.start('cmaq','rebuild',accept_reference_defaults=True)


def test_missing_mechanism_species_stops_before_any_native_tool(tmp_path):
    with patch.object(chemistry.subprocess,'run') as run:
        with pytest.raises(ValueError,match='APOCI'):
            chemistry.build_pair(tmp_path/'new',profile(),{'APOCI':'ug/m3'},tmp_path,'grid',tmp_path/'GRIDDESC')
    run.assert_not_called();assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('mutation,pattern',[
    (lambda r:r.append(r[3]),'重复'),
    (lambda r:r[3].__setitem__(1,'kg/s'),'单位不匹配'),
    (lambda r:r[3].__setitem__(2,'nan'),'非有限'),
    (lambda r:r[3].__setitem__(2,'-1'),'负值'),
    (lambda r:r[1].__setitem__(3,110000),'严格单调'),
    (lambda r:r[2].__setitem__(3,5),'严格单调'),
    (lambda r:r[3].pop(),'长度'),
])
def test_invalid_profile_rejected_with_specific_reason(mutation,pattern):
    with pytest.raises(ValueError,match=pattern):chemistry.validate_profile(profile(mutation),{'O3':'ppmV'})


def test_profile_selection_with_no_upload_does_not_fall_back_to_old_icbc():
    page=AppTest.from_string('from ecodata.interface.cmaq_run_page import render_cmaq_run_page; render_cmaq_run_page()').run()
    page.selectbox[0].set_value(cases.PROFILES['cmaq']['label']).run()
    page.radio[0].set_value(page.radio[0].options[1]).run()
    page.checkbox(key='cmaq_complete_profile_enabled').check().run()
    with patch.object(cases,'start') as start:
        page.button(key='cmaq_independent_start').click().run()
    start.assert_not_called()
    assert not page.exception and not page.success
    assert '不会回退' in page.error[0].value


@pytest.mark.parametrize('model,route',[('camx','rebuild'),('cmaq','existing')])
def test_complete_profile_cannot_silently_change_other_routes(model,route):
    with pytest.raises(ValueError,match='仅用于CMAQ'):
        cases.start(model,route,chemical_profile=profile())


def test_incomplete_profile_start_records_failure_and_never_launches(tmp_path,monkeypatch):
    monkeypatch.setattr(cases,'RUNS',tmp_path/'runs')
    with patch.object(chemistry,'required_species',return_value={'APOCI':'ug/m3'}),patch.object(cases.threading,'Thread') as thread,patch('ecodata.round1.convert') as convert:
        with pytest.raises(ValueError,match='APOCI'):
            cases.start('cmaq','rebuild',chemical_profile=profile())
        thread.assert_not_called();convert.assert_not_called()
    assert cases.ACTIVE is None
    import json
    record=json.loads(next((tmp_path/'runs').glob('*/status.json')).read_text(encoding='utf-8'))
    assert record['state']=='failed' and not record['native_started']
