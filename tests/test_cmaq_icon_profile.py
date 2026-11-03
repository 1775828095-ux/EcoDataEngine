from pathlib import Path
import copy,json,io,csv
import numpy as np
import pandas as pd
import pytest
from ecodata.round1 import convert
from ecodata.models.adapters.cmaq_icon_profile import export_icon_profile
from ecodata.core.exceptions import SchemaError
CASE=Path(__file__).resolve().parents[1]/'examples/ai_round1/cmaq/icon_profile'
@pytest.fixture(scope='module')
def sample():return pd.read_csv(CASE/'A_profile_long.csv'),json.loads((CASE/'config.json').read_text())
def native(data):return list(csv.reader(line for line in data.decode('ascii').splitlines() if not line.startswith('#')))
def test_official_reference_and_cli(sample):
    frame,cfg=sample
    files,report=convert('cmaq',(CASE/'A_profile_long.csv').read_bytes(),cfg)
    got=native(files['IC_PROFILE_CMAQ55.csv']);ref=native((CASE/'reference_official.csv').read_bytes())
    assert got[0]==ref[0] and len(got)==252
    assert [r[:2] for r in got[1:]]==[r[:2] for r in ref[1:]]
    np.testing.assert_array_equal(np.array([r[2:] for r in got[1:]],float),np.array([r[2:] for r in ref[1:]],float))
    assert report['values']==11044 and not report['ICON_run_verified']
def test_order_and_unit_conversion(sample):
    frame,cfg=sample;expected,_=export_icon_profile(frame,cfg)
    shuffled,_=export_icon_profile(frame.sample(frac=1,random_state=4),cfg);assert shuffled==expected
    changed=frame.copy()
    for name,unit,factor in [('O3','ppbV',1000),('PRES','hPa',0.01),('ZH','km',0.001),('ZF','km',0.001)]:
        mask=changed.variable==name;changed.loc[mask,'value']*=factor;changed.loc[mask,'unit']=unit
    files,report=export_icon_profile(changed,cfg)
    expected_rows=native(expected['IC_PROFILE_CMAQ55.csv']);got=native(files['IC_PROFILE_CMAQ55.csv'])
    np.testing.assert_allclose(np.array([r[2:] for r in got[1:]],float),np.array([r[2:] for r in expected_rows[1:]],float),rtol=2e-15,atol=0)
    assert report['conversion_factors']['O3']==.001
@pytest.mark.parametrize('defect',['missing','duplicate','nan','negative','height','unit','unknown','extra'])
def test_bad_input_rejected(sample,defect):
    frame,cfg=sample;bad=frame.copy()
    if defect=='missing':bad=bad.iloc[1:]
    elif defect=='duplicate':bad=pd.concat([bad,bad.iloc[:1]])
    elif defect=='nan':bad.loc[0,'value']=np.nan
    elif defect=='negative':bad.loc[0,'value']=-1
    elif defect=='height':bad.loc[(bad.variable=='ZH')&(bad.layer==2),'value']=1
    elif defect=='unit':bad.loc[0,'unit']='ug/m3'
    elif defect=='unknown':bad.loc[0,'variable']='UNKNOWN'
    else:bad['time']='2016'
    with pytest.raises(SchemaError):export_icon_profile(bad,cfg)
@pytest.mark.parametrize('defect',['version','layer_count','injection'])
def test_configuration_rejected(sample,defect):
    frame,cfg=sample;bad=copy.deepcopy(cfg)
    if defect=='version':bad['version']='CMAQ-5.0.2-CB05'
    elif defect=='layer_count':bad['layers']=43
    else:bad['source_description']='source\nwrong'
    with pytest.raises(SchemaError):export_icon_profile(frame,bad)
