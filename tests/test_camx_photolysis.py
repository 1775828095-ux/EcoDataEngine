from pathlib import Path
import copy, io, json
import numpy as np
import pandas as pd
import pytest
from ecodata.core.exceptions import SchemaError
from ecodata.round1 import convert
from ecodata.models.adapters.camx_photolysis import export_photolysis

ROOT=Path(__file__).resolve().parents[1]
CASE=ROOT/'examples/ai_round1/camx'

@pytest.fixture(scope='module')
def sample():
    return pd.read_csv(CASE/'A_photolysis_table.csv'), json.loads((CASE/'config_photolysis.json').read_text())

def test_actual_official_values_and_fixed_columns(sample):
    frame,cfg=sample
    files,report=convert('camx',(CASE/'A_photolysis_table.csv').read_bytes(),cfg)
    lines=files['photolysis.CAMx7.40.CB6r5.txt'].decode('ascii').splitlines()
    assert lines[0]=='TUV4.8CAMx7.40_CB6r5'
    rate_lines=[line for line in lines if 'CBX' in line]
    actual=np.array([[float(line[1+12*i:1+12*(i+1)]) for i in range(10)] for line in rate_lines])
    # Independent fixed-column read of existing official input, not exporter-derived expectation.
    raw=(CASE/'raw/tuv.do_CB6r5.20160610').read_text(encoding='ascii').splitlines()
    ref=[line for line in raw if 'CBX' in line]
    expected=np.array([[float(line[1+12*i:1+12*(i+1)]) for i in range(10)] for line in ref])
    assert len(rate_lines)==18975
    np.testing.assert_array_equal(actual,expected)
    assert [line.split(':')[-1].strip() for line in rate_lines]==[line.split(':')[-1].strip() for line in ref]
    assert report['units']['photolysis_native_value']=='min^-1' and not report['camx_run_verified']
    assert len(report['block_context_map'])==825

def test_shuffled_rows_and_seconds_conversion(sample):
    frame,cfg=sample
    expected,_=export_photolysis(frame,cfg)
    shuffled,_=export_photolysis(frame.sample(frac=1,random_state=47),cfg)
    assert shuffled==expected
    seconds=frame.copy();seconds['photolysis_native_value']/=60
    scfg=copy.deepcopy(cfg);scfg['units']['photolysis_native_value']='s^-1'
    converted,report=export_photolysis(seconds,scfg)
    assert converted==expected and report['rate_conversion_factor']==60

@pytest.mark.parametrize('defect',['missing','duplicate','nan','negative','reaction','zenith','block','extra','condition'])
def test_corrupt_table_rejected(sample,defect):
    frame,cfg=sample;bad=frame.copy()
    if defect=='missing':bad=bad.iloc[1:]
    elif defect=='duplicate':bad=pd.concat([bad,bad.iloc[:1]],ignore_index=True)
    elif defect=='nan':bad.loc[0,'photolysis_native_value']=np.nan
    elif defect=='negative':bad.loc[0,'photolysis_native_value']=-1
    elif defect=='reaction':bad.loc[0,'reaction']='OTHER.PHF'
    elif defect=='zenith':
        bad['zenith_index']=bad['zenith_index'].astype(float)
        bad.loc[0,'zenith_index']=0.5
    elif defect=='block':bad.loc[0,'block_id']=9999
    elif defect=='extra':bad['unhandled_field']=1
    elif defect=='condition':bad.loc[0,'ozone_column_native']=240
    with pytest.raises(SchemaError):export_photolysis(bad,cfg)

@pytest.mark.parametrize('defect',['version','unit','angles','options'])
def test_ambiguous_configuration_rejected(sample,defect):
    frame,cfg=sample;bad=copy.deepcopy(cfg)
    if defect=='version':bad['version']='CAMx-7.40-CB6r4-photolysis'
    elif defect=='unit':bad['units']['photolysis_native_value']='h^-1'
    elif defect=='angles':bad['zenith_degrees']=list(range(10))
    else:bad['fill_missing']=0
    with pytest.raises(SchemaError):export_photolysis(frame,bad)

def test_duplicate_csv_headers_rejected(sample):
    _,cfg=sample
    with pytest.raises(SchemaError):convert('camx',b'block_id,block_id\n0,0\n',cfg)

def test_condition_coordinates_not_rounded(sample):
    frame,cfg=sample;changed=frame.copy()
    coordinate=0.040000000000123456
    changed.loc[changed.albedo==0.04,'albedo']=coordinate
    files,_=export_photolysis(changed,cfg)
    condition=files['photolysis.CAMx7.40.CB6r5.txt'].decode('ascii').splitlines()[1]
    assert float(condition.split('Albedo=')[1].split()[0])==coordinate
