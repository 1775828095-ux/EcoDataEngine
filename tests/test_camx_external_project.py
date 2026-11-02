"""Contract negatives are synthetic; real runs are recorded separately."""
from pathlib import Path
import pytest
from ecodata.interface.camx_external_project import inspect
from ecodata.interface import camx_runner as runner
import io,json,zipfile

def test_real_original_control_resolves_without_rewriting():
    path=runner.CASE/'native_platform_photolysis_5min_20261021/CAMx.in'
    original=path.read_bytes();r=inspect(str(path))
    assert r['grids']==2 and r['minutes']==5 and r['inputs']
    assert r['duplicate_output_species_indexes']==['Output_Species_Names(22)']*2
    assert path.read_bytes()==original and not r['native_executed']

def test_unknown_command_rejected_not_executed(tmp_path):
    p=tmp_path/'CAMx.in';p.write_text(' &CAMx_Control\nrm -rf /\n /\n')
    with pytest.raises(ValueError,match='一行一个完整赋值'):inspect(str(p))

def test_unresolved_shell_variable_rejected(tmp_path):
    p=tmp_path/'CAMx.in';p.write_text(" &CAMx_Control\n Initial_Conditions = '$HOME/ic.nc',\n /\n")
    with pytest.raises(ValueError,match='不执行变量'):inspect(str(p))


def conversion_payload(mechanism=None):
    folder=runner.CASE/'native_complx_emission_probe/B';out=io.BytesIO()
    report=json.loads((folder/'conversion_report.json').read_text())
    if mechanism:report['mechanism']=mechanism
    with zipfile.ZipFile(out,'w') as z:
        z.writestr('conversion_report.json',json.dumps(report));z.writestr('emissions.CAMx7.40.nc',(folder/'emissions.CAMx7.40.nc').read_bytes())
    return out.getvalue()


def test_actual_existing_two_species_only_update_preserves_complement(tmp_path):
    from ecodata.interface.camx_external_project import emission_subset
    path=runner.CASE/'native_platform_photolysis_5min_20261021/CAMx.in'
    engineering=inspect(str(path));ref=runner.CASE/'emiss/camx_area.area.POA_OP.20160610.36km.nc'
    original=runner.sha(ref)
    result=emission_subset(conversion_payload(),ref,engineering,tmp_path/'merged.nc')
    assert result['replaced_species']==['NO','PEC'] and result['preserved_variables'] and runner.sha(ref)==original


def test_different_mechanism_never_translates_species():
    from ecodata.interface.camx_external_project import emission_subset
    with pytest.raises(ValueError,match='不翻译物种'):
        emission_subset(conversion_payload('CB05'),runner.CASE/'emiss/camx_area.area.POA_OP.20160610.36km.nc',dict(inputs=[dict(key='Chemistry_Parameters',path=str(runner.CASE/'inputs/CAMx7.40.chemparam.CB6r5_CF3_COMPLX'))]))
