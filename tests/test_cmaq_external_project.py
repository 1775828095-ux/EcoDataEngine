"""Synthetic control tests; scientific acceptance uses real native runs."""
import pytest
from ecodata.interface.cmaq_external_project import launcher_contract,instrument,inspect_manifest
from ecodata.interface.cmaq_external_project import griddesc_contract,physical_input_contract

def test_capture_excludes_process_secrets_and_keeps_only_explicit_model_keys(tmp_path):
    script='# setenv FAKE secret\nsetenv PATH arbitrary\nsetenv CTM_RUNLEN 000500\nsetenv S_CGRID $OUTDIR/x.nc\n  ( /usr/bin/time -p $BLD/$EXEC ) |& tee buff_${EXECUTION_ID}.txt'
    c=launcher_contract(script)
    assert set(c['keys'])=={'CTM_RUNLEN','S_CGRID'} and c['output_keys']==['S_CGRID']

def test_wrong_engine_never_runs_or_changes_version():
    with pytest.raises(ValueError,match='受控CMAQ'):inspect_manifest(dict(schema='ecodata.cmaq.local-project.v1',engine_sha256='wrong'))

def test_unknown_env_key_rejected_before_dependency_access():
    from ecodata.interface.cmaq_runner import PIN
    with pytest.raises(ValueError,match='非模型配置'):
        inspect_manifest(dict(schema='ecodata.cmaq.local-project.v1',engine_sha256=PIN,mechanism='cb6r5_ae7_aq_m3dry',environment={'LD_PRELOAD':'bad'}))

def test_unresolved_value_not_silently_substituted():
    from ecodata.interface.cmaq_runner import PIN
    with pytest.raises(ValueError,match='已解析'):
        inspect_manifest(dict(schema='ecodata.cmaq.local-project.v1',engine_sha256=PIN,mechanism='cb6r5_ae7_aq_m3dry',environment={'CTM_RUNLEN':'$NSTEPS'}))


def test_griddesc_selects_declared_grid_not_first_record():
    raw=b"' '\n'PROJ'\n2 33 45 -97 -97 40\n' '\n'A'\n'PROJ' 0 0 1000 1000 2 3 1\n'B'\n'PROJ' 10 20 2000 2000 4 5 1\n' '\n"
    assert griddesc_contract(raw,'B')['NCOLS']==4
    with pytest.raises(ValueError,match='GRID_NAME'):griddesc_contract(raw,'missing')


def real_manifest():
    from pathlib import Path
    import json
    path=Path(__file__).resolve().parents[3]/'CMAQ已有工程接入_20261101/platform_exported_project.json'
    if not path.is_file():pytest.skip('Local native acceptance evidence not installed')
    return json.loads(path.read_text(encoding='utf8'))


def test_actual_original_vertical_header_difference_is_never_silently_accepted():
    m=real_manifest()
    with pytest.raises(ValueError,match='INIT_CONC_1.VGTYP'):physical_input_contract(m['environment'],m['inputs'])
    r=physical_input_contract(m['environment'],m['inputs'],True)
    assert {i['key'] for i in r['vertical_header_differences']}=={'INIT_CONC_1','BNDY_CONC_1'}
    assert not r['physical_equivalence_certified'] and r['preserve_original_vertical_headers']


def test_declared_active_point_stream_cannot_be_missing():
    m=real_manifest();inputs=dict(m['inputs']);del inputs['STK_EMIS_001']
    with pytest.raises(ValueError,match='活动模块的必需输入缺失'):physical_input_contract(m['environment'],inputs,True)
