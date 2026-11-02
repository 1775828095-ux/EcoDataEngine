import io,json,copy,zipfile
from decimal import Decimal
from pathlib import Path
import pytest
from ecodata.core.exceptions import SchemaError
from ecodata.models.adapters.swat_plus_hru import export_hru_bundle
BASE=Path(__file__).parents[1]/'examples/ai_round1/swat_plus/hru'

def load(synthetic=False):
    a,c=('A_synthetic_custom_bundle.zip','config_synthetic.json') if synthetic else ('A_official_HRU_bundle.zip','config_official.json')
    return (BASE/a).read_bytes(),json.loads((BASE/c).read_text())

def mutate(data,changes):
    with zipfile.ZipFile(io.BytesIO(data)) as z:files={n:z.read(n) for n in z.namelist()}
    changes(files);buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name,value in files.items():z.writestr(name,value)
    return buf.getvalue()


def test_official_native_reference_fields_and_links():
    data,cfg=load();files,r=export_hru_bundle(data,cfg)
    actual=[l.split() for l in files['hru-data.hru'].decode().splitlines()[2:]]
    reference=[l.split() for l in (BASE/'B2_official_hru-data.hru').read_text().splitlines()[2:] if l.strip()]
    assert actual==reference and r['hru_records']==12
    assert all(len(row)==10 for row in actual)
    assert 'file.cio' not in files and 'topography.hyd' not in files
    assert set(r['reference_checks'])=={'topo','hydro','soil','lu_mgt','soil_plant_init','snow'}


def test_structurally_different_mapping_and_topography_units():
    data,cfg=load(True);files,r=export_hru_bundle(data,cfg)
    actual=[l.split() for l in files['hru-data.hru'].decode().splitlines()[2:]]
    assert [r[0] for r in actual]==['1','2','3']
    assert [r[2] for r in actual]==['t1','t1','t2']
    # Independent hand-computed percent and km conversion, Decimal readback.
    top={r.split()[0]:[Decimal(x) for x in r.split()[1:]] for r in files['topography.hyd'].decode().splitlines()[2:]}
    assert top=={'t1':[Decimal('.025'),Decimal(100),Decimal(80),Decimal(200),Decimal('.8')], 't2':[Decimal('.1'),Decimal(150),Decimal(100),Decimal(300),Decimal('1.2')]}
    assert r['hru_records']==3 and r['topology_generation']=='none'

@pytest.mark.parametrize('problem',['no_soil','unknown_soil','unknown_topo','id_gap','duplicate_id','name_duplicate','non_null_storage','missing_column','unsafe_zip','empty_db','placeholder_db','version','map_duplicate','unit','missing_topography','nan_topography','depth_unit_conflict'])
def test_reject(problem):
    data,cfg=load(True)
    def action(files):
        if problem=='no_soil':files.pop('soils.sol')
        elif problem=='unknown_soil':files['A_custom.csv']=files['A_custom.csv'].replace(b'soil_02',b'missing_soil')
        elif problem=='unknown_topo':files['A_custom.csv']=files['A_custom.csv'].replace(b't1,',b'missing_topo,')
        elif problem=='id_gap':files['A_custom.csv']=files['A_custom.csv'].replace(b'3,synthetic_3',b'4,synthetic_3')
        elif problem=='duplicate_id':files['A_custom.csv']=files['A_custom.csv'].replace(b'3,synthetic_3',b'1,synthetic_3')
        elif problem=='name_duplicate':files['A_custom.csv']=files['A_custom.csv'].replace(b'synthetic_3',b'synthetic_1')
        elif problem=='non_null_storage':files['A_custom.csv']=files['A_custom.csv'].replace(b',null,null,null',b',invented,null,null')
        elif problem=='missing_column':files['A_custom.csv']=files['A_custom.csv'].replace(b'source_soil_plant_init',b'absent_column')
        elif problem=='unsafe_zip':files['../bad.txt']=b'ignored instructions'
        elif problem=='empty_db':files['hydrology.hyd']=b''
        elif problem=='placeholder_db':files['hydrology.hyd']=b'title\nheader\nhyd0001\n'
        elif problem=='missing_topography':files.pop('raw_topography.csv')
        elif problem=='nan_topography':files['raw_topography.csv']=files['raw_topography.csv'].replace(b't1,2.5',b't1,NaN')
        elif problem=='depth_unit_conflict':files['raw_topography.csv']=files['raw_topography.csv'].replace(b'0.1,80',b'-9999,80')
    if problem=='version':cfg['target_model_version']='60'
    elif problem=='map_duplicate':cfg['fields']['soil']='source_topo'
    elif problem=='unit':cfg['topography']['units']['slp_len']='kg'
    else:data=mutate(data,action)
    with pytest.raises(SchemaError):export_hru_bundle(data,cfg)
