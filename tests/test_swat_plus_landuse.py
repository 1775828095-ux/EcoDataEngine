from pathlib import Path
import json,io,zipfile
import pytest
from ecodata.models.adapters.swat_plus_landuse import export_landuse
from ecodata.core.exceptions import SchemaError
BASE=Path(__file__).parents[1]/'examples/ai_round1/swat_plus/landuse'
def load(synthetic=False):
    a,c=('A_synthetic_custom_bundle.zip','config_synthetic.json') if synthetic else ('A_official_vegetated_bundle.zip','config_official.json')
    return (BASE/a).read_bytes(),json.loads((BASE/c).read_text(encoding='utf8'))
def mutate(data,func):
    with zipfile.ZipFile(io.BytesIO(data)) as z:files={n:z.read(n) for n in z.namelist()}
    func(files);buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for n,v in files.items():z.writestr(n,v)
    return buf.getvalue()
def test_official_reference_fields_not_raster_claim():
    a,c=load();files,r=export_landuse(a,c)
    actual=[l.split() for l in files['landuse.lum'].decode().splitlines()[2:]]
    expected=[l.split() for l in (BASE/'B2_official_vegetated_landuse.lum').read_text().splitlines()[2:]]
    assert actual==expected and len(actual)==r['records']
    assert set(r['reference_checks'])=={'plnt_com','mgt','cn2','cons_prac','ov_mann'}
    assert 'file.cio' not in files and not r['native_program_read_verified']
def test_custom_columns_and_explicit_null_management():
    a,c=load(True);files,r=export_landuse(a,c)
    rows=[l.split() for l in files['landuse.lum'].decode().splitlines()[2:]]
    assert len(rows)==3 and [x[0] for x in rows]==['synthetic_class_1','synthetic_class_2','synthetic_class_3']
    assert all(len(x)==14 and x[1]=='group_A' and x[3]=='null' for x in rows)
    assert 'mgt' not in r['reference_checks']
def test_official_unused_definitions_with_unresolved_management_rejected():
    _,c=load()
    with pytest.raises(SchemaError,match='mgt引用不存在'):
        export_landuse((BASE/'A_official_unresolved_negative_case.zip').read_bytes(),c)
@pytest.mark.parametrize('problem',['missing_file','unknown_plant','unknown_cn','missing_column','duplicate_name','urban','structure','unsafe_zip','empty_db','placeholder_db','missing_children','invalid_cn','invalid_manning','version','profile'])
def test_fail_concrete_missing_no_guess(problem):
    a,c=load(True)
    def change(files):
        if problem=='missing_file':files.pop('plant.ini')
        elif problem=='unknown_plant':files['A_custom.csv']=files['A_custom.csv'].replace(b'cosy_comm',b'unknown_plant')
        elif problem=='unknown_cn':files['A_custom.csv']=files['A_custom.csv'].replace(b'rc_strow_g',b'unknown_cn')
        elif problem=='missing_column':files['A_custom.csv']=files['A_custom.csv'].replace('曲线数'.encode(),'未知'.encode())
        elif problem=='duplicate_name':files['A_custom.csv']=files['A_custom.csv'].replace(b'synthetic_class_2',b'synthetic_class_1')
        elif problem in {'urban','structure'}:
            text=files['A_custom.csv'].decode();lines=text.splitlines();row=lines[1].split(',');row[6 if problem=='urban' else 9]='invented';lines[1]=','.join(row);files['A_custom.csv']=('\n'.join(lines)+'\n').encode()
        elif problem=='unsafe_zip':files['../bad.txt']=b'not instructions'
        elif problem=='empty_db':files['plant.ini']=b'title\nheader\n'
        elif problem=='placeholder_db':files['plant.ini']=b'title\nheader\ncosy_comm\n'
        elif problem=='missing_children':files['plant.ini']=b'title\nheader\ncosy_comm 2 1\ncorn n 0 0 0 0 0 0\n'
        elif problem=='invalid_cn':files['cntable.lum']=files['cntable.lum'].replace(b'77.00000',b'177.00000')
        elif problem=='invalid_manning':files['ovn_table.lum']=files['ovn_table.lum'].replace(b'0.01000',b'0.00010')
    if problem=='version':c['target_model_version']='60.5.7'
    elif problem=='profile':c['profile']='guess_from_raster'
    else:a=mutate(a,change)
    with pytest.raises(SchemaError):export_landuse(a,c)
