import io,json,zipfile
from decimal import Decimal
from pathlib import Path
import pytest
from ecodata.round1 import convert
from ecodata.core.exceptions import SchemaError
BASE=Path(__file__).parents[1]/'examples/ai_round1/swat_plus/management'
def case(label='standard'):
    return (BASE/f'A_synthetic_{label}.zip').read_bytes(),json.loads((BASE/f'config_{label}.json').read_text(encoding='utf8'))
def change(data,name,func):
    raw=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as z,zipfile.ZipFile(raw,'w') as out:
        for n in z.namelist():out.writestr(n,func(z.read(n)) if n==name else z.read(n))
    return raw.getvalue()

@pytest.mark.parametrize('label,expected',[('standard',{'field_a':[('fert',3,15,'elem_n','broadcast',Decimal('50')),('manu',6,1,'beef_salt','band',Decimal('2500'))]}),('mapped',{'field_a':[('fert',3,15,'elem_n','broadcast',Decimal('50')),('manu',6,1,'beef_salt','band',Decimal('250'))],'field_b':[('fert',10,10,'elem_p','broadcast',Decimal('100'))]})])
def test_independent_native_schedule_reader_units_counts_and_refs(label,expected):
    data,cfg=case(label);files,r=convert('swat_plus',data,cfg)
    lines=files['management.sch'].decode('ascii').splitlines()[2:];i=0;found={}
    while i<len(lines):
        name,nops,nauto=lines[i].split();i+=1;assert nauto=='0';found[name]=[]
        for _ in range(int(nops)):
            op,mon,day,hu,material,application,amount=lines[i].split();i+=1
            assert Decimal(hu)==0
            found[name].append((op,int(mon),int(day),material,application,Decimal(amount)))
    assert found==expected and r['amount_unit']=='kg/ha'
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for n in ('fertilizer.frt','manure_db.frt','manure_om.frt','chem_app.ops'):assert files[n]==z.read(n)
    assert 'file.cio' not in files

@pytest.mark.parametrize('error,match',[('column','缺列'),('reference','材料引用'),('application','施用方式'),('order','连续唯一'),('date','月日'),('hu','热量触发'),('negative','负数'),('marker','缺测标记'),('unknown','缺operation_mapping'),('unit','单位'),('manure_om','org_min引用'),('db','缺完整字段'),('db_nan','含缺测')])
def test_missing_invalid_input_does_not_create_management(error,match):
    data,cfg=case()
    def mutate(b):
        if error=='column':return b.replace(b'amount_unit',b'unknown_column')
        if error=='reference':return b.replace(b'elem_n',b'missing_fert')
        if error=='application':return b.replace(b'broadcast',b'missing_app')
        if error=='order':return b.replace(b'field_a,2',b'field_a,1')
        if error=='date':return b.replace(b',3,15,',b',2,29,')
        if error=='hu':return b.replace(b',15,0,',b',15,1,')
        if error=='negative':return b.replace(b',50,',b',-1,')
        if error=='marker':return b.replace(b',50,',b',-9999,')
        if error=='unknown':return b.replace(b'fertilize',b'irrigate')
        if error=='unit':return b.replace(b'kg/ha',b'K')
        if error=='manure_om':return b.replace(b'mw_beef_liq',b'missing_om')
        if error=='db':return b'title\nheader\nelem_n\n'
        if error=='db_nan':return b.replace(b'1.00000',b'nan',1)
        return b
    name='manure_db.frt' if error=='manure_om' else 'fertilizer.frt' if error in {'db','db_nan'} else 'operations.csv'
    with pytest.raises(SchemaError,match=match):convert('swat_plus',change(data,name,mutate),cfg)

@pytest.mark.parametrize('key,value,match',[('automatic_operations',['auto_plant'],'自动'),('target_model_version','60','62.0.0'),('references',{},'引用文件'),('units',{},'单位')])
def test_unsupported_controls_rejected(key,value,match):
    data,cfg=case();cfg[key]=value
    with pytest.raises(SchemaError,match=match):convert('swat_plus',data,cfg)
