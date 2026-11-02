import csv,io,json,zipfile
from decimal import Decimal
from pathlib import Path
import pytest
from ecodata.round1 import convert
from ecodata.core.exceptions import SchemaError

BASE=Path(__file__).parents[1]/'examples/ai_round1/swat_plus/connections'

def load(case='official'):
    return (BASE/f'A_{case}.zip').read_bytes(),json.loads((BASE/f'config_{case}.json').read_text())

def parse(raw):
    records={}
    for text in raw.decode('ascii').splitlines()[2:]:
        values=text.split();count=int(values[12]);assert len(values)==13+4*count
        records[int(values[0])]=values
    return records

def test_four_official_connections_independent_b2():
    data,cfg=load();files,report=convert('swat_plus',data,cfg)
    assert report['nodes']==4 and report['edges']==3 and 'file.cio' not in files
    numeric={0,2,3,4,5,6,7,9,10,12}
    for name in ('hru.con','aquifer.con','rout_unit.con','chandeg.con'):
        actual=parse(files[name]);reference=parse((BASE/f'B2_{name}').read_bytes())
        assert actual.keys()==reference.keys()
        for ident,row in actual.items():
            ref=reference[ident];assert len(row)==len(ref)
            for i,(a,b) in enumerate(zip(row,ref)):
                if i in numeric or i>=13 and (i-13)%4 in {1,3}:assert abs(Decimal(a)-Decimal(b))<Decimal('1e-12')
                else:assert a==b

def test_projected_shuffled_alternate_columns():
    data,cfg=load('synthetic_projected');files,report=convert('swat_plus',data,cfg)
    # Independent analytic Web Mercator equator: x=R*pi/180 => longitude=1 degree.
    for name in ('hru.con','aquifer.con','rout_unit.con','chandeg.con'):
        row=parse(files[name])[1]
        assert float(row[3])==10 and abs(float(row[4]))<1e-10 and abs(float(row[5])-1)<1e-10
        assert all(float(row[i])==1 for i in range(16,len(row),4))

def mutate(content,member,operation):
    out=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(content)) as z,zipfile.ZipFile(out,'w') as dest:
        for name in z.namelist():
            raw=z.read(name)
            if name==member:
                reader=csv.DictReader(io.StringIO(raw.decode()));rows=list(reader);fields=reader.fieldnames
                operation(rows);b=io.StringIO();writer=csv.DictWriter(b,fieldnames=fields);writer.writeheader();writer.writerows(rows);raw=b.getvalue().encode()
            dest.writestr(name,raw)
    return out.getvalue()

@pytest.mark.parametrize('problem',['field','unit','target','weather','property','fraction','duplicate','advanced','missing'])
def test_reject_invalid_prerequisites(problem):
    data,cfg=load()
    if problem=='field':cfg['nodes']['fields'].pop('props')
    if problem=='unit':cfg['units']['area']='kelvin'
    if problem=='target':data=mutate(data,'edges.csv',lambda r:r[0].update(to_id='99'))
    if problem=='weather':data=mutate(data,'nodes.csv',lambda r:r[0].update(wst='missing'))
    if problem=='property':data=mutate(data,'nodes.csv',lambda r:r[0].update(props='999'))
    if problem=='fraction':data=mutate(data,'edges.csv',lambda r:r[0].update(fraction='0.9'))
    if problem=='duplicate':data=mutate(data,'nodes.csv',lambda r:r.append(dict(r[0])))
    if problem=='advanced':data=mutate(data,'nodes.csv',lambda r:r[0].update(ovfl='1'))
    if problem=='missing':data=mutate(data,'nodes.csv',lambda r:r[0].update(elev='NaN'))
    with pytest.raises((SchemaError,ValueError)):convert('swat_plus',data,cfg)
