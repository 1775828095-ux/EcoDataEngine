import io,json,re,zipfile
from pathlib import Path
import numpy as np
import pytest
from ecodata.round1 import convert
from ecodata.core.exceptions import SchemaError
BASE=Path(__file__).parents[1]/'examples/ai_round1/calpuff/points'
def case(label='NAD27'):
    return (BASE/f'A_synthetic_{label}.zip').read_bytes(),json.loads((BASE/f'config_{label}.json').read_text(encoding='utf8'))
def change(data,name,fn):
    b=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as z,zipfile.ZipFile(b,'w') as out:
        for n in z.namelist():out.writestr(n,fn(z.read(n)) if n==name else z.read(n))
    return b.getvalue()
@pytest.mark.parametrize('label,count',[('NAD27',1),('mapped_WGS84',2)])
def test_independent_active_records_and_mass_units(label,count):
    data,cfg=case(label);files,r=convert('calpuff',data,cfg)
    with zipfile.ZipFile(io.BytesIO(data)) as z:source=z.read('supplied.INP').decode()
    text=files['CALPUFF.INP'].decode('ascii')
    # Independent delimiter scanning, not the converter token helper.
    start=text.index('! NPT1');end=text.index('NAR1');region=text[start:end]
    chunks=region.split('!');x=[]
    for chunk in chunks[1::2]:
        if chunk.strip().startswith('X ='):
            cells=chunk.split('=',1)[1].split(',');assert cells[7].strip()=='0.0'
            x.append([float(v) for v in cells])
    assert len(x)==count and len(x[0])==15
    for i,row in enumerate(x):np.testing.assert_allclose(row,[340.5+i,4870.5+i,40,160,3,5,355,0,10,0,4,1,0,0,10],rtol=0,atol=1e-12)
    assert '! IPTU = 1 !' in text and r['species_order']==['SO2','SO4','NO','NO2','HNO3','NO3','PM10']
    assert text[:text.index('! NPT1')]==source[:source.index('!  NPT1')]
    assert text[text.index('NAR1'):]==source[source.index('NAR1'):]
    assert r['model_run_verified'] is False
@pytest.mark.parametrize('problem,match',[('missing_species','每个控制物种'),('column','缺列'),('unit','排放率'),('crs','CRS'),('negative','不能负'),('marker','缺测'),('downwash','无下洗'),('unemitted','不排放'),('unknown_group','未知记录'),('variable','变排放'),('coordinates','超出控制网格')])
def test_missing_invalid_parameters_rejected(problem,match):
    data,cfg=case()
    if problem=='missing_species':cfg['species'].pop()
    elif problem=='column':cfg['fields']['diameter']='missing'
    elif problem=='unit':cfg['species'][0]['unit']='t/year'
    elif problem=='crs':cfg['source_crs']='EPSG:32619'
    elif problem in {'negative','marker','unemitted','downwash','coordinates'}:
        import pandas as pd
        def edit(b):
            f=pd.read_csv(io.BytesIO(b))
            column='rate_SO2' if problem in {'negative','marker'} else 'rate_SO4' if problem=='unemitted' else 'downwash' if problem=='downwash' else 'x'
            f.loc[0,column]=-1 if problem=='negative' else -9999 if problem=='marker' else 1 if problem in {'downwash','unemitted'} else 100
            return f.to_csv(index=False).encode()
        data=change(data,'points.csv',edit)
    else:
        def edit(b):
            if problem=='variable':return re.sub(rb'(NSPT1\s*=\s*)0',rb'\g<1>1',b)
            return b.replace(b'SRCNAM = STK1',b'UNKNOWN = STK1')
        data=change(data,'supplied.INP',edit)
    with pytest.raises(SchemaError,match=match):convert('calpuff',data,cfg)
