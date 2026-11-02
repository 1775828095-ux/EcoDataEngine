import io,json,struct,zipfile
import numpy as np
import pytest
from ecodata.interface.calpuff7_results import read_discrete
from ecodata.interface.calpuff_project import unpack
from ecodata.core.exceptions import SchemaError

def control():
    v=dict(LSAMP='F',MSOURCE=1,NRGRP=0,NREC=2,NPT1=1,NPT2=0,NSPT1=0,NAR1=0,NAR2=0,NVL1=0,NVL2=0,NLN2=0,NRD1=0,NRD2=0,NFL2=0,ABTZ='UTC-0800',IBYR=2021,IBMO=3,IBDY=13,IBHR=4,IEYR=2021,IEMO=3,IEDY=13,IEHR=5,CSPEC='NOx',NSPEC=1,LCOMPRS='T',SRCNAM='STACK1',LCFILES='F',METDAT='CALMET.DAT',AUXEXT='aux',MCHEM=0,NPTDAT=0,NARDAT=0,NVOLDAT=0,NFLDAT=0,NRDDAT=0,NLNDAT=0,PUFLST='RUN.LST',CONDAT='RUN.CON',NX=9,NY=9,NZ=10)
    return 'CALPUFF.INP     7.0\n'+'\n'.join(f'! {k} = {x} !' for k,x in v.items())

def binary(override=None):
    label=b'NOX           1';coords=np.array([[1,2],[3,4],[5,6],[0,0],[0,0]],dtype='<f4').tobytes()
    rs=[b'CONC.DAT        7.0',struct.pack('<i',0),b'x'*300,b'a',b'b',label,b'g/m3',coords,b'DISCRETE',b'STACK1']
    for name in ('STACK1','TOTAL'):
        rs.extend([struct.pack('<8i',2021,72,4,0,2021,72,5,0),struct.pack('<2i',1,1)+name.encode().ljust(16)+b' '*8,struct.pack('<i',2),label+np.array([-1.0001,2e-9],dtype='<f4').tobytes()])
    if override:override(rs)
    return b''.join(struct.pack('<i',len(r))+r+struct.pack('<i',len(r)) for r in rs)

def test_native_zero_compression_and_source_sum():
    r=read_discrete(binary(),control())
    assert r['times']==['2021-03-13T05:00:00-08:00']
    np.testing.assert_allclose(r['values_g_m3']['NOX'],[[0,2e-9]],rtol=1e-7)

@pytest.mark.parametrize('problem',['time','source','sum','compression','nonfinite','truncated'])
def test_inconsistent_native_records_rejected(problem):
    def edit(rs):
        if problem=='time':rs[10]=struct.pack('<8i',2021,72,5,0,2021,72,6,0)
        elif problem=='source':rs[11]=rs[11].replace(b'STACK1',b'OTHER1')
        elif problem in {'sum','compression','nonfinite'}:
            vals=[-1.0001,4e-9] if problem=='sum' else [-3.0001,2e-9] if problem=='compression' else [-1.0001,float('nan')]
            rs[-1]=rs[-1][:15]+np.array(vals,dtype='<f4').tobytes()
        else:rs.pop()
    with pytest.raises((SchemaError,ValueError)):read_discrete(binary(edit),control())

def package(changes=None):
    data={'CALPUFF.INP':control().encode(),'CALMET.DAT':b'met','CALMET.DAT.AUX':b'aux','provenance.json':json.dumps(dict(sources=['source declaration'],defaults=['explicit reference choice'],limitations=['not accuracy tested'])).encode()}
    if changes:changes(data)
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:
        for n,b in data.items():z.writestr(n,b)
    return out.getvalue()

def test_bundle_contract_and_provenance():
    _,p=unpack(package(),'existing');assert p['defaults']
    with pytest.raises(ValueError,match='不安全'):unpack(package(lambda d:d.update({'../extra':b'x'})),'existing')
    with pytest.raises(ValueError,match='外部路径'):unpack(package(lambda d:d.update({'CALPUFF.INP':d['CALPUFF.INP'].replace(b'RUN.LST',b'../RUN.LST')})),'existing')
    with pytest.raises(ValueError,match='provenance'):unpack(package(lambda d:d.update({'provenance.json':b'{}'})),'existing')
    with pytest.raises(ValueError,match='常量点源'):unpack(package(lambda d:d.update({'CALPUFF.INP':d['CALPUFF.INP'].replace(b'NFL2 = 0',b'NFL2 = 1')})),'existing')

def test_page_routes_render_and_missing_inputs_stay_disabled():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_string('from ecodata.interface.calpuff_project import render\nrender()').run()
    assert not app.exception and app.button(key='calpuff_project_start').disabled
    app.radio[0].set_value('从WRF与地理参数资料新建工程').run()
    assert not app.exception and app.button(key='calpuff_project_start').disabled
    assert any('原始DEM' in w.value for w in app.warning)
