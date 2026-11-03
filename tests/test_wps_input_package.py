import io,json,struct,tempfile,zipfile
from pathlib import Path
import numpy as np
import pytest
from scipy.io import FortranFile
from ecodata.round1 import convert
from ecodata.core.exceptions import SchemaError
BASE=Path(__file__).parents[1]/'examples/ai_round1/wrf/input_package'
def case(label='latlon'):
    return (BASE/f'A_synthetic_{label}.zip').read_bytes(),json.loads((BASE/f'config_{label}.json').read_text())
def transform(data,func):
    raw=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as z,zipfile.ZipFile(raw,'w') as out:
        for name in z.namelist():out.writestr(name,func(z.read(name)))
    return raw.getvalue()

@pytest.mark.parametrize('label,shape,iproj',[('latlon',(2,3),0),('lambert',(3,2),3)])
def test_independent_scipy_native_parse_and_values(label,shape,iproj):
    data,cfg=case(label);files,report=convert('wrf',data,cfg)
    for it,h in enumerate(('00','06')):
        raw=files[f'COMBINED_2024-02-29_{h}.wps'];found={}
        with tempfile.TemporaryFile() as fp:
            fp.write(raw);fp.seek(0);f=FortranFile(fp,header_dtype='>u4')
            while fp.tell()<len(raw):
                assert f.read_record('>i4').tolist()==[5]
                header=f.read_record('u1').tobytes();level,nx,ny,p=struct.unpack('>fiii',header[140:])
                field=header[60:69].decode().strip();assert header[69:94].strip()==b'K'
                assert header[:24].strip()==f'2024-02-29_{h}:00:00'.encode() and (ny,nx)==shape and p==iproj
                assert f.read_record('u1').tobytes()[:8]==b'SWCORNER'
                assert f.read_record('>i4').tolist()==[0]
                found[field,level]=f.read_record('>f4').reshape(ny,nx)
        for il,key in enumerate([('TT',85000),('TT',50000),('SST',200100)]):
            np.testing.assert_array_equal(found[key],np.arange(6).reshape(shape)+280+it*10+il)
    assert report['interpolation']=='none' and report['native_model_read_verified'] is False
    assert not any(n.startswith(('met_em','geo_em','wrfinput')) for n in files)

@pytest.mark.parametrize('error,match',[('time','缺时刻'),('levels','缺字段/层'),('unit','单位'),('grid','网格/投影'),('nan','NaN'),('missing','缺测标记'),('duplicate','重复字段'),('tail','尾标记'),('wind','earth-relative'),('interval','缺口')])
def test_native_errors_never_silently_modified(error,match):
    data,cfg=case()
    if error=='time':cfg['expected_times_utc'].append('2024-02-29T12:00:00Z')
    elif error=='levels':cfg['required_levels']['TT'].append(70000)
    elif error=='interval':cfg['interval_seconds']=3600
    else:
        counter=[0]
        def mutate(raw):
            counter[0]+=1
            if counter[0]>1:return raw
            b=bytearray(raw)
            # Five records: version12, header164, projection36, wind12, slab32.
            if error=='unit':b[12+4+69:12+4+94]=b'degC'.ljust(25,b' ')
            elif error=='grid':struct.pack_into('>f',b,12+164+4+12,106)
            elif error in {'nan','missing'}:struct.pack_into('>f',b,12+164+36+12+4,float('nan') if error=='nan' else -9999)
            elif error=='duplicate':return raw+raw
            elif error=='tail':b[-1]^=1
            elif error=='wind':struct.pack_into('>i',b,12+164+36+4,1)
            return bytes(b)
        data=transform(data,mutate)
    with pytest.raises(SchemaError,match=match):convert('wrf',data,cfg)

@pytest.mark.parametrize('key,value,match',[('required_levels',{},'required_levels'),('missing_values',None,'missing_values'),('time_basis','local','UTC'),('prefix','../../wrong','prefix')])
def test_missing_configuration_rejected(key,value,match):
    data,cfg=case();cfg[key]=value
    with pytest.raises(SchemaError,match=match):convert('wrf',data,cfg)
