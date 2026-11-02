import struct
import numpy as np
import pytest
from ecodata.core.exceptions import SchemaError
from ecodata.interface.wrf_assembly import assemble_pressure_update, read_fields
from test_wps_fields import case, read_records
from ecodata.models.adapters.wps_fields import export_pressure


def field(name, level, values, lat=1, lon=230, dy=-1, radius=6371.2294921875):
    units = {'TT':'K', 'UU':'m s-1', 'VV':'m s-1', 'GHT':'m', 'RH':'%'}[name]
    text=lambda s,n:s.encode().ljust(n,b' ')
    header=(text('2023-03-31_00:00:00',24)+struct.pack('>f',0)+text('test',32)
            +text(name,9)+text(units,25)+text('test',46)+struct.pack('>fiii',level,2,2,0))
    parts=[struct.pack('>i',5),header,b'SWCORNER'+struct.pack('>5f',lat,lon,dy,1,radius),struct.pack('>i',0),np.array(values,dtype='>f4').tobytes()]
    return b''.join(struct.pack('>i',len(p))+p+struct.pack('>i',len(p)) for p in parts)


def inputs():
    base=b''.join(field(f,85000,[250]*4) for f in ['TT','UU','VV','GHT'])+field('RH',85000,[10,20,30,40])
    update=b''.join(field(f,85000,[270]*4,lat=0,lon=-130,dy=1) for f in ['TT','UU','VV','GHT'])
    return base,update


def test_exact_selection_normalizes_longitude_and_reverses_rows():
    base,update=inputs()
    merged,report=assemble_pressure_update(base,update)
    rows=read_fields(merged)
    assert np.frombuffer(rows[0].parts[4],dtype='>f4').tolist()==[30,40,10,20]
    assert b''.join(f.raw for f in rows[1:])==update
    assert report['pressure_fallback'] is False
    assert report['native_model_executed'] is False


def test_reject_radius_difference_and_partial_replacement():
    base,update=inputs()
    with pytest.raises(SchemaError,match='半径'):
        assemble_pressure_update(base,b''.join(field(f,85000,[270]*4,lat=0,dy=1,radius=6367.47) for f in ['TT','UU','VV','GHT']))
    with pytest.raises(SchemaError,match='完整替换'):
        assemble_pressure_update(base,b''.join(f.raw for f in read_fields(update)[1:]))


def test_converter_records_explicit_radius():
    ds,cfg=case();cfg['earth_radius_km']=6371.2294921875
    files,report=export_pressure(ds,cfg)
    assert read_records(next(v for k,v in files.items() if k.endswith('.wps')))[0]['grid'][-1]==cfg['earth_radius_km']
    assert report['projection']['earth_radius_source']=='configuration'
    cfg['earth_radius_km']=float('nan')
    with pytest.raises(SchemaError,match='earth_radius_km'):
        export_pressure(ds,cfg)
