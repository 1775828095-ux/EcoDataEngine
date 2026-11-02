"""Independent WPS-v5 readback: SciPy FortranFile, never import the writer."""
import json
import pathlib
import struct
import sys

import numpy as np
import xarray as xr
from scipy.io import FortranFile


def read(path):
    rows=[]
    with path.open('rb') as stream:
        f=FortranFile(stream,header_dtype=np.dtype('>u4'))
        while stream.tell()<path.stat().st_size:
            version=f.read_record('>i4')
            assert version.tolist()==[5]
            h=f.read_record('u1').tobytes()
            assert len(h)==156
            date=h[:24].decode().strip();field=h[60:69].decode().strip();units=h[69:94].decode().strip()
            level,nx,ny,iproj=struct.unpack('>fiii',h[140:]);assert iproj==0
            g=f.read_record('u1').tobytes();assert g[:8]==b'SWCORNER'
            grid=struct.unpack('>5f',g[8:]);wind=f.read_record('>i4')[0];assert wind in (0,1)
            values=f.read_record('>f4').reshape(ny,nx)
            rows.append({'date':date,'field':field,'unit':units,'level':level,'grid':grid,'wind_grid_relative':bool(wind),'values':values})
    return rows


def verify(source,outdir):
    # Source has NOAA units degK,m/s,%,m and millibar; independent explicit reference conversions.
    ds=xr.open_dataset(source.open('rb'),engine='scipy').load()
    names={'TT':'air','UU':'uwnd','VV':'vwnd','RH':'rhum','GHT':'hgt'}
    units={'TT':'K','UU':'m s-1','VV':'m s-1','RH':'%','GHT':'m'}
    expected=ds.sortby('lat').sortby('lon')
    observed=[]
    for path in sorted(outdir.glob('ATM_*.wps')):observed.extend(read(path))
    assert len(observed)==ds.sizes['time']*ds.sizes['level']*len(names)
    seen=set();errors={key:0. for key in names};count=0
    for row in observed:
        label=row['field'];assert label in names and row['unit']==units[label]
        stamp=np.datetime64(row['date'].replace('_','T'))
        pressure=row['level']/100
        assert (stamp,label,pressure) not in seen;seen.add((stamp,label,pressure))
        a=expected[names[label]].sel(time=stamp,level=pressure).values
        assert row['values'].shape==a.shape and np.isfinite(row['values']).all()
        error=float(np.max(np.abs(a.astype('float64')-row['values'].astype('float64'))))
        assert error<=3e-5 # predeclared float32 conversion allowance, all source values already float32.
        errors[label]=max(errors[label],error);count+=a.size
        np.testing.assert_allclose(row['grid'][:4],[35,105,2.5,2.5],rtol=0,atol=1e-6)
        assert not row['wind_grid_relative']
    assert count==720
    result={'independent_reader':'SciPy FortranFile','records':len(observed),'compared_values':count,
            'max_absolute_error_by_field':errors,'header_units_time_pressure_grid':'passed','missing_nonfinite':0,
            'B2_comparison':'not performed; no same-case independent WPS reference file',
            'official_metgrid_read':'not executed','complete_WRF_run':'not executed'}
    (outdir/'independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':verify(pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]))
