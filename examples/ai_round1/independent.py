"""Test-only readers. Do not import EcoData exporters here.

SURF reader translates official CALMET RDHD/RDS layout; SWAT reader uses
handbook fixed positions; WPS reader uses scipy FortranFile and NCAR layout.
These check format/values, not execution of full atmospheric/hydrologic models.
"""
from datetime import datetime,timedelta
from pathlib import Path
import io
import re
import numpy as np
from scipy.io import FortranFile


def surf(path):
    lines=Path(path).read_text(encoding='ascii').splitlines()
    assert lines[0][:16].strip()=='SURF.DAT' and lines[0][16:32].strip()=='2.0'
    ncomments=int(lines[1]);assert lines[2+ncomments].strip()=='NONE'
    tokens=' '.join(lines[3+ncomments:]).split()
    h=list(map(int,tokens[:8]));start=datetime(h[0],1,1)+timedelta(days=h[1]-1,hours=h[2]);end=datetime(h[3],1,1)+timedelta(days=h[4]-1,hours=h[5])
    n=h[7];stations=list(map(int,tokens[8:8+n]));assert len(set(stations))==n
    tokens=tokens[8+n:];records=[];times=[]
    while tokens:
        assert len(tokens)>=3+8*n
        y,d,hour=map(int,tokens[:3]);t=datetime(y,1,1)+timedelta(days=d-1,hours=hour)
        assert 0<=hour<24 and t.year==y
        times.append(t);records.append(np.array(tokens[3:3+8*n],dtype=float).reshape(n,8));tokens=tokens[3+8*n:]
    assert times[0]==start and times[-1]==end
    assert all(b-a==timedelta(hours=1) for a,b in zip(times,times[1:]))
    return {'stations':stations,'timezone_west_hours':h[6],'times':times,'values':np.array(records)}


def swat(path,kind):
    lines=Path(path).read_text(encoding='ascii').splitlines();width=5 if kind=='pcp' else 10
    station=[float(line[7:7+width]) for line in lines[1:4]]
    rows=[];dates=[]
    for line in lines[4:]:
        assert len(line)==(12 if kind=='pcp' else 17)
        y,d=int(line[:4]),int(line[4:7]);date=datetime(y,1,1)+timedelta(days=d-1)
        assert date.year==y
        dates.append(date);rows.append([float(line[7:12])] if kind=='pcp' else [float(line[7:12]),float(line[12:17])])
    assert all(b-a==timedelta(days=1) for a,b in zip(dates,dates[1:]))
    return {'station':station,'dates':dates,'values':np.array(rows)}


def wps(path):
    with FortranFile(path,'r',header_dtype='>u4') as f:
        assert f.read_ints('>i4').tolist()==[5]
        meta=f.read_record('u1').tobytes();assert len(meta)==156
        # fixed official positions: hdate24,xfcst4,source32,field9,unit25,desc46,xlvl4,nx4,ny4,iproj4
        xfcst=float(np.frombuffer(meta[24:28],dtype='>f4')[0]);level=float(np.frombuffer(meta[140:144],dtype='>f4')[0])
        nx,ny,proj=np.frombuffer(meta[144:156],dtype='>i4');assert proj==0
        projection=f.read_record('u1').tobytes();assert len(projection)==28
        assert projection[:8].decode('ascii')=='SWCORNER'
        grid=np.frombuffer(projection[8:],dtype='>f4').astype(float)
        wind=f.read_ints('>i4').tolist();assert wind==[0]
        slab=f.read_reals('>f4').reshape(int(ny),int(nx))
        # Reject extra records, not merely read the first slab.
        assert f._fp.read(1)==b''
    return {'date':meta[:24].decode().strip(),'field':meta[60:69].decode().strip(),'unit':meta[69:94].decode().strip(),
            'xfcst':xfcst,'level':level,'projection':grid,'values':slab}


def photolysis(path):
    lines=Path(path).read_text(encoding='ascii').splitlines();heights=[];rows=[];reactions=[];current=[];conditions=[];condition=None
    for line in lines[1:]:
        if 'O3 Column' in line:
            condition=[float(re.search(re.escape(key)+r'\s*=\s*([-+0-9.eE]+)',line).group(1)) for key in ('O3 Column','Albedo','Terrain')]
        elif 'km above surface' in line:
            if current:rows.append(current)
            current=[];heights.append(float(line.split()[0]))
            conditions.append(condition)
        elif 'CBX' in line:
            parts=line.split();values=[float(v) for v in parts[:10]]
            assert len(values)==10 and all(v>=0 for v in values)
            current.append(values)
            if len(heights)==1:reactions.append(parts[-1])
    if current:rows.append(current)
    array=np.array(rows);assert array.ndim==3 and array.shape[1]==len(reactions)
    assert len(conditions)==len(rows) and all(c is not None for c in conditions)
    return {'title':lines[0],'condition_line':lines[1],'heights':heights,'conditions':conditions,'reactions':reactions,'values':array}
