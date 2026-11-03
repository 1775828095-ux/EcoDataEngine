"""Bounded MM4 formatted reader following native CALMET RDHD4/RDMM4.

No uploaded code is executed. Format constants, temperature/wind encoding and
NWS-84 source datum come from the existing official CALMET 5.8.5 source.
The output remains a model estimate, never a station observation.
"""
from datetime import datetime,timedelta
import numpy as np
import xarray as xr

ORIGIN='CALMET 5.8.5 calmet.for RDHD4 (formats 20,30,38,48), RDMM4 (formats 60,70), BLOCK DATA datum3d=NWS-84'

def decode_mm4(raw):
    lines=raw.decode('ascii').rstrip('\x1a\r\n ').splitlines()
    if len(lines)<4:raise ValueError('MM4 header incomplete')
    line=lines[1]
    y,mo,day,hour=[int(line[i:i+2]) for i in range(0,8,2)]
    # Match native YR4's supported twentieth/twenty-first century encoding.
    year=1900+y if y>=70 else 2000+y
    count,fullx,fully,nz=[int(line[i:i+4]) for i in range(8,24,4)]
    top=float(line[24:30])
    line=lines[2];ix,iy,nx,ny=[int(line[i:i+4]) for i in range(0,16,4)]
    if not (1<=nz<=200 and 1<=nx<=1000 and 1<=ny<=1000 and 1<=count<=100000):raise ValueError('MM4 dimensions invalid')
    if nx*ny*count*(nz+9)>100000000:raise ValueError('MM4 record count exceeds bounded reader limit')
    sigma=np.array([float(lines[3+i]) for i in range(nz)])
    if not np.isfinite(sigma).all() or (sigma<=0).any() or (sigma>1).any():raise ValueError('MM4 sigma levels invalid')
    pos=3+nz;geometry=[]
    for j in range(ny):
        for i in range(nx):
            line=lines[pos];pos+=1
            qi,qj=int(line[:3]),int(line[3:6])
            if (qi,qj)!=(ix+i,iy+j):raise ValueError('MM4 geometry index order invalid')
            lat,lon,elev,land=float(line[6:13]),float(line[13:21]),float(line[21:26]),int(line[26:29])
            if not (-90<=lat<=90 and -180<=lon<=180):raise ValueError('MM4 latitude/longitude invalid')
            geometry.append((lat,lon,elev,land))
    npoints=nx*ny;nlevels=nz+8
    expected=pos+count*npoints*(nlevels+1)
    if len(lines)!=expected:raise ValueError(f'MM4 actual records do not match header: {len(lines)} / {expected}')
    profiles=np.empty((count,npoints,nlevels,6));surface=np.empty((count,npoints,3));times=[]
    start=datetime(year,mo,day,hour)
    for k in range(count):
        wanted=start+timedelta(hours=k);times.append(np.datetime64(wanted,'ns'))
        for j in range(ny):
            for i in range(nx):
                q=j*nx+i;line=lines[pos];pos+=1
                yy,mm,dd,hh=[int(line[s:s+2]) for s in range(0,8,2)]
                yy=1900+yy if yy>=70 else 2000+yy
                if datetime(yy,mm,dd,hh)!=wanted or (int(line[8:11]),int(line[11:14]))!=(ix+i,iy+j):raise ValueError('MM4 profile time/index mismatch')
                surface[k,q]=[float(line[14:21]),float(line[21:26]),int(line[26:28])]
                for lev in range(nlevels):
                    line=lines[pos];pos+=1
                    p,z,it,dew,wd,ws=(float(line[:5])/10,float(line[5:11]),int(line[11:15]),int(line[15:17]),float(line[17:21]),float(line[21:23]))
                    # CALMET original NWS sign encoding: odd temperature = negative.
                    temperature=(-1 if it%2 else 1)*it/10+273.15
                    profiles[k,q,lev]=[p,z,temperature,dew,wd,ws*.51444]
    if not np.isfinite(profiles).all() or not np.isfinite(surface).all():raise ValueError('MM4 contains nonfinite native numbers')
    g=np.array(geometry);z=profiles[:,:,:,1]-g[None,:,None,2]
    return dict(profiles=profiles,height_agl=z,geometry=g,times=np.array(times),surface=surface,
        source_datum='NWS-84',source_geographic_crs='+proj=longlat +R=6370000 +no_defs',
        reader_origin=ORIGIN,shape=dict(hours=count,points=npoints,levels=nlevels),
        native_shape=dict(nx=nx,ny=ny,nz=nz,full_x=fullx,full_y=fully,top_pressure_hPa=top),
        surface_wind_height='native MM4 surface level is represented at terrain elevation; explicit vertical interpolation is required for station anemometer heights')

def wind_at_height(decoded,height):
    """Earth-relative vector wind, bounded linear vertical interpolation in AGL.

    Geometric/vertical alignment only, not algorithm ranking or fusion.
    Native surface + mandatory + sigma levels are sorted/deduplicated by height.
    Underground mandatory levels and their calm placeholders are excluded.
    """
    if not np.isfinite(height) or height<0:raise ValueError('Missing/invalid target wind height')
    profile=decoded['profiles'];z=decoded['height_agl']
    wd=profile[:,:,:,4]*np.pi/180;ws=profile[:,:,:,5]
    u=-ws*np.sin(wd);v=-ws*np.cos(wd)
    result=np.empty((*z.shape[:2],2))
    for t in range(z.shape[0]):
        for p in range(z.shape[1]):
            keep=z[t,p]>=0
            zz=z[t,p,keep];uu=u[t,p,keep];vv=v[t,p,keep]
            order=np.argsort(zz,kind='stable');zz=zz[order];uu=uu[order];vv=vv[order]
            unique,index=np.unique(zz,return_index=True);uu=uu[index];vv=vv[index]
            if len(unique)<2 or not unique[0]<=height<=unique[-1]:raise ValueError('MM4 profile cannot bracket requested anemometer height; no extrapolation')
            result[t,p]=[np.interp(height,unique,uu),np.interp(height,unique,vv)]
    return result
