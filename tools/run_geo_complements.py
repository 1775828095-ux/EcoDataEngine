"""Isolated official Whittaker and AROSICS calls; NPZ-only numeric boundary."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import json,sys
from pathlib import Path


def run():
    cfg=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'));p=cfg['parameters']
    if cfg['operation']=='whittaker':
        # The upstream wheel uses the NumPy1 ABI. Isolate it from both platform
        # NumPy2 and the independent GDAL/AROSICS environment.
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'external/geo-runtime/whittaker-packages'))
    import numpy as np
    with np.load(sys.argv[1]) as data:
        values=data['values'].copy();report={}
        if cfg['operation']=='whittaker':
            from vam.whittaker import ws2d
            shape=values.shape;matrix=values.reshape(shape[0],-1);result=np.empty_like(matrix,dtype=float)
            for j in range(matrix.shape[1]):
                y=matrix[:,j];w=np.isfinite(y).astype(float)
                result[:,j]=np.asarray(ws2d(np.where(w,y,0.).astype(float),p['smoothing_lambda'],w))
            result=result.reshape(shape);report=dict(kernel='vam.whittaker.ws2d',temporal_penalty_order=2,external_labels_used=False)
        elif cfg['operation']=='arosics':
            from arosics import COREG
            from geoarray import GeoArray
            from osgeo import osr
            from scipy.ndimage import shift
            lat=data['lat'];lon=data['lon'];reference=data['reference'];shape=values.shape
            flip_y=lat[1]>lat[0];flip_x=lon[1]<lon[0]
            dx=abs(float(lon[1]-lon[0]));dy=abs(float(lat[1]-lat[0]))
            gt=(float(lon.min())-dx/2,dx,0.,float(lat.max())+dy/2,0.,-dy)
            srs=osr.SpatialReference();srs.SetFromUserInput(p['crs']);prj=srs.ExportToWkt()
            result=[];events=[]
            for frame,ref in zip(values.reshape(-1,*shape[-2:]),reference.reshape(-1,*shape[-2:])):
                a=frame.copy();b=ref.copy()
                if flip_y:a=a[::-1];b=b[::-1]
                if flip_x:a=a[:,::-1];b=b[:,::-1]
                size=min(256,*a.shape)
                coreg=COREG(GeoArray(b,gt,prj),GeoArray(a,gt,prj),ws=(size,size),max_shift=p['max_shift_pixels'],
                    CPUs=1,progress=False,q=True,ignore_errors=False)
                coreg.calculate_spatial_shifts()
                if not coreg.success:raise ValueError('AROSICS could not establish a valid shift')
                # AROSICS pixel shifts are georeferencing corrections. Re-sample
                # that corrected image onto the unchanged reference pixel grid.
                sy=float(coreg.y_shift_map)/gt[5];sx=float(coreg.x_shift_map)/gt[1]
                corrected=shift(a,(sy,sx),order=1,mode='constant',cval=np.nan,prefilter=False)
                if flip_x:corrected=corrected[:,::-1]
                if flip_y:corrected=corrected[::-1]
                result.append(corrected)
                events.append(dict(x_pixels=-sx if flip_x else sx,y_pixels=-sy if flip_y else sy,
                    pixel_orientation='original input array',reliability=None if coreg.shift_reliability is None else float(coreg.shift_reliability)))
            result=np.stack(result).reshape(shape)
            report=dict(kernel='arosics.COREG.calculate_spatial_shifts',warp='scipy.ndimage.shift order1; missing edges retained',frames=events)
        else:raise ValueError('Unknown isolated operation')
    np.savez(sys.argv[3],values=result)
    Path(sys.argv[4]).write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':run()
