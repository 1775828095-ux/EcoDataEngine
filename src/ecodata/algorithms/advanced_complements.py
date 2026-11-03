"""Independent complementary operators, composed by the existing public DAG.

Numerical backends: VAM Whittaker, sinkr DINEOF, OpenCV, AROSICS and starfm4py.
Unknown performance remains unknown; compatibility checks are not rankings.
"""
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from tempfile import TemporaryDirectory
import json,subprocess
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS
from ecodata.algorithms.contracts import AlgorithmSpec,Compatibility
from ecodata.algorithms.raster_complements import dependency,raster,stamp
from ecodata.algorithms.open_source_spatial import finish
from ecodata.core.schemas import EcoDataBundle

ROOT=Path(__file__).resolve().parents[3]
GEO=ROOT/'external/geo-runtime/env/python.exe'
RSCRIPT=ROOT/'external/R-4.5.2/bin/Rscript.exe'


def runtime_ready(name):
    path=ROOT/'external/geo-runtime/verified.json'
    return GEO.exists() and path.exists() and name in json.loads(path.read_text(encoding='utf-8'))['packages']


def timed(context):
    if len(context.sources)!=1 or context.semantic_type!='continuous':raise ValueError('One continuous timed raster required')
    ds=context.sources[0].dataset;a=ds[context.variable]
    if a.dims!=('time','lat','lon') or min(a.shape)<2:raise ValueError('time,lat,lon raster required')
    dates=pd.DatetimeIndex(a.time.values)
    if dates.has_duplicates or not dates.is_monotonic_increasing:raise ValueError('Increasing unique timestamps required')
    if not a.attrs.get('units') or not a.attrs.get('physical_variable') or not ds.attrs.get('crs'):raise ValueError('Explicit physical metadata required')
    if np.isinf(a.values).any():raise ValueError('Infinity is not a missing marker')
    return ds,a


def output(context,spec,ds,values,parameters,report,started):
    out=ds[[context.variable]].copy(deep=True);out[context.variable].values=values
    stamp(out,context.variable,[s.dataset for s in context.sources],spec.id,parameters)
    result=finish(context,spec,out,dict(reference_used=False,accuracy_gain='UNKNOWN',parameters=parameters,**report),started)
    history=tuple(h for s in context.sources for h in s.history)+result.data.history[-1:]
    return replace(result,data=EcoDataBundle(result.data.dataset,history))


def geo_call(operation,arrays,parameters):
    with TemporaryDirectory(prefix='ecodata_geo_') as folder:
        folder=Path(folder);np.savez(folder/'input.npz',**arrays)
        (folder/'config.json').write_text(json.dumps(dict(operation=operation,parameters=parameters)),encoding='utf-8')
        job=subprocess.run([str(GEO),str(ROOT/'tools/run_geo_complements.py'),str(folder/'input.npz'),
            str(folder/'config.json'),str(folder/'output.npz'),str(folder/'report.json')],capture_output=True,text=True,encoding='utf-8',timeout=600)
        if job.returncode:raise RuntimeError('Official geospatial backend failed: '+job.stderr[-2500:])
        with np.load(folder/'output.npz') as ds:result=ds['values'].copy()
        return result,json.loads((folder/'report.json').read_text(encoding='utf-8'))


class WhittakerRaster:
    def __init__(self):
        self.spec=AlgorithmSpec('temporal.whittaker','Weighted Whittaker temporal reconstruction','vam.whittaker 2.0.6',
            'temporal_reconstruction','available' if runtime_ready('vam.whittaker') else 'dependency_missing',('timed_raster',),('continuous',),
            license='MIT',source_url='https://pypi.org/project/vam.whittaker/2.0.6/',backend='Official vam.whittaker ws2d; isolated Python 3.11 and NumPy1 ABI',
            capabilities=('temporal_missing_recovery','temporal_noise_suppression'),input_contract=dict(source_arity=1,requires_parameters=['smoothing_lambda']),
            output_contract=dict(accuracy_gain='UNKNOWN',range_preservation='UNKNOWN'),notes='Regular time steps; >=2 visible values per pixel. May smooth real extremes.')
    def compatibility(self,context):
        try:
            if self.spec.status!='available':raise ValueError('Verified Whittaker runtime required')
            _,a=timed(context)
            if len(a.time)<4 or not np.all(np.diff(a.time.values)==np.diff(a.time.values)[0]):raise ValueError('At least four regularly spaced dates required; align time first')
            if (np.isfinite(a.values).sum(0)<2).any():raise ValueError('At least two visible values in each spatial trace required')
            value=context.parameters.get('smoothing_lambda')
            if not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:raise ValueError('Positive lambda required')
            return Compatibility(True,())
        except (ValueError,KeyError,TypeError) as exc:return Compatibility(False,(str(exc),))
    def parameter_proposals(self,context,rng):
        try:timed(context)
        except (ValueError,KeyError):return []
        p=[dict(smoothing_lambda=float(v)) for v in [1.,10**rng.uniform(-2,4)]]
        return [x for x in p if self.compatibility(replace(context,parameters=dict(context.parameters,**x))).eligible]
    def run(self,context):
        if not self.compatibility(context).eligible:raise ValueError(str(self.compatibility(context).reasons))
        started=perf_counter();ds,a=timed(context);p=dict(smoothing_lambda=context.parameters['smoothing_lambda'])
        result,report=geo_call('whittaker',dict(values=a.values),p)
        return output(context,self.spec,ds,result,p,report,started)


class DineofRaster:
    def __init__(self):
        self.spec=AlgorithmSpec('reconstruction.dineof','DINEOF joint space-time missing reconstruction','sinkr pinned dd3f9e1',
            'spatiotemporal_reconstruction','available' if RSCRIPT.exists() and (ROOT/'external/sinkr/R/dineof.R').exists() else 'dependency_missing',
            ('timed_raster',),('continuous',),compute='medium',license='GPL-3.0',source_url='https://github.com/marchtaylor/sinkr',
            backend='Unmodified sinkr::dineof, base R svd, isolated process; not the Fortran distribution',
            capabilities=('spatiotemporal_missing_recovery','low_rank_reconstruction'),input_contract=dict(source_arity=1,requires_parameters=['max_modes','internal_seed']),
            output_contract=dict(observed_values_preserved=True,accuracy_gain='UNKNOWN'),
            notes='No entirely missing time or pixel traces. Internal holdout uses supplied visible values only; not external validation labels.')
    def compatibility(self,context):
        try:
            if self.spec.status!='available':raise ValueError('R runtime and pinned sinkr source required')
            _,a=timed(context);v=a.values.reshape(len(a.time),-1);valid=np.isfinite(v)
            if len(a.time)<4 or (valid.sum(0)<3).any() or (valid.sum(1)<3).any():raise ValueError('At least four dates and three observations per time/pixel trace required; entirely hidden dates unsupported')
            if valid.all() or np.nanstd(v)==0:raise ValueError('Nonconstant incomplete raster required')
            k=context.parameters.get('max_modes');seed=context.parameters.get('internal_seed')
            if type(k)!=int or not 1<=k<min(v.shape) or type(seed)!=int or seed<0:raise ValueError('Valid mode bound and nonnegative internal seed required')
            return Compatibility(True,())
        except (ValueError,KeyError,TypeError) as exc:return Compatibility(False,(str(exc),))
    def parameter_proposals(self,context,rng):
        try:_,a=timed(context)
        except (ValueError,KeyError):return []
        bound=min(len(a.time),a.sizes['lat']*a.sizes['lon'])-1
        p=dict(max_modes=bound,internal_seed=0)
        return [p] if self.compatibility(replace(context,parameters=dict(context.parameters,**p))).eligible else []
    def run(self,context):
        if not self.compatibility(context).eligible:raise ValueError(str(self.compatibility(context).reasons))
        started=perf_counter();ds,a=timed(context);p={k:context.parameters[k] for k in ['max_modes','internal_seed']}
        values=np.asarray(a.values,float);matrix=values.reshape(len(a.time),-1).T
        means=np.nanmean(matrix,axis=1,keepdims=True);centered=matrix-means;valid=np.isfinite(centered)
        rows=valid.sum(1).copy();cols=valid.sum(0).copy();refs=[]
        for index in np.random.default_rng(p['internal_seed']).permutation(np.flatnonzero(valid.ravel(order='F'))):
            row,col=np.unravel_index(index,valid.shape,order='F')
            if rows[row]<=2 or cols[col]<=2:continue
            refs.append(int(index)+1);rows[row]-=1;cols[col]-=1
            if len(refs)>=max(1,min(30,int(valid.sum()*.01))):break
        if not refs:raise ValueError('No safe internal holdout supports mode selection')
        with TemporaryDirectory(prefix='ecodata_dineof_') as folder:
            folder=Path(folder);np.savetxt(folder/'input.csv',centered,delimiter=',',fmt='%.17g')
            text=(folder/'input.csv').read_text().replace('nan','NA');(folder/'input.csv').write_text(text)
            np.savetxt(folder/'refs.txt',refs,fmt='%d')
            job=subprocess.run([str(RSCRIPT),'--vanilla',str(ROOT/'tools/run_dineof.R'),str(ROOT/'external/sinkr/R/dineof.R'),
                str(folder/'input.csv'),str(folder/'refs.txt'),str(folder/'output.csv'),str(p['max_modes']),'0.0001',str(p['internal_seed'])],
                capture_output=True,text=True,encoding='utf-8',timeout=600)
            if job.returncode:raise RuntimeError('DINEOF backend failed: '+job.stderr[-2500:])
            filled=np.loadtxt(folder/'output.csv',delimiter=',')+means;filled[valid]=matrix[valid]
        result=filled.T.reshape(values.shape)
        if not np.isfinite(result).all():raise ValueError('DINEOF produced nonfinite reconstruction')
        return output(context,self.spec,ds,result,p,dict(selected_modes=int(job.stdout.strip().splitlines()[-1]),
            internal_holdout_count=len(refs),external_labels_used=False,centering='Visible per-pixel temporal mean',observed_values_preserved=True),started)


class GuidedRaster:
    def __init__(self):
        v=dependency('opencv-contrib-python-headless')
        self.spec=AlgorithmSpec('fusion.guided_filter','Guided cross-source structure filter',v or 'missing','multi_source_fusion',
            'available' if v else 'dependency_missing',('aligned_rasters',),('continuous',),license='Apache-2.0',
            source_url='https://github.com/opencv/opencv_contrib',backend='OpenCV ximgproc.guidedFilter, float32 normalized intensity',
            capabilities=('multi_source_fusion','edge_guided_reconstruction'),input_contract=dict(source_arity=2,requires_parameters=['guide_index','radius_pixels','relative_eps']),
            output_contract=dict(accuracy_gain='UNKNOWN',range_preservation='UNKNOWN'),notes='Guide and target must be same-variable aligned fields. Unrelated texture may create artifacts.')
    def compatibility(self,context):
        try:
            if self.spec.status!='available':raise ValueError('OpenCV contrib required')
            fields=raster(context,2);p=context.parameters
            if p.get('guide_index') not in (0,1):raise ValueError('Explicit guide branch required')
            r=p.get('radius_pixels');eps=p.get('relative_eps')
            if type(r)!=int or not 1<=r<min(fields[0].shape[-2:]) or not isinstance(eps,(int,float)) or not np.isfinite(eps) or eps<=0:raise ValueError('Valid pixel radius and relative epsilon required')
            if all(np.std(f.values)==0 for f in fields):raise ValueError('No spatial guide structure')
            return Compatibility(True,())
        except (ValueError,KeyError,TypeError) as exc:return Compatibility(False,(str(exc),))
    def parameter_proposals(self,context,rng):
        try:fields=raster(context,2)
        except (ValueError,KeyError,TypeError):return []
        return [dict(guide_index=i,radius_pixels=r,relative_eps=float(e)) for i in [0,1]
            for r in sorted({1,max(1,int(np.sqrt(min(fields[0].shape[-2:]))))}) for e in [1e-3,10**rng.uniform(-5,-1)]]
    def run(self,context):
        if not self.compatibility(context).eligible:raise ValueError(str(self.compatibility(context).reasons))
        import cv2
        started=perf_counter();p={k:context.parameters[k] for k in ['guide_index','radius_pixels','relative_eps']};i=p['guide_index']
        guide=context.sources[i].dataset[context.variable];ds=context.sources[1-i].dataset;target=ds[context.variable]
        result=[]
        for g,t in zip(guide.values.reshape(-1,*guide.shape[-2:]),target.values.reshape(-1,*target.shape[-2:])):
            go=float(g.min());gs=float(np.ptp(g)) or 1.;to=float(t.min());ts=float(np.ptp(t)) or 1.
            value=cv2.ximgproc.guidedFilter(np.asarray((g-go)/gs,np.float32),np.asarray((t-to)/ts,np.float32),p['radius_pixels'],p['relative_eps'])
            result.append(value.astype(float)*ts+to)
        return output(context,self.spec,ds,np.stack(result).reshape(target.shape),p,dict(temporal_mixing=False,guide_source=context.sources[i].dataset.attrs.get('source_id')),started)


class ArosicsRaster:
    def __init__(self):
        self.spec=AlgorithmSpec('spatial.arosics_coreg','AROSICS image co-registration','isolated pinned runtime','geometric_alignment',
            'available' if runtime_ready('arosics') else 'dependency_missing',('aligned_imagery',),('continuous',),
            compute='medium',license='Apache-2.0',source_url='https://github.com/GFZ/arosics',
            backend='Official AROSICS COREG shift estimation + SciPy subpixel warp onto original grid',
            capabilities=('image_coregistration','geometric_alignment'),input_contract=dict(source_arity=2,requires_parameters=['reference_index','max_shift_pixels']),
            output_contract=dict(boundary_missing_possible=True,accuracy_gain='UNKNOWN'),
            notes='Explicit allow_image_coregistration required. Imagery only; rejects forbidden spatial shifts. Edge nodata requires a later recovery node.')
    def compatibility(self,context):
        try:
            if self.spec.status!='available':raise ValueError('Verified GDAL/AROSICS environment required')
            fields=raster(context,2);p=context.parameters
            if p.get('allow_image_coregistration') is not True or context.model_constraints.get('forbid_spatial_shift'):raise ValueError('Image shift not explicitly permitted')
            if min(fields[0].shape[-2:])<64:raise ValueError('At least64x64 image support required')
            if p.get('reference_index') not in (0,1) or type(p.get('max_shift_pixels'))!=int or p['max_shift_pixels']<1:raise ValueError('Explicit reference and positive shift limit required')
            if any(np.std(f.values)<1e-12 for f in fields):raise ValueError('Textured reference and target required')
            return Compatibility(True,())
        except (ValueError,KeyError,TypeError) as exc:return Compatibility(False,(str(exc),))
    def parameter_proposals(self,context,rng):
        p=[dict(reference_index=i,max_shift_pixels=5) for i in [0,1]]
        return [x for x in p if self.compatibility(replace(context,parameters=dict(context.parameters,**x))).eligible]
    def run(self,context):
        if not self.compatibility(context).eligible:raise ValueError(str(self.compatibility(context).reasons))
        started=perf_counter();p={k:context.parameters[k] for k in ['reference_index','max_shift_pixels']};i=p['reference_index']
        ds=context.sources[1-i].dataset;ref=context.sources[i].dataset
        arrays=dict(values=ds[context.variable].values,reference=ref[context.variable].values,lat=ds.lat.values,lon=ds.lon.values)
        result,report=geo_call('arosics',arrays,dict(p,crs=ds.attrs['crs']))
        return output(context,self.spec,ds,result,p,report,started)


class StarfmRaster:
    def __init__(self):
        self.spec=AlgorithmSpec('fusion.starfm','STARFM visible-pair reflectance fusion','starfm4py pinned b1a917f','multi_source_fusion',
            'available' if (ROOT/'external/starfm4py/src/starfm4py.py').exists() and dependency('dask') and dependency('zarr') else 'dependency_missing',
            ('fine_sparse_reflectance','coarse_dense_reflectance'),('continuous',),compute='medium',license='GPL-3.0',
            source_url='https://github.com/nmileva/starfm4py',backend='Unmodified official starfm4py.predict; streamed window arrays, isolated parameters',
            capabilities=('multi_source_fusion','spatiotemporal_reflectance_fusion'),input_contract=dict(source_arity=2,
                requires_parameters=['fine_index','window_pixels','spatial_impact_m']),output_contract=dict(accuracy_gain='UNKNOWN',range_preservation='UNKNOWN'),
            notes='Reflectance only, not arbitrary NDVI/precipitation. Explicit sensor uncertainties; exact-date visible anchor pairs and target coarse observations required.')
    def fields(self,context):
        if len(context.sources)!=2 or context.semantic_type!='continuous':raise ValueError('Two timed reflectance sources required')
        fields=[s.dataset[context.variable] for s in context.sources]
        for f,s in zip(fields,context.sources):
            if f.dims!=('time','lat','lon') or f.attrs.get('units')!='1' or 'reflectance' not in str(f.attrs.get('physical_variable','')).lower():raise ValueError('Dimensionless timed reflectance required')
            if not s.dataset.attrs.get('crs') or np.isinf(f).any():raise ValueError('Explicit CRS and no infinity required')
        for axis in ['lat','lon']:
            if not np.array_equal(fields[0][axis],fields[1][axis]):raise ValueError('Regrid first; STARFM does not silently align images')
            d=np.diff(fields[0][axis]);
            if len(d)<2 or not np.allclose(d,d[0]) or not ((d>0).all() or (d<0).all()):raise ValueError('Regular spatial grid required')
        if CRS(context.sources[0].dataset.attrs['crs'])!=CRS(context.sources[1].dataset.attrs['crs']):raise ValueError('CRS mismatch')
        for key in ['physical_variable','physical_definition','spatial_semantics','temporal_semantics']:
            if fields[0].attrs.get(key) is None or fields[0].attrs.get(key)!=fields[1].attrs.get(key):raise ValueError('Physical semantics mismatch')
        for f in fields:
            dates=pd.DatetimeIndex(f.time.values)
            if dates.has_duplicates or not dates.is_monotonic_increasing:raise ValueError('Increasing unique dates required')
            if not isinstance(f.attrs.get('measurement_uncertainty'),(int,float)) or not np.isfinite(f.attrs['measurement_uncertainty']) or f.attrs['measurement_uncertainty']<0:raise ValueError('Measured/declared sensor uncertainty required; no invented default')
        return fields
    def compatibility(self,context):
        try:
            if self.spec.status!='available':raise ValueError('STARFM source and dependencies required')
            fields=self.fields(context);p=context.parameters;i=p.get('fine_index')
            if i not in (0,1):raise ValueError('Explicit fine branch required')
            fine,coarse=fields[i],fields[1-i];w=p.get('window_pixels');impact=p.get('spatial_impact_m')
            if type(w)!=int or w<3 or w%2!=1 or w>min(fine.shape[-2:]):raise ValueError('Odd supported window required')
            if not isinstance(impact,(int,float)) or not np.isfinite(impact) or impact<=0:raise ValueError('Positive spatial impact metres required')
            targets=pd.DatetimeIndex(context.parameters.get('raster_target_times',[]))
            if not len(targets) or not targets.isin(coarse.time.values).all():raise ValueError('Every target needs a visible coarse date; align time first')
            chosen=coarse.sel(time=targets)
            if not np.isfinite(chosen.values).all():raise ValueError('Target coarse fields must be finite')
            anchors=[t for t in fine.time.values if t in coarse.time.values and np.isfinite(fine.sel(time=t)).all() and np.isfinite(coarse.sel(time=t)).all()]
            if not anchors:raise ValueError('At least one complete visible same-date fine/coarse pair required')
            if any(float(f.min(skipna=True))<0 or float(f.max(skipna=True))>1 for f in fields):raise ValueError('Surface reflectance in[0,1] required')
            return Compatibility(True,())
        except (ValueError,KeyError,TypeError) as exc:return Compatibility(False,(str(exc),))
    def parameter_proposals(self,context,rng):
        try:fields=self.fields(context)
        except (ValueError,KeyError,TypeError):return []
        from pyproj import Geod
        a=fields[0];crs=CRS(context.sources[0].dataset.attrs['crs'])
        spacing=abs(float(a.lon[1]-a.lon[0]))
        if crs.is_geographic:spacing=Geod(ellps='WGS84').inv(float(a.lon[0]),float(a.lat.mean()),float(a.lon[1]),float(a.lat.mean()))[2]
        elif crs.axis_info:spacing*=crs.axis_info[0].unit_conversion_factor
        proposals=[dict(fine_index=i,window_pixels=w,spatial_impact_m=float(spacing*w)) for i in [0,1] for w in [3,5] if w<=min(a.shape[-2:])]
        return [p for p in proposals if self.compatibility(replace(context,parameters=dict(context.parameters,**p))).eligible]
    def run(self,context):
        if not self.compatibility(context).eligible:raise ValueError(str(self.compatibility(context).reasons))
        import sys
        started=perf_counter();p={k:context.parameters[k] for k in ['fine_index','window_pixels','spatial_impact_m']};i=p['fine_index']
        fine,coarse=self.fields(context)[i],self.fields(context)[1-i]
        anchors=[t for t in fine.time.values if t in coarse.time.values and np.isfinite(fine.sel(time=t)).all() and np.isfinite(coarse.sel(time=t)).all()]
        targets=pd.DatetimeIndex(context.parameters['raster_target_times']);indices=[int(np.argmin(abs(pd.DatetimeIndex(anchors)-t))) for t in targets]
        fa=np.stack([fine.sel(time=anchors[j]).values for j in indices]);ca=np.stack([coarse.sel(time=anchors[j]).values for j in indices]);ct=coarse.sel(time=targets).values
        crs=CRS(context.sources[0].dataset.attrs['crs']);spacing=abs(float(fine.lon[1]-fine.lon[0]))
        if crs.is_geographic:
            from pyproj import Geod
            spacing=Geod(ellps='WGS84').inv(float(fine.lon[0]),float(fine.lat.mean()),float(fine.lon[1]),float(fine.lat.mean()))[2]
        else:spacing*=crs.axis_info[0].unit_conversion_factor
        with TemporaryDirectory(prefix='ecodata_starfm_') as folder:
            folder=Path(folder);np.savez(folder/'input.npz',fine=fa,coarse_anchor=ca,coarse_target=ct)
            params=dict(p,spatial_impact_pixels=p['spatial_impact_m']/spacing,fine_uncertainty=float(fine.attrs['measurement_uncertainty']),coarse_uncertainty=float(coarse.attrs['measurement_uncertainty']))
            (folder/'config.json').write_text(json.dumps(params),encoding='utf-8')
            job=subprocess.run([sys.executable,str(ROOT/'tools/run_starfm.py'),str(folder/'input.npz'),str(folder/'config.json'),str(folder/'output.npy')],
                capture_output=True,text=True,encoding='utf-8',timeout=600)
            if job.returncode:raise RuntimeError('STARFM backend failed: '+job.stderr[-2500:])
            result=np.load(folder/'output.npy')
        if not np.isfinite(result).all():raise ValueError('STARFM window has no valid contributors; not silently replaced by another method')
        ds=context.sources[i].dataset.reindex(time=targets)
        return output(context,self.spec,ds,result,p,dict(anchors=[str(anchors[j]) for j in indices],
            anchor_policy='nearest complete visible pair, future visible anchor permitted for historical reconstruction',zero_reflectance='Upstream similarity threshold treats zero as invalid'),started)


def plugins():
    result=[WhittakerRaster(),DineofRaster(),GuidedRaster(),ArosicsRaster(),StarfmRaster()]
    for plugin in result:
        plugin.spec=replace(plugin.spec,verification_evidence=('tests/test_advanced_complements.py',),
            verification_status='VERIFIED' if plugin.spec.status=='available' else 'UNVERIFIED')
    return result
