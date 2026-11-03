"""Complementary raster operators over scikit-image and PyWavelets APIs.

No workflow pool, fitted ranking model or task-specific algorithm preference.
The wavelet detail is diagnostic data, never a stand-alone target prediction.
"""
from importlib.metadata import version,PackageNotFoundError
from dataclasses import replace
from time import perf_counter
import hashlib,json
import numpy as np
import xarray as xr
from pyproj import CRS
from ecodata.algorithms.contracts import AlgorithmSpec,Compatibility
from ecodata.algorithms.open_source_spatial import finish
from ecodata.core.schemas import EcoDataBundle


def dependency(name):
    try:return version(name)
    except PackageNotFoundError:return None


def raster(context,arity=1):
    if len(context.sources)!=arity or context.semantic_type!='continuous':raise ValueError('Continuous raster parents required')
    fields=[]
    for source in context.sources:
        ds=source.dataset;a=ds[context.variable]
        if a.dims not in {('lat','lon'),('time','lat','lon')} or min(a.sizes['lat'],a.sizes['lon'])<8:
            raise ValueError('Regular spatial raster of at least8x8 required')
        if not np.isfinite(a.values).all():raise ValueError('Finite parents required; retain QA/missing recovery as separate operations')
        if not a.attrs.get('units') or not a.attrs.get('physical_variable') or not ds.attrs.get('crs'):
            raise ValueError('Explicit units, physical variable and CRS required')
        for axis in ['lat','lon']:
            d=np.diff(ds[axis].values)
            if not ((d>0).all() or (d<0).all()) or not np.allclose(d,d[0]):raise ValueError('Regular monotone pixel axes required')
        fields.append(a)
    if arity==2:
        xr.align(*fields,join='exact')
        for key in ['units','physical_variable','physical_definition','spatial_semantics','temporal_semantics']:
            if fields[0].attrs.get(key) in {None,'UNKNOWN'} or fields[0].attrs.get(key)!=fields[1].attrs.get(key):
                raise ValueError('Aligned same-variable fields required: '+key)
        if CRS(context.sources[0].dataset.attrs['crs'])!=CRS(context.sources[1].dataset.attrs['crs']):raise ValueError('CRS mismatch')
    return fields


def stamp(output,variable,parents,operation,parameters):
    prior=[p[variable].attrs.get('distribution_transform','raw') for p in parents]
    signature=hashlib.sha256(json.dumps([prior,operation,parameters],sort_keys=True).encode()).hexdigest()
    output[variable].attrs['distribution_transform']=signature
    output.attrs['distribution_transform']=signature


class EdgeDenoise:
    def __init__(self,method):
        self.method=method;v=dependency('scikit-image');wave=dependency('PyWavelets')
        self.spec=AlgorithmSpec('spatial.denoise_'+method,'Edge-aware '+method+' denoising',v or 'not installed',
            'spatial_filter','available' if v and wave else 'dependency_missing',('regular_raster','time_grid'),('continuous',),
            license='BSD-3-Clause',backend='scikit-image restoration; wavelet-MAD noise estimate',source_url='https://github.com/scikit-image/scikit-image',
            capabilities=('spatial_noise_suppression','edge_aware_regularization'),input_contract=dict(source_arity=1,
                requires_parameters=['noise_multiplier']+(['sigma_spatial_pixels'] if method=='bilateral' else [])),
            output_contract=dict(accuracy_gain='UNKNOWN',conservation='UNKNOWN',range_preservation='UNKNOWN'),
            verification_evidence=('tests/test_raster_complements.py',),
            verification_status='VERIFIED' if v=='0.26.0' and wave=='1.10.0' else 'UNVERIFIED',
            notes='Noise estimate can contain real fine structure; suppression/edge preservation tradeoff requires real evaluation. Pixel units, not metres.')
    def parameter_proposals(self,context,rng):
        if self.spec.status!='available':return []
        try:raster(context)
        except (ValueError,KeyError,TypeError):return []
        return [dict(noise_multiplier=float(m),**(dict(sigma_spatial_pixels=2.) if self.method=='bilateral' else {})) for m in [1.,rng.uniform(.5,2.)]]
    def compatibility(self,context):
        try:
            if self.spec.status!='available':raise ValueError('Optional image dependencies unavailable')
            raster(context)
            m=context.parameters.get('noise_multiplier')
            if not isinstance(m,(int,float)) or not np.isfinite(m) or m<=0:raise ValueError('Positive noise multiplier required')
            s=context.parameters.get('sigma_spatial_pixels',2.)
            if not isinstance(s,(int,float)) or not np.isfinite(s) or s<=0:raise ValueError('Positive spatial pixel scale required')
            return Compatibility(True,())
        except (ValueError,KeyError,TypeError) as exc:return Compatibility(False,(str(exc),))
    def run(self,context):
        if not self.compatibility(context).eligible:raise ValueError('Invalid denoising contract')
        from skimage.restoration import estimate_sigma,denoise_tv_chambolle,denoise_bilateral
        started=perf_counter();source=context.sources[0].dataset;field=source[context.variable]
        values=np.asarray(field.values,float);frames=values.reshape(-1,*values.shape[-2:]);outputs=[];sigmas=[]
        multiplier=context.parameters['noise_multiplier'];spatial=context.parameters.get('sigma_spatial_pixels',2.)
        for frame in frames:
            sigma=float(estimate_sigma(frame,channel_axis=None,average_sigmas=True)) if np.ptp(frame)>0 else 0.
            sigmas.append(sigma)
            if sigma<=np.finfo(float).eps*max(1.,float(abs(frame).max())):output=frame.copy()
            elif self.method=='tv':output=denoise_tv_chambolle(frame,weight=multiplier*sigma,channel_axis=None)
            else:
                # The backend's intensity lookup table depends on absolute values.
                # Work in a dimensionless range, then restore the original units.
                offset=float(frame.min());scale=float(np.ptp(frame))
                output=denoise_bilateral((frame-offset)/scale,sigma_color=multiplier*sigma/scale,
                    sigma_spatial=spatial,mode='reflect',channel_axis=None)*scale+offset
            outputs.append(output)
        out=source[[context.variable]].copy(deep=True);out[context.variable].values=np.stack(outputs).reshape(values.shape)
        parameters=dict(noise_multiplier=multiplier,**(dict(sigma_spatial_pixels=spatial) if self.method=='bilateral' else {}))
        stamp(out,context.variable,[source],self.spec.id,parameters)
        return finish(context,self.spec,out,dict(reference_used=False,estimated_sigma=sigmas,parameters=parameters,
             temporal_mixing=False,parameter_units='noise_multiplier dimensionless; spatial scale in pixels',accuracy_gain='UNKNOWN'),started)


def levels(field):
    import pywt
    return range(1,pywt.dwt_max_level(min(field.sizes['lat'],field.sizes['lon']),pywt.Wavelet('db2').dec_len)+1)


def lowpass(values,level):
    import pywt
    values=np.asarray(values,float);result=[];block=2**level
    for frame in values.reshape(-1,*values.shape[-2:]):
        h,w=frame.shape;padded=np.pad(frame,((0,(-h)%block),(0,(-w)%block)),mode='symmetric')
        coefficients=pywt.swt2(padded,'db2',level=level,norm=True)
        zero_details=[(a,tuple(np.zeros_like(d) for d in detail)) for a,detail in coefficients]
        result.append(pywt.iswt2(zero_details,'db2',norm=True)[:h,:w])
    return np.stack(result).reshape(values.shape)


class WaveletRaster:
    def __init__(self,fusion=False):
        self.fusion=fusion;v=dependency('PyWavelets')
        self.spec=AlgorithmSpec('fusion.wavelet_detail_transfer' if fusion else 'spatial.wavelet_lowpass',
            'Stationary wavelet detail transfer' if fusion else 'Stationary wavelet lowpass',v or 'not installed',
            'multi_source_fusion' if fusion else 'spatial_filter','available' if v else 'dependency_missing',
            ('aligned_rasters',) if fusion else ('regular_raster',),('continuous',),license='MIT AND BSD-3-Clause',
            backend='PyWavelets swt2/iswt2, db2, norm=True, symmetric pad/trim; no image registration or super-resolution model',
            source_url='https://github.com/PyWavelets/pywt',
            capabilities=('multi_source_fusion','multiscale_detail_transfer') if fusion else ('multiscale_background_extraction','spatial_noise_suppression'),
            input_contract=dict(source_arity=2 if fusion else 1,requires_parameters=['wavelet_level']+(['detail_index','detail_strength'] if fusion else [])),
            output_contract=dict(range_preservation='UNKNOWN',conservation='UNKNOWN',accuracy_gain='UNKNOWN'),
            verification_evidence=('tests/test_raster_complements.py',),
            verification_status='VERIFIED' if v=='1.10.0' else 'UNVERIFIED',
            notes='Transfer supplied detail only; cannot invent observed native resolution. May introduce ringing or amplify noise; no universal gain.')
    def parameter_proposals(self,context,rng):
        if self.spec.status!='available':return []
        try:fields=raster(context,2 if self.fusion else 1)
        except (ValueError,KeyError,TypeError):return []
        if not self.fusion:return [dict(wavelet_level=k) for k in levels(fields[0])]
        return [dict(wavelet_level=k,detail_index=i,detail_strength=float(a)) for k in levels(fields[0]) for i in [0,1] for a in [1.,rng.uniform(0,1)]]
    def compatibility(self,context):
        try:
            if self.spec.status!='available':raise ValueError('PyWavelets unavailable')
            fields=raster(context,2 if self.fusion else 1);p=context.parameters;k=p.get('wavelet_level')
            if isinstance(k,bool) or not isinstance(k,int) or k not in levels(fields[0]):raise ValueError('Valid db2 decomposition level required')
            if self.fusion:
                a=p.get('detail_strength')
                if p.get('detail_index') not in (0,1) or not isinstance(a,(int,float)) or not np.isfinite(a) or not 0<=a<=1:
                    raise ValueError('Explicit detail branch and strength in[0,1] required')
            return Compatibility(True,())
        except (ValueError,KeyError,TypeError) as exc:return Compatibility(False,(str(exc),))
    def run(self,context):
        if not self.compatibility(context).eligible:raise ValueError('Invalid wavelet contract')
        started=perf_counter();p=context.parameters;k=p['wavelet_level'];sources=[s.dataset for s in context.sources]
        index=p['detail_index'] if self.fusion else 0;source=sources[index];out=source[[context.variable]].copy(deep=True)
        if self.fusion:
            base=sources[1-index][context.variable];detail=source[context.variable]-lowpass(source[context.variable].values,k)
            background=lowpass(base.values,k);out[context.variable].values=background+p['detail_strength']*detail.values
            out['_wavelet_background']=xr.DataArray(background,dims=base.dims,coords=base.coords,attrs=dict(units=base.attrs['units'],diagnostic_only=1))
            out['_wavelet_detail']=detail;out['_wavelet_detail'].attrs=dict(units=base.attrs['units'],diagnostic_only=1)
            out.attrs.update(source_id='wavelet_composite:'+hashlib.sha256(str([s.attrs.get('source_id') for s in sources]).encode()).hexdigest()[:16],
                fusion_method='stationary_wavelet_detail_transfer')
        else:out[context.variable].values=lowpass(source[context.variable].values,k)
        parameters={name:p[name] for name in ['wavelet_level',*(['detail_index','detail_strength'] if self.fusion else [])]}
        stamp(out,context.variable,sources,self.spec.id,parameters)
        result=finish(context,self.spec,out,dict(reference_used=False,parameters=parameters,padding='symmetric to2**level, trim original grid',
            formula='L(background)+alpha*(detail_source-L(detail_source))' if self.fusion else 'inverse SWT with detail coefficients zeroed',
            accuracy_gain='UNKNOWN',coarse_consistency='UNKNOWN'),started)
        if self.fusion:
            history=tuple(record for source in context.sources for record in source.history)+result.data.history[-1:]
            result=replace(result,data=EcoDataBundle(result.data.dataset,history))
        return result


def plugins():return [EdgeDenoise('tv'),EdgeDenoise('bilateral'),WaveletRaster(),WaveletRaster(fusion=True)]
