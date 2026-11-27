"""Reference-based structure diagnostics; never a reward for invented detail."""
import numpy as np
import hashlib
from importlib.metadata import version, PackageNotFoundError


def compare_fields(prediction, reference, *, threshold=None):
    """Same regular target grid and physical unit; no resampling of reference."""
    if not {'lat', 'lon'} <= set(reference.dims):
        return dict(status='NOT_APPLICABLE', reason='A regular reference raster is required; paired stations are not a raster')
    import xarray as xr
    prediction, reference = xr.align(prediction, reference, join='exact')
    for axis in ('lat', 'lon'):
        delta = np.diff(reference[axis].values.astype(float))
        if len(delta) < 1 or not (np.all(delta > 0) or np.all(delta < 0)) or not np.allclose(delta, delta[0]):
            return dict(status='NOT_APPLICABLE', reason='Structure metrics require regular monotonic axes')
    if reference.attrs.get('units') != prediction.attrs.get('units'):
        raise ValueError('Structure comparison unit mismatch')
    original_shape=dict(reference.sizes)
    # Trim empty margins only. Interior holes stay UNKNOWN, never zero-filled.
    for axis in reference.dims:
        finite=np.isfinite(reference.values)
        other=tuple(i for i,d in enumerate(reference.dims) if d!=axis)
        positions=np.flatnonzero(finite.any(axis=other))
        if not len(positions):return dict(status='NO_REFERENCE',reason='No actual reference on this range')
        section={axis:slice(int(positions[0]),int(positions[-1])+1)}
        reference=reference.isel(section);prediction=prediction.isel(section)
    dims = [d for d in reference.dims if d not in {'lat', 'lon'}] + ['lat', 'lon']
    y = reference.transpose(*dims).values.astype(float)
    p = prediction.transpose(*dims).values.astype(float)
    valid = np.isfinite(y)
    if not np.isfinite(p[valid]).all():
        raise ValueError('Structure prediction missing on fixed reference support')
    errors = []
    for axis in (-2, -1):
        dy, dp = np.diff(y, axis=axis), np.diff(p, axis=axis)
        keep = np.isfinite(dy) & np.isfinite(dp)
        errors.extend((dp[keep] - dy[keep]).tolist())
    result = dict(version='spatial_structure_v2_scores', status='PASS', reference_required=True,
        original_shape=original_shape,evaluated_bounding_shape=dict(reference.sizes),
        support_sha256=hashlib.sha256(valid.tobytes()+repr((dims, y.shape, reference.attrs.get('units'))).encode()+
            b''.join(np.asarray(reference[d].values).tobytes() for d in dims)).hexdigest(),
        gradient_RMSE_per_cell=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        gradient_pairs=len(errors), unit=reference.attrs.get('units', 'UNKNOWN'),
        interpretation='Gradient per grid cell and normalized frequency shape; not physical-km spectra or independent truth',
        used_for_selection=False, spectrum_status='INSUFFICIENT', FSS_status='THRESHOLD_NOT_DECLARED')
    shape = y.shape[-2:]
    ys, ps = y.reshape(-1, *shape), p.reshape(-1, *shape)
    complete = np.isfinite(ys).all(axis=(1, 2)) & np.isfinite(ps).all(axis=(1, 2))
    result.update(spectrum_frames=int(complete.sum()), total_frames=len(ys))
    if not complete.all():
        result['spectrum_status'] = 'MISSING_VALUES_NO_ZERO_FILL'
        return result
    result['backend']=dict(spectrum='Existing NumPy normalized 2D FFT diagnostic; not a new reconstruction kernel')
    distances = []
    for a, b in zip(ps, ys):
        a,b=a-a.mean(),b-b.mean()
        aa=np.abs(np.fft.fft2(a))**2
        bb=np.abs(np.fft.fft2(b))**2
        aa=aa/aa.sum() if aa.sum() else aa
        bb=bb/bb.sum() if bb.sum() else bb
        distances.append(float(np.abs(aa-bb).sum()))
    if min(shape)>=4:result.update(spectrum_status='PASS', normalized_spectrum_L1=float(np.mean(distances)))
    else:result['spectrum_status']='GRID_TOO_SMALL'
    if threshold is not None:
        threshold = float(threshold)
        if not np.isfinite(threshold):raise ValueError('Finite event threshold in reference units required')
        try:
            from scores.spatial import fss_2d
            result['backend']['FSS']=dict(package='scores',version=version('scores'),source='https://github.com/nci/scores')
        except (ImportError,PackageNotFoundError) as exc:
            result.update(FSS_status='DEPENDENCY_UNAVAILABLE',reason=str(exc))
            return result
        scores = []
        forecast=xr.DataArray(ps,dims=('sample','lat','lon'))
        observed=xr.DataArray(ys,dims=('sample','lat','lon'))
        for scale in (1, 3, 5):
            if scale > min(shape):continue
            value=float(fss_2d(forecast,observed,event_threshold=threshold,window_size=(scale,scale),
                spatial_dims=('lat','lon'),zero_padding=False).item())
            scores.append(dict(scale_pixels=scale,value=value if np.isfinite(value) else None,
                status='PASS' if np.isfinite(value) else 'NO_EVENTS'))
        if not np.any(ps>threshold) and not np.any(ys>threshold):
            scores=[dict(scale_pixels=s['scale_pixels'],value=None,status='NO_EVENTS') for s in scores]
        result.update(FSS_status='EVALUATED', FSS=scores, event_threshold=threshold,event_comparison='> (scores default)')
    return result


def paired_grid(prediction, reference, query, times, indices, unit, *, threshold=None):
    """Reconstruct only declared Cartesian geometry, mask all held labels."""
    import xarray as xr
    lon,lat=np.unique(query[:,0]),np.unique(query[:,1])
    xx,yy=np.meshgrid(lon,lat)
    if len(lon)*len(lat)!=len(query) or not np.array_equal(np.column_stack([xx.ravel(),yy.ravel()]),query):
        return dict(status='NOT_APPLICABLE',reason='Paired sites do not prove regular raster topology')
    masked=np.full_like(reference,np.nan,dtype=float);masked[:,indices]=reference[:,indices]
    shape=(len(times),len(lat),len(lon))
    kwargs=dict(dims=('time','lat','lon'),coords=dict(time=times,lat=lat,lon=lon),attrs=dict(units=unit))
    return compare_fields(xr.DataArray(prediction.reshape(shape),**kwargs),xr.DataArray(masked.reshape(shape),**kwargs),threshold=threshold)
