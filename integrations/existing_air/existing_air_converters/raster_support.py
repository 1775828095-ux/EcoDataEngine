"""Shared quality-mask and footprint checks; no model/case/field names."""
import numpy as np

def aggregate_complete_blocks(values, valid, transform, target_spacing, *, require_all=True):
    """Mean complete, all-QC-good source blocks at an engineering target scale.

    Keeps the source CRS/origin. Never moves source centres onto query centres,
    subsamples observations, invents missing pixels or relaxes pixel QC. Edge
    blocks remain excluded. With require_all=False, partially supported blocks
    are means of passed pixels ONLY, with explicit effective coverage receipts.
    This is a raster scale adapter, not a fusion estimator or accuracy test.
    Only square north-up metre grids are accepted by this narrow primitive.
    """
    a=np.asarray(values,dtype='float64');good=np.asarray(valid,bool)
    if a.ndim!=2 or good.shape!=a.shape or not np.isfinite(a[good]).all():
        raise ValueError('Invalid continuous raster / quality mask')
    if transform.b or transform.d or transform.a<=0 or transform.e>=0 or not np.isclose(transform.a,-transform.e):
        raise ValueError('Block aggregation requires a square north-up source grid')
    if not np.isfinite(target_spacing) or target_spacing<=0:raise ValueError('Invalid target spacing')
    factor=max(1,int(np.ceil(target_spacing/transform.a)))
    nr,nc=a.shape;out_shape=((nr+factor-1)//factor,(nc+factor-1)//factor)
    means=np.full(out_shape,np.nan);coverage=np.zeros(out_shape);complete=np.zeros(out_shape,bool)
    for r in range(out_shape[0]):
        for c in range(out_shape[1]):
            sl=(slice(r*factor,min(nr,(r+1)*factor)),slice(c*factor,min(nc,(c+1)*factor)))
            g=good[sl];coverage[r,c]=g.sum()/(factor*factor)
            complete[r,c]=g.shape==(factor,factor) and (g.all() if require_all else g.any())
            if complete[r,c]:means[r,c]=a[sl][g].mean()
    from affine import Affine
    aggregated_transform=transform*Affine.scale(factor,factor)
    return means,complete,aggregated_transform,dict(
        source_shape=list(a.shape),aggregated_shape=list(out_shape),factor=factor,
        source_spacing=float(transform.a),requested_target_spacing=float(target_spacing),
        actual_spacing=float(factor*transform.a),complete_good_blocks=int(complete.sum()),
        coverage_fraction=coverage.tolist(),require_all_pixels_good=bool(require_all),
        policy=('Full source blocks, every pixel QC-good; edge/partial/failed blocks excluded' if require_all else
            'Complete geometric blocks; average QC-passed pixels only; no imputation for failed pixels; effective coverage explicitly retained'),
        weighting='Arithmetic mean of equal-area pixels in original projected source CRS',
        limitations='Source-grid aggregation to nominal engineering scale; not geodesically conservative target-cell remapping')

def remap_valid_fraction(valid,shape,kwargs):
    """Fraction of contributing raster area with passed QC; zero is real data."""
    from rasterio.warp import reproject,Resampling
    a=np.asarray(valid)
    if a.ndim!=2 or not np.isin(a,[0,1]).all():raise ValueError('Quality mask must be 2D zero/one')
    out=np.full(shape,-9999.,dtype='float64')
    reproject(a.astype('float64'),out,src_nodata=-9999.,dst_nodata=-9999.,resampling=Resampling.average,**kwargs)
    if np.any((out!=-9999.)&((out<0)|(out>1))):raise ValueError('Invalid effective coverage')
    return out

def remap_quality(valid,shape,kwargs):
    from rasterio.warp import reproject,Resampling
    a=np.asarray(valid)
    if a.ndim!=2 or not np.isin(a,[0,1]).all():raise ValueError('质量掩码必须是二维0/1')
    out=np.full(shape,-9999.,dtype='float64')
    reproject(a.astype('float64'),out,src_nodata=-9999.,dst_nodata=-9999.,
              resampling=Resampling.min,**kwargs)
    if not np.isin(out,[-9999.,0.,1.]).all():raise ValueError('质量重网格产生了非法类别值')
    return out

def covered_cells(transform,shape,target_crs,source_bounds,source_crs,candidates=None):
    """Densified target-cell bounds in SOURCE CRS, not same-unit assumptions."""
    from rasterio.warp import transform_bounds
    mask=np.ones(shape,bool) if candidates is None else np.asarray(candidates,bool)
    if mask.shape!=tuple(shape):raise ValueError('覆盖候选掩码维度错误')
    inside=np.zeros(shape,bool);left,bottom,right,top=source_bounds
    for row,col in zip(*np.where(mask)):
        x0,y0=transform*(int(col),int(row));x1,y1=transform*(int(col)+1,int(row)+1)
        a,b,c,d=transform_bounds(target_crs,source_crs,min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1),densify_pts=21)
        inside[row,col]=bool(np.isfinite([a,b,c,d]).all() and a>=left and b>=bottom and c<=right and d<=top)
    return inside
