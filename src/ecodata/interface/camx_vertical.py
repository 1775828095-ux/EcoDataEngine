"""Explicit layer-centre linear AGL interpolation; no extrapolation/fill.

Concentration-point interpolation, not a conservative column-mass scheme.
Ground interface is exactly0 m AGL by coordinate definition, not missing data.
"""
import numpy as np


def layer_centers(top_interfaces):
    z=np.asarray(top_interfaces,dtype='f8')
    if z.ndim!=3 or not np.isfinite(z).all() or np.any(z[0]<=0) or np.any(np.diff(z,axis=0)<=0):raise ValueError('Invalid positive monotonic layer-top AGL heights')
    return (z+np.concatenate([np.zeros_like(z[:1]),z[:-1]]))/2


def interpolation_weights(source_centers,target_centers):
    s=np.asarray(source_centers,dtype='f8');t=np.asarray(target_centers,dtype='f8')
    if s.ndim!=3 or t.ndim!=3 or s.shape[1:]!=t.shape[1:] or min(s.shape[0],t.shape[0])<2:raise ValueError('Height dimensions mismatch')
    if not np.isfinite(s).all() or not np.isfinite(t).all() or np.any(np.diff(s,axis=0)<=0) or np.any(np.diff(t,axis=0)<=0):raise ValueError('Non-finite or non-monotonic layer centres')
    below=int(np.sum(t<s[:1]));above=int(np.sum(t>s[-1:]))
    if below or above:raise ValueError(f'No extrapolation allowed: {below} target centres below source, {above} above source')
    indices=[];fractions=[]
    for height in t:
        idx=np.minimum(np.sum(s<=height,axis=0)-1,s.shape[0]-2)
        lower=np.take_along_axis(s,idx[None],axis=0)[0];upper=np.take_along_axis(s,(idx+1)[None],axis=0)[0]
        fraction=(height-lower)/(upper-lower)
        if np.any((fraction<0)|(fraction>1)):raise ValueError('Interpolation weight outside0..1')
        indices.append(idx);fractions.append(fraction)
    return np.asarray(indices),np.asarray(fractions)


def interpolate(values,weights):
    a=np.asarray(values,dtype='f8');indices,fraction=weights
    if not np.isfinite(a).all():raise ValueError('Concentration contains non-finite values')
    lower=np.take_along_axis(a,indices,axis=0);upper=np.take_along_axis(a,indices+1,axis=0)
    result=lower+(upper-lower)*fraction
    if not np.isfinite(result).all() or np.any(result<np.minimum(lower,upper)) or np.any(result>np.maximum(lower,upper)):raise ValueError('Interpolation violates source bracket bounds')
    return result.astype('f4')
