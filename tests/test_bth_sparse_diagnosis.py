"""Diagnostic arithmetic and frozen-state observation-dependency checks only."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from diagnose_bth_sparse_fusion import error_decomposition, replay


def test_error_decomposition_is_exact():
    data=pd.DataFrame(dict(time=[0,0,1,1],truth=[1.,3.,2.,4.],estimate=[2.,4.,2.,2.]))
    result=error_decomposition(data,"estimate")
    np.testing.assert_allclose(result["RMSE"]**2,1.5)
    np.testing.assert_allclose(result["common_hour_error_RMSE"]**2,1.)
    np.testing.assert_allclose(result["spatial_error_RMSE"]**2,.5)


def test_fixed_rf_does_not_receive_live_values_but_idw_does():
    class FrozenModel:
        def predict(self,x):return np.asarray(x)[:,0]
        def fit(self,*args,**kwargs):raise AssertionError("Diagnosis must not fit")
    live=pd.DataFrame(dict(longitude=[115.,117.],latitude=[39.,40.],value=[10.,20.]))
    target=pd.DataFrame(dict(longitude=[116.],latitude=[39.5],geos=[15.]))
    state=dict(model=FrozenModel(),features=["geos"],residuals=live.assign(value=[-1.,1.]))
    original=replay(target,live,state)
    shifted=replay(target,live.assign(value=live.value+1),state)
    for name in original:
        np.testing.assert_allclose(shifted[name]-original[name],1. if name=="IDW" else 0.)
