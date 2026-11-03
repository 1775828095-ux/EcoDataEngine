"""Existing temporal numerical kernels, independent of graph runtimes."""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from ecodata.algorithms.dag import DAGError


def reconstruct(series,method):
    if not isinstance(series.index,pd.DatetimeIndex) or series.index.has_duplicates or not series.index.is_monotonic_increasing:
        raise DAGError("Unique increasing per-entity time axis required")
    visible=series.dropna()
    if len(visible)<2 or np.isinf(visible).any():raise DAGError("Insufficient finite retained observations")
    if method=="linear":return series.interpolate(method="time",limit_area="inside")
    if method=="ffill":return series.ffill()
    if method=="hour_median":
        medians=visible.groupby(visible.index.hour).median()
        fill=pd.Series(series.index.hour,index=series.index).map(medians).fillna(visible.median())
        return series.fillna(fill)
    if method=="nearest":
        # Explicit nearest-observation endpoint extension, not a physical forecast.
        t=(series.index-series.index[0]).total_seconds().to_numpy()
        valid=series.notna().to_numpy()
        values=interp1d(t[valid],visible.to_numpy(),kind="nearest",bounds_error=False,
                       fill_value=(visible.iloc[0],visible.iloc[-1]))(t)
        return pd.Series(values,index=series.index)
    raise DAGError("Unknown temporal baseline")

