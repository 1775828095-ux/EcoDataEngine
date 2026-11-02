"""CAMx7.40 BOUNDARY full-grid arrays, explicit one/six-hour intervals.

NCF_READBND reads COL*ROW*LAY, then consumes lateral edge cells. Requiring the
provided complete volume avoids inventing zero/empty interior and corner cells.
"""
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from ecodata.core.exceptions import SchemaError
from ecodata.models.adapters.camx_initial import export_initial
from ecodata.models.adapters.camx_emissions import _grid


def export_boundary(frame,cfg):
    seconds=cfg.get('interval_seconds')
    if cfg.get('input_layout')!='full_grid_3d' or type(seconds)!=int or seconds not in {3600,21600}:
        raise SchemaError('CAMx边界需input_layout=full_grid_3d、interval_seconds=3600或21600；不将外圈表填零扩展')
    step=pd.Timedelta(seconds=seconds)
    try:
        timecol=cfg['fields']['time'];rawtime=frame[timecol].astype(str)
        if not rawtime.str.contains(r'(?:Z|[+-]\d{2}:?\d{2})$',regex=True).all():raise SchemaError('源时间需明确UTC偏移')
        times=pd.to_datetime(rawtime,utc=True,format='mixed')
        start=pd.Timestamp(cfg['start_time']);end=pd.Timestamp(cfg['valid_until'])
    except (ValueError,KeyError,TypeError) as exc:raise SchemaError('缺time映射或明确start_time/valid_until') from exc
    if start.tzinfo is None or end.tzinfo is None or end<=start or any(t.minute or t.second or t.microsecond or t.nanosecond for t in (start,end)):
        raise SchemaError('start_time/valid_until需带偏移整点、递增区间')
    if (end-start)>pd.Timedelta(hours=744):raise SchemaError('当前单包最多744小时；大案例须显式分包')
    if (end-start)%step:raise SchemaError('边界结束时间须与显式间隔整除匹配')
    expected=pd.date_range(start.tz_convert('UTC'),end.tz_convert('UTC'),freq=step,inclusive='left')
    actual=pd.DatetimeIndex(times.unique()).sort_values()
    if len(expected)<2 or not actual.equals(expected):raise SchemaError('边界须至少2个按显式间隔连续的UTC时段；缺时刻或配置外时刻，不复制快照')
    grid,*_=_grid(cfg.get('grid'))
    if grid['ncols']<3 or grid['nrows']<3:
        raise SchemaError('CAMx边界网格ncols/nrows至少3，包含边界环及内部格')
    # Validate every mapped species, full grid, units and layer definition using
    # the proven converter, without any intermediate change of physical values.
    prepared=[];reports=[]
    for t in expected:
        local=dict(cfg,valid_until=(t+step).isoformat())
        files,report=export_initial(frame.loc[times==t],local)
        prepared.append(files['initial.CAMx7.40.nc']);reports.append(report)
    temporary=tempfile.TemporaryDirectory(prefix='ecodata_camx_boundary_');path=Path(temporary.name)/'boundary.nc';out=None
    try:
        out=Dataset(path,'w',format='NETCDF3_CLASSIC')
        for it,raw in enumerate(prepared):
            with Dataset('input',memory=raw) as source:
                if it==0:
                    for name,dim in source.dimensions.items():out.createDimension(name,None if dim.isunlimited() else len(dim))
                    attrs={k:source.getncattr(k) for k in source.ncattrs()}
                    attrs.update(CAMx_NAME='BOUNDARY',TSTEP=(seconds//3600)*10000,NSTEPS=len(expected),
                                 NOTE='Supplied explicit-interval full-volume boundary concentrations; no interpolation',
                                 FILEDESC='CAMx7.40 NCF_READBND full-grid boundary arrays')
                    out.setncatts(attrs)
                    for name,v in source.variables.items():
                        target=out.createVariable(name,v.dtype,v.dimensions)
                        target.setncatts({k:v.getncattr(k) for k in v.ncattrs() if k!='_FillValue'})
                for name,v in source.variables.items():out.variables[name][it:it+1]=v[:]
        out.close();out=None;raw=path.read_bytes()
    finally:
        if out is not None:out.close()
        temporary.cleanup()
    scope=('CAMx7.40 BOUNDARY NetCDF，NCF_READBND读取完整COL×ROW×LAY后使用侧边格点。'
           '本profile要求用户给全三维显式1或6小时间隔浓度，不将仅外圈表扩展成虚假内部值。'
           '当前机制CB6r5_CF3_SIMPLE_NoEvap或COMPLX、LATLON/LCC，gas ppm、aerosol ug/m3。'
           '无时空插值、机制拆分、默认浓度，不生成气象/控制/完整工程；未运行CAMx。')
    errors={name:max(r['float32_max_absolute_error'][name] for r in reports) for name in reports[0]['species']}
    return {'boundary.CAMx7.40.nc':raw,'README_CAMx_boundary.txt':scope.encode('utf8')},dict(scope=scope,grid=reports[0]['grid'],
        source=cfg.get('source'),mechanism=reports[0]['mechanism'],species=reports[0]['species'],units=reports[0]['units'],layers=reports[0]['layers'],
        start_time=reports[0]['time'],valid_until=reports[-1]['valid_until'],hours=len(expected)*seconds//3600,interval_seconds=seconds,intervals=len(expected),input_layout='full_grid_3d',
        time_processing='no interpolation; original contiguous explicitly declared validity intervals',float32_max_absolute_error=errors,
        complete_model_input=False,model_run_verified=False)
