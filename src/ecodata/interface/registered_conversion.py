"""Fail-closed acceptance of a current conversion in a bounded benchmark.

This is deliberately not an arbitrary-region scientific compatibility test.
The registered benchmark requires identical arrays, units and grid metadata.
"""
import io
import zipfile
import hashlib
import numpy as np
from netCDF4 import Dataset


def member(payload, name):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        matches = [item for item in archive.infolist() if item.filename == name]
        if len(matches) != 1:
            raise ValueError('本次转换ZIP必须唯一包含：' + name)
        if matches[0].file_size > 512 * 1024 * 1024:
            raise ValueError('转换文件超过登记验收大小限制')
        return archive.read(matches[0])


def accept_netcdf(payload, filename, reference):
    raw = member(payload, filename)
    with Dataset('current', memory=raw) as actual, Dataset('reference', memory=reference.read_bytes()) as expected:
        if set(actual.variables) != set(expected.variables):
            raise ValueError('当前物种/变量集合不符合登记案例；不自动补齐或删减')
        if {n: len(d) for n, d in actual.dimensions.items()} != {n: len(d) for n, d in expected.dimensions.items()}:
            raise ValueError('当前维度不符合登记案例')
        for key in ('GDTYP', 'P_ALP', 'P_BET', 'P_GAM', 'XCENT', 'YCENT', 'XORIG', 'YORIG', 'XCELL', 'YCELL', 'SDATE', 'STIME', 'TSTEP', 'VGTYP', 'VGTOP', 'VGLVLS'):
            if (key in actual.ncattrs()) != (key in expected.ncattrs()):
                raise ValueError('登记网格/时间元数据缺失：' + key)
            if key in expected.ncattrs() and not np.array_equal(actual.getncattr(key), expected.getncattr(key)):
                raise ValueError('登记网格/时段/垂直定义不匹配：' + key)
        for name, variable in expected.variables.items():
            other = actual.variables[name]
            if other.dimensions != variable.dimensions or str(getattr(other, 'units', '')).strip() != str(getattr(variable, 'units', '')).strip():
                raise ValueError('变量维度或单位不匹配：' + name)
            if not np.array_equal(other[:], variable[:]):
                raise ValueError('当前数值不属于已核验登记案例：' + name + '；任意新数据兼容性尚未实现')
    return raw, dict(input_sha256=hashlib.sha256(raw).hexdigest(), reference=str(reference), scope='Exact registered benchmark only; current ZIP used, no fixture substitution')


def compatible_netcdf(raw, reference):
    """Whole-stream replacement: exact contract, new finite values permitted.

    No species complement, time interpolation, unit conversion or grid repair.
    This intentionally remains stricter than arbitrary model input compatibility.
    """
    changed=[];fields=[]
    with Dataset('current',memory=raw) as actual, Dataset('reference',memory=reference.read_bytes()) as expected:
        if set(actual.variables)!=set(expected.variables):raise ValueError('完整替换流必须保留全部变量；不补物种')
        if {n:(len(d),d.isunlimited()) for n,d in actual.dimensions.items()}!={n:(len(d),d.isunlimited()) for n,d in expected.dimensions.items()}:raise ValueError('替换流维度定义不一致')
        # Creation clocks and descriptions are provenance, not physical identity.
        ignore={'CDATE','CTIME','WDATE','WTIME','HISTORY','FILEDESC','UPNAM'}
        keys=set(expected.ncattrs())-ignore
        if keys!=set(actual.ncattrs())-ignore:raise ValueError('替换流元数据集合不一致')
        for key in keys:
            if not np.array_equal(expected.getncattr(key),actual.getncattr(key)):raise ValueError('替换流网格/机制/时段元数据不一致：'+key)
        for name,var in expected.variables.items():
            other=actual[name]
            if var.dimensions!=other.dimensions or var.dtype!=other.dtype or set(var.ncattrs())!=set(other.ncattrs()):raise ValueError('替换流变量结构不一致：'+name)
            for key in var.ncattrs():
                if not np.array_equal(var.getncattr(key),other.getncattr(key)):raise ValueError('替换流单位或变量属性不一致：'+name+'.'+key)
            value=np.ma.asarray(other[:]);before=var[:]
            if name=='TFLAG' or value.dtype.kind not in 'f':
                if not np.array_equal(value,before):raise ValueError('替换流时标或非浮点结构发生变化：'+name)
            else:
                if np.ma.getmaskarray(value).any() or not np.isfinite(value).all() or (value<0).any():raise ValueError('替换流含缺失、非有限或负值：'+name)
                fields.append(name)
                if not np.array_equal(value,before):changed.append(name)
    return dict(input_sha256=hashlib.sha256(raw).hexdigest(),reference=str(reference),validated_fields=fields,changed_fields=changed,scope='Complete same-structure stream replacement; native reading and accuracy are separate checks')
