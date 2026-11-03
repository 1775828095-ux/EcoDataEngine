"""Update explicitly selected CMAQ species in a read-only engineering copy.

Unselected species are preserved, never filled or replaced with old data in place
of selected products. Whole-stream compatibility is checked by the old reader.
"""
from pathlib import Path
import hashlib


def assemble_species(raw, reference, output, *, selected, preserve_vertical=False):
    import numpy as np
    from netCDF4 import Dataset
    reference=Path(reference); output=Path(output)
    before=reference.read_bytes(); source_hash=hashlib.sha256(before).hexdigest()
    with Dataset('current', memory=raw) as current, Dataset('original', memory=before) as original:
        expected_fields=set(selected)
        if set(current.variables)-{'TFLAG'} != expected_fields or not expected_fields:
            raise ValueError('本次 CMAQ 转换物种与明确选定的替换物种不一致')
        if not expected_fields.issubset(original.variables):
            raise ValueError('不能向已有工程静默添加未知物种')
        keys=('GDTYP','P_ALP','P_BET','P_GAM','XCENT','YCENT','XORIG','YORIG','XCELL','YCELL',
              'SDATE','STIME','TSTEP','VGTYP','VGTOP','VGLVLS')
        differences=[]
        for key in keys:
            if key not in current.ncattrs() or key not in original.ncattrs():
                raise ValueError('工程或本次转换缺少网格/时段属性：'+key)
            if not np.array_equal(current.getncattr(key),original.getncattr(key)):
                if preserve_vertical and key == 'VGTYP':
                    differences.append(key)
                else:
                    raise ValueError('CMAQ 工程与新数据网格/时段不一致：'+key)
        if current['TFLAG'].shape[0] != original['TFLAG'].shape[0]:
            raise ValueError('CMAQ 替换数据时段长度不符')
        original_names=[str(original.getncattr('VAR-LIST'))[i:i+16].strip()
                        for i in range(0,len(str(original.getncattr('VAR-LIST'))),16)]
        current_names=[str(current.getncattr('VAR-LIST'))[i:i+16].strip()
                       for i in range(0,len(str(current.getncattr('VAR-LIST'))),16)]
        if set(current_names) != expected_fields:
            raise ValueError('本次 CMAQ VAR-LIST 与选择物种不一致')
        for name in selected:
            actual=current[name]; target=original[name]
            if actual.dimensions != target.dimensions or actual.shape != target.shape or actual.dtype != target.dtype:
                raise ValueError('CMAQ 替换物种维度不一致：'+name)
            if str(getattr(actual,'units','')).strip() != str(getattr(target,'units','')).strip():
                raise ValueError('CMAQ 替换物种单位不一致：'+name)
            if not np.array_equal(current['TFLAG'][:,current_names.index(name),:],
                                  original['TFLAG'][:,original_names.index(name),:]):
                raise ValueError('CMAQ 替换物种逐时标记不一致：'+name)
            values=np.ma.asarray(actual[:])
            if np.ma.getmaskarray(values).any() or not np.isfinite(values).all() or (values<0).any():
                raise ValueError('CMAQ 替换物种含缺失、非有限或负值：'+name)
        output.write_bytes(before)
        with Dataset(str(output), 'r+') as assembled:
            for name in selected:
                assembled[name][:]=current[name][:]
        with Dataset(str(output)) as assembled:
            preserved=[]
            for name,var in original.variables.items():
                wanted=current[name][:] if name in expected_fields else var[:]
                if not np.array_equal(assembled[name][:], wanted):
                    raise ValueError('CMAQ 工程副本独立读回不符：'+name)
                if name not in expected_fields:
                    preserved.append(name)
    if hashlib.sha256(reference.read_bytes()).hexdigest() != source_hash:
        raise ValueError('原工程文件被改写')
    return dict(original_sha256=source_hash, updated_species=list(selected),
                preserved_variables=preserved, preserved_vertical_differences=differences,
                vertical_policy='Explicit existing-case policy only; no physical-equivalence claim',
                output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
