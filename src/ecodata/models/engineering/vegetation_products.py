"""Identify evidenced MODIS LAI without guessing a model fusion contract.

NASA AppEEARS documents version-specific quality layers and four-day support:
https://github.com/nasa/AppEEARS-Data-Resources
This inspector does not filter pixels, interpolate time, or overwrite MCIP LAI.
"""
from urllib.parse import urlsplit
import numpy as np

DOCUMENTATION='https://github.com/nasa/AppEEARS-Data-Resources/blob/main/R/tutorials/AppEEARS_API_Area_R.Rmd'


def inspect_vegetation(ds):
    if ds.attrs.get('product')!='MCD15A3H' or 'Lai_500m' not in ds:return None
    try:
        url=urlsplit(str(ds.attrs.get('source_url','')));port=url.port
    except ValueError:return None
    if (url.scheme!='https' or url.hostname!='modisrest.ornl.gov' or url.username
            or port not in (None,443) or url.path!='/rst/api/v1/MCD15A3H/subset'):
        return None
    field=ds['Lai_500m']
    if field.attrs.get('long_name')!='Leaf area index' or field.attrs.get('units')!='1':return None
    version=str(ds.attrs.get('collection','UNKNOWN'))
    blockers=[]
    def issue(code,message):blockers.append(dict(code=code,message=message))
    if version.upper().startswith('UNKNOWN'):
        issue('PRODUCT_VERSION_UNCONFIRMED','MODIS集合版本未在资料中确定；不能套用其他版本的质量标记规则')
    quality={}
    for name in ['FparLai_QC_raw','FparExtra_QC_raw']:
        if name in ds:
            values=ds[name].values
            unique,counts=np.unique(values,return_counts=True)
            quality[name]=dict(dimensions=list(ds[name].dims),
                values={str(v):int(n) for v,n in zip(unique,counts)},filter_applied=False)
    issue('QUALITY_POLICY_UNCONFIRMED','保留原始QC字段；尚未核验并绑定该版本的可用像元质量规则，未静默筛选')
    issue('TARGET_TIME_SUPPORT_UNCONFIRMED','四日合成LAI不能直接当逐小时MCIP观测；缺少明确的目标时间支持与使用规则')
    issue('TARGET_LAI_DEFINITION_UNCONFIRMED','需核对遥感LAI与目标MCIP LAI的植被面积/网格面积定义；同单位不等于同物理支持')
    issue('TARGET_GRID_SUPPORT_UNCONFIRMED','需核对正弦投影细网格与目标网格的范围、覆盖和面积支持；尚未重采样或外推')
    return dict(product='MODIS_MCD15A3H',source_id='MODIS_MCD15A3H:'+version,
        documentation=DOCUMENTATION,independence='NOT_ESTABLISHED',
        evidence={k:ds.attrs[k] for k in ['product','collection','source_url','source_sha256',
            'time_support','time_coverage_start','time_coverage_end','actual_cell_size_m'] if k in ds.attrs},
        quality_flags=quality,fields=[dict(field='Lai_500m',quantity='leaf_area_index',
            dimensions=list(field.dims),units=field.attrs['units'],blockers=blockers)])
