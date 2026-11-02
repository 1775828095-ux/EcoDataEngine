"""Associate provided homogeneous HRU polygons with supplied land-use polygons.

SWAT+62 hru-data.hru lu_mgt is a database name, not a raster class number.
No HRU splitting, majority rule, resampling or physical parameter inference.
"""
import csv
import io
import json
import zipfile
from pathlib import PurePosixPath
import numpy as np
from pyproj import CRS
from pyproj.exceptions import CRSError
from shapely.geometry import shape
from shapely.errors import GEOSException
from shapely.validation import explain_validity
from ecodata.core.exceptions import SchemaError
from ecodata.models.adapters.swat_plus_hru import FIELDS, _table, export_hru_bundle


def _features(data, property_name, crs, label):
    try:
        obj=json.loads(data.decode('utf-8-sig'))
    except (ValueError,UnicodeError) as exc:
        raise SchemaError(f'{label}不是有效GeoJSON') from exc
    if not isinstance(obj,dict) or obj.get('type')!='FeatureCollection' or not obj.get('features'):
        raise SchemaError(f'{label}需非空FeatureCollection')
    if obj.get('crs'):
        try:declared=CRS.from_user_input(obj['crs']['properties']['name'])
        except (ValueError,KeyError,TypeError,CRSError) as exc:raise SchemaError(f'{label}CRS声明无效') from exc
        if not declared.equals(crs):raise SchemaError(f'{label}CRS与配置冲突')
    records=[]
    for i,f in enumerate(obj['features'],1):
        if not isinstance(f,dict) or f.get('type')!='Feature' or not isinstance(f.get('properties'),dict) or property_name not in f['properties']:
            raise SchemaError(f'{label}第{i}项缺属性{property_name}')
        value=f['properties'][property_name]
        if value is None or isinstance(value,(dict,list,bool)) or not str(value).strip():
            raise SchemaError(f'{label}第{i}项属性{property_name}无效')
        if not isinstance(f.get('geometry'),dict):raise SchemaError(f'{label}第{i}项缺有效几何')
        geom=f['geometry']
        if geom.get('type') not in {'Polygon','MultiPolygon'}:raise SchemaError(f'{label}需有效非空面')
        try:
            polygons=[geom['coordinates']] if geom['type']=='Polygon' else geom['coordinates']
            for rings in polygons:
                for ring in rings:
                    xy=np.asarray(ring,dtype=float)
                    if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all():
                        raise SchemaError(f'{label}只接受有限二维xy，不丢弃高程或NaN')
        except (KeyError,ValueError,TypeError) as exc:raise SchemaError(f'{label}需有效非空面坐标') from exc
        try:g=shape(f['geometry'])
        except (ValueError,KeyError,TypeError,GEOSException) as exc:raise SchemaError(f'{label}第{i}项需有效非空面') from exc
        if g.geom_type not in {'Polygon','MultiPolygon'} or g.is_empty or not g.is_valid or not np.isfinite(g.area) or g.area<=0:
            raise SchemaError(f'{label}第{i}项需有效非空面：{explain_validity(g)}')
        polygons=list(g.geoms) if g.geom_type=='MultiPolygon' else [g]
        for poly in polygons:
            for ring in [poly.exterior,*poly.interiors]:
                xy=np.asarray(ring.coords)
                if xy.shape[1]!=2 or not np.isfinite(xy).all():
                    raise SchemaError(f'{label}只接受有限二维xy，不丢弃高程或NaN')
        records.append((str(value),g))
    return records


def export_spatial_landuse(content, config):
    if config.get('target_model_version')!='62.0.0':raise SchemaError('空间关联仅核对SWAT+62.0.0')
    source=config.get('source')
    if not isinstance(source,dict) or not source.get('kind') or not source.get('description'):
        raise SchemaError('source.kind/description必须明确；HRU须来自已有工程划分')
    try:crs=CRS.from_user_input(config['source_crs'])
    except (KeyError,ValueError,TypeError,CRSError) as exc:raise SchemaError('需明确source_crs') from exc
    if not crs.is_projected or len(crs.axis_info)<2 or any(a.unit_name!='metre' for a in crs.axis_info[:2]) or config.get('coordinate_unit')!='m':
        raise SchemaError('本空间关联需同一明确米制投影CRS，coordinate_unit=m；不默默转换经纬度面积')
    if config.get('spatial_policy')!='homogeneous_full_coverage':
        raise SchemaError('spatial_policy须homogeneous_full_coverage；不取多数地类或拆分HRU')
    mapping=config.get('category_mapping')
    if not isinstance(mapping,dict) or not mapping or any(not isinstance(k,str) or not k or not isinstance(v,str) or not v for k,v in mapping.items()):
        raise SchemaError('category_mapping须显式地类代码→landuse.lum名称映射')
    try:archive=zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:raise SchemaError('上传ZIP：HRU表、两类GeoJSON及真实引用数据库') from exc
    with archive:
        entries=archive.infolist();names=[e.filename for e in entries]
        if len(names)!=len(set(names)) or len(entries)>500 or sum(e.file_size for e in entries)>64*1024**2:
            raise SchemaError('ZIP重复或超过64MB/500项限制')
        for n in names:
            p=PurePosixPath(n)
            if p.is_absolute() or '..' in p.parts or '\\' in n or ':' in n:raise SchemaError('ZIP路径不安全')
        if '__associated_hru.csv' in names:raise SchemaError('ZIP不能使用保留内部名称__associated_hru.csv')
        for k in ('table','hru_geometry','landuse_geometry'):
            if config.get(k) not in names or archive.getinfo(config[k]).is_dir():raise SchemaError(f'{k}缺ZIP内文件')
        hprop=config.get('hru_id_property');lprop=config.get('landuse_code_property')
        if not isinstance(hprop,str) or not hprop or not isinstance(lprop,str) or not lprop:
            raise SchemaError('需明确hru_id_property和landuse_code_property')
        hrus=_features(archive.read(config['hru_geometry']),hprop,crs,'HRU')
        land=_features(archive.read(config['landuse_geometry']),lprop,crs,'土地利用')
        if len(hrus)>2000 or len(land)>10000:raise SchemaError('本关联最多2000 HRU、10000地类面；大型工程需上游分区')
        ids=[k for k,g in hrus]
        if len(ids)!=len(set(ids)):raise SchemaError('HRU几何身份重复')
        # 1e-10 relative geometric tolerance, fixed before testing; only floating
        # polygon operation roundoff, never a missing-coverage filling rule.
        matches={};checks=[]
        for i,(hid,g) in enumerate(hrus):
            tol=g.area*1e-10
            if any(g.intersection(other).area>tol for _,other in hrus[:i]):
                raise SchemaError(f'HRU {hid}与其它HRU面积重叠')
            pieces=[];categories=set()
            for code,other in land:
                if code not in mapping:raise SchemaError(f'地类{code}缺category_mapping')
                part=g.intersection(other)
                if part.area>tol:
                    if any(part.intersection(p).area>tol for p in pieces):raise SchemaError(f'HRU {hid}内土地利用面重叠')
                    pieces.append(part);categories.add(code)
            from shapely.ops import unary_union
            union=unary_union(pieces)
            uncovered=g.difference(union).area
            if uncovered>tol:raise SchemaError(f'HRU {hid}土地利用覆盖缺口{uncovered:g} m2；不能填补')
            if len(categories)!=1:raise SchemaError(f'HRU {hid}含混合地类{sorted(categories)}；需上游划分，不取多数类别')
            code=next(iter(categories));matches[hid]=mapping[code]
            checks.append({'hru_id':hid,'category':code,'lu_mgt':mapping[code],'area_m2':g.area,'uncovered_m2':uncovered})
        fields=tuple(k for k in FIELDS if k!='lu_mgt')
        frame=_table(archive.read(config['table']),fields,config)
        if set(frame.id)!=set(ids):raise SchemaError('HRU表id与几何身份不一致；不重编号或丢弃对象')
        frame['lu_mgt']=[matches[k] for k in frame.id]
        stream=io.StringIO();writer=csv.writer(stream);writer.writerow(FIELDS)
        for _,r in frame.iterrows():writer.writerow([r[k] for k in FIELDS])
        raw=io.BytesIO()
        with zipfile.ZipFile(raw,'w',zipfile.ZIP_DEFLATED) as z:
            for e in entries:
                if not e.is_dir() and e.filename!='__associated_hru.csv':z.writestr(e.filename,archive.read(e))
            z.writestr('__associated_hru.csv',stream.getvalue().encode('utf8'))
        inner=dict(config,mode='hru_attributes',table='__associated_hru.csv',fields={k:k for k in FIELDS})
        files,report=export_hru_bundle(raw.getvalue(),inner)
    files['landuse_HRU_association.json']=json.dumps(checks,ensure_ascii=False,indent=2).encode('utf8')
    scope='用户已划分HRU与地类多边形的同投影、完整同类覆盖关联→hru-data.hru lu_mgt。无重采样、HRU划分、混合地类归并、参数推导、file.cio或完整流域工程。参考数据库仅核对直接名称，不认证深层参数。'
    files['README_spatial_landuse.txt']=scope.encode('utf8')
    report.update(scope=scope,spatial_checks=checks,source_crs=crs.to_string(),coordinate_unit='m',geometry_relative_roundoff_tolerance=1e-10,interpolation='none',time_axis='static')
    return files,report
