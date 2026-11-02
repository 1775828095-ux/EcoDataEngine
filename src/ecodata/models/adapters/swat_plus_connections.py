"""SWAT+ 62 hyd_read_connect records from explicitly supplied network tables."""
from collections import defaultdict, deque
import io
import re
import zipfile
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from ecodata.core.exceptions import SchemaError
from ecodata.models.adaptation import unit_transform
from ecodata.models.adapters.swat_plus_hru import _table, _name

NODES = ('kind','id','name','gis_id','area','lat','lon','elev','props','wst','cst','ovfl','rule')
EDGES = ('from_kind','from_id','order','to_kind','to_id','hyd_typ','fraction')
PROFILES = {'hru':('hru.con','hru-data.hru',10), 'aqu':('aquifer.con','aquifer.aqu',18),
            'ru':('rout_unit.con','rout_unit.rtu',6), 'sdc':('chandeg.con','channel-lte.cha',6)}
HYDRO = {'tot','rhg','sur','lat','til'}


def _integer(value, name, minimum=0):
    try:result=int(str(value))
    except (ValueError, TypeError) as exc:raise SchemaError(f'{name}必须为整数') from exc
    if str(result)!=str(value).strip() or result<minimum:raise SchemaError(f'{name}整数范围不符')
    return result


def _number(value, name, markers):
    try:result=float(value)
    except (ValueError,TypeError) as exc:raise SchemaError(f'{name}不是有效数值') from exc
    numeric_markers=[]
    for marker in markers:
        try:numeric_markers.append(float(marker))
        except (ValueError,TypeError):pass
    if not np.isfinite(result) or result in numeric_markers:
        raise SchemaError(f'{name}含缺测或非有限值')
    return result


def export_connections(content, config):
    if config.get('target_model_version')!='62.0.0':raise SchemaError('连接格式仅核对SWAT+62.0.0')
    if config.get('fraction_policy')!='sum_to_one_per_hydrograph':raise SchemaError('需明确fraction_policy=sum_to_one_per_hydrograph；不估算出流比例')
    if not isinstance(config.get('source'),dict) or not config['source'].get('description'):raise SchemaError('需说明真实工程拓扑来源或明确标为合成/反解样例')
    markers=config.get('missing_values')
    if not isinstance(markers,list):raise SchemaError('需明确missing_values列表')
    units=config.get('units')
    if not isinstance(units,dict) or set(units)!={'area','elev','coordinates','fraction'}:raise SchemaError('需明确area/elev/coordinates/fraction单位')
    if units['fraction'] not in {'1','%'}:raise SchemaError('fraction单位只允许1或%')
    try:archive=zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:raise SchemaError('连接输入必须为ZIP：对象CSV、出流CSV和实际引用数据库') from exc
    with archive:
        entries=archive.infolist();names=[e.filename for e in entries]
        if len(entries)>200 or sum(e.file_size for e in entries)>64*1024**2 or len(names)!=len(set(names)):
            raise SchemaError('ZIP重复成员或超过64MB/200文件限制')
        if any(not re.fullmatch(r'[A-Za-z0-9_.-]+',n) or '..' in n for n in names):raise SchemaError('ZIP仅接受安全单层文件名')
        def read(name):
            if name not in names:raise SchemaError(f'ZIP缺少引用文件{name}')
            return archive.read(name)
        def table(role,fields):
            setting=config.get(role)
            if not isinstance(setting,dict):raise SchemaError(f'缺少{role}字段映射')
            return _table(read(setting.get('table')),fields,setting)
        nodes=table('nodes',NODES)
        # An explicit empty edges CSV is valid for terminal objects, but its columns must be complete.
        edge_setting=config.get('edges',{})
        edge_raw=read(edge_setting.get('table'))
        try:edges=_table(edge_raw,EDGES,edge_setting)
        except SchemaError:
            import csv
            rows=list(csv.reader(io.StringIO(edge_raw.decode(edge_setting.get('encoding','utf-8-sig')))))
            mapping=edge_setting.get('fields',{})
            if len(rows)!=1 or set(mapping)!=set(EDGES) or len(set(mapping.values()))!=len(EDGES) or set(rows[0])!=set(mapping.values()):raise
            edges=pd.DataFrame(columns=EDGES)
        if not set(nodes.kind)<=set(PROFILES):raise SchemaError('仅支持hru/aqu/ru/sdc连接；其它对象需专用数据库与读取规则')
        try:crs=CRS.from_user_input(config.get('source_crs'))
        except Exception as exc:raise SchemaError('需明确可解析source_crs') from exc
        if crs.is_geographic:
            if not crs.equals(CRS.from_epsg(4326)) or units['coordinates']!='degree':raise SchemaError('经纬度需EPSG:4326及degree单位')
            transform=None
        elif crs.is_projected:
            if units['coordinates']!='m' or any(a.unit_name!='metre' for a in crs.axis_info):raise SchemaError('投影坐标需明确米制CRS和m单位')
            transform=Transformer.from_crs(crs,4326,always_xy=True)
        else:raise SchemaError('不支持该坐标系统')
        catalog={}; property_refs={}
        for kind in sorted(set(nodes.kind)):
            filename=config.get('property_files',{}).get(kind)
            lines=read(filename).decode('ascii').splitlines()
            ids=set()
            for line in lines[2:]:
                tokens=line.split()
                if len(tokens)<PROFILES[kind][2]:raise SchemaError(f'{filename}数据库记录字段不足')
                ident=_integer(tokens[0],f'{filename}.id',1)
                if ident in ids:raise SchemaError(f'{filename}数据库id重复')
                if ident!=len(ids)+1:raise SchemaError(f'{filename}数据库必须按连续1..N排列；props是读取器数组位置')
                ids.add(ident)
            if not ids:raise SchemaError(f'{filename}数据库为空')
            property_refs[kind]=ids
        weather=read(config.get('weather_stations_file')).decode('ascii').splitlines()[2:]
        weather_names=set()
        for line in weather:
            tokens=line.split()
            if len(tokens)<8:raise SchemaError('weather-sta.cli记录不完整')
            if tokens[0] in weather_names:raise SchemaError('weather-sta.cli站点名称重复')
            weather_names.add(tokens[0])
        area_scale,area_offset=unit_transform(units['area'],'ha')
        elev_scale,elev_offset=unit_transform(units['elev'],'m')
        if area_offset or elev_offset:raise SchemaError('面积和高程单位不得带偏移')
        for _,raw in nodes.iterrows():
            row=dict(raw);kind=row['kind'];ident=_integer(row['id'],'node.id',1);key=(kind,ident)
            if key in catalog:raise SchemaError(f'对象重复{key}')
            row.update(id=ident,name=_name(row['name'],'name',16),gis_id=_integer(row['gis_id'],'gis_id'),props=_integer(row['props'],'props',1))
            row['wst']=_name(row['wst'],'wst',50)
            if row['props'] not in property_refs[kind]:raise SchemaError(f'{key}属性引用props={row["props"]}不存在')
            if row['wst'] not in weather_names:raise SchemaError(f'{key}天气站引用{row["wst"]}不存在')
            if row['cst']!='0' or row['ovfl']!='0' or row['rule']!='0':raise SchemaError('本连接profile暂不支持非0 cst/ovfl/rule；不能隐式忽略高级引用')
            for field in ('area','lat','lon','elev'):row[field]=_number(row[field],field,markers)
            row['area']*=area_scale;row['elev']*=elev_scale
            if not np.isfinite([row['area'],row['elev']]).all():raise SchemaError('单位转换后面积或高程非有限')
            if row['area']<=0:raise SchemaError('对象面积必须为正')
            if transform:row['lon'],row['lat']=transform.transform(row['lon'],row['lat'],errcheck=True)
            if not -90<=row['lat']<=90 or not -180<=row['lon']<=180:raise SchemaError('对象经纬度超范围')
            catalog[key]=row
        for kind in set(nodes.kind):
            selected=[row for (k,_),row in catalog.items() if k==kind]
            if sorted(r['id'] for r in selected)!=list(range(1,len(selected)+1)):raise SchemaError(f'{kind}连接行号必须连续1..N；不重编号拓扑')
            if len({r['name'] for r in selected})!=len(selected):raise SchemaError(f'{kind}名称重复')
        routes=defaultdict(list);fractions=defaultdict(float);graph=defaultdict(set);indegree=dict.fromkeys(catalog,0)
        for _,raw in edges.iterrows():
            row=dict(raw);origin=(row['from_kind'],_integer(row['from_id'],'from_id',1));target=(row['to_kind'],_integer(row['to_id'],'to_id',1))
            if origin not in catalog or target not in catalog:raise SchemaError(f'出流引用缺失：{origin}->{target}')
            if origin==target:raise SchemaError('不能连接对象自身')
            hydro=row['hyd_typ']
            if hydro not in HYDRO:raise SchemaError('hyd_typ须为tot/rhg/sur/lat/til')
            fraction=_number(row['fraction'],'fraction',markers)*(0.01 if units['fraction']=='%' else 1)
            if not 0<fraction<=1:raise SchemaError('出流比例须位于(0,1]')
            order=_integer(row['order'],'edge.order',1)
            routes[origin].append((order,target[0],target[1],hydro,fraction));fractions[origin,hydro]+=fraction
            if target not in graph[origin]:graph[origin].add(target);indegree[target]+=1
        for key,items in routes.items():
            if sorted(item[0] for item in items)!=list(range(1,len(items)+1)):raise SchemaError(f'{key}出流order必须唯一连续1..N')
            if len({(i[1],i[2],i[3]) for i in items})!=len(items):raise SchemaError('出流边重复')
        if any(abs(total-1)>1e-8 for total in fractions.values()):raise SchemaError('每对象每hyd_typ出流比例合计必须为1（暂定数值容差1e-8）')
        queue=deque(key for key,n in indegree.items() if n==0);visited=0
        while queue:
            current=queue.popleft();visited+=1
            for target in graph[current]:
                indegree[target]-=1
                if indegree[target]==0:queue.append(target)
        if visited!=len(catalog):raise SchemaError('对象网络存在循环；不擅自改变拓扑或运行顺序')
    output={}
    for kind in sorted(set(nodes.kind)):
        filename=PROFILES[kind][0];lines=[filename+' EcoDataEngine SWAT+62 supplied connections', 'id name gis_id area lat lon elev props wst cst ovfl rule out_tot obj_typ obj_id hyd_typ frac']
        for key,row in sorted(catalog.items()):
            if key[0]!=kind:continue
            values=[row[f] for f in NODES[1:]]+[len(routes[key])]
            for _,typ,ident,hydro,fraction in sorted(routes[key]):values.extend([typ,ident,hydro,fraction])
            lines.append(' '.join(format(v,'.17g') if isinstance(v,float) else str(v) for v in values))
        output[filename]=('\n'.join(lines)+'\n').encode('ascii')
    scope='SWAT+62显式对象连接；直接属性ID/天气站/网络端点已校验，不划分流域或HRU，不校验属性库内部深层引用，不生成file.cio；非0高级引用与循环网络暂不支持。'
    output['README_connections.txt']=scope.encode('utf8')
    return output,dict(scope=scope,nodes=len(catalog),edges=len(edges),node_types=sorted(set(nodes.kind)),source=config['source'],units={'area':'ha','coordinates':'degree EPSG:4326','elev':'m','fraction':'1'},complete_model_input=False,model_run_verified=False)
