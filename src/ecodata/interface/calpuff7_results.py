"""CONC7.0 discrete, source-resolved reader; native COMPRS/UNCOMPRS layout."""
import re,struct
from datetime import datetime,timedelta,timezone
import numpy as np
from ecodata.core.exceptions import SchemaError
from ecodata.interface.calpuff_results import records,_expand
from ecodata.models.adapters.calpuff_receptors import TOKEN

def control_values(text):
    result={}
    for t in TOKEN.finditer(text):
        m=re.fullmatch(r'\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*',t[1],re.S)
        if m:result.setdefault(m[1].upper(),[]).append(m[2].strip())
    return result

def read_discrete(raw,control):
    values=control_values(control)
    def one(k):
        if len(values.get(k,[]))!=1:raise SchemaError('浓度读回控制字段缺少或重复：'+k)
        return values[k][0]
    if one('LSAMP')!='F' or one('MSOURCE')!='1' or int(one('NRGRP'))!=0:
        raise SchemaError('7.0当前只支持LSAMP=F、MSOURCE=1、NRGRP=0离散受体')
    nrec=int(one('NREC'));npt=int(one('NPT1'));names=values.get('SRCNAM',[])
    if not 1<=nrec<=11000 or not 1<=npt<=200 or len(names)!=npt or len(set(names))!=npt:raise SchemaError('受体/点源计数无效')
    if any(int(one(k))!=0 for k in ('NPT2','NSPT1','NAR1','NAR2','NVL1','NVL2','NLN2','NRD1','NRD2','NFL2')):
        raise SchemaError('当前读回仅验证常量点源，不丢弃其他源')
    zone=re.fullmatch(r'UTC([+-])(\d{2})(\d{2})',one('ABTZ'))
    if not zone:raise SchemaError('需明确UTC偏移时区')
    hours=int(zone[2])+int(zone[3])/60
    if hours>14:raise SchemaError('时区越界')
    tz=timezone(timedelta(hours=hours*(1 if zone[1]=='+' else -1)))
    def date(prefix):
        return datetime(int(one(prefix+'YR')),int(one(prefix+'MO')),int(one(prefix+'DY')),int(one(prefix+'HR')),tzinfo=tz)
    begin,end=date('IB'),date('IE');span=(end-begin).total_seconds()
    if not 3600<=span<=168*3600 or span%3600:raise SchemaError('当前仅支持1..168连续整小时时段')
    periods=int(span/3600);chunks=list(records(raw))
    if len(chunks)<10 or not chunks[0].startswith(b'CONC.DAT        7.0'):raise SchemaError('需CONC.DAT7.0')
    if len(chunks[1])!=4:raise SchemaError('注释头无效')
    ncom=struct.unpack('<i',chunks[1])[0]
    if not 0<=ncom<=10000 or ncom+8>=len(chunks):raise SchemaError('注释头截断')
    labels=[chunks[ncom+5][i:i+15] for i in range(0,len(chunks[ncom+5]),15)]
    species=[x[:12].decode('ascii').strip() for x in labels]
    if species!=[x.upper() for x in values.get('CSPEC',[])] or len(species)!=int(one('NSPEC')):raise SchemaError('二进制物种与控制不一致')
    if chunks[ncom+6].strip()!=b'g/m3' or len(chunks[ncom+7])!=20*nrec:raise SchemaError('单位或受体坐标头不符')
    coords=np.frombuffer(chunks[ncom+7],dtype='<f4').reshape(5,nrec).copy()
    if not np.isfinite(coords).all():raise SchemaError('受体坐标非有限')
    compressed=one('LCOMPRS')=='T'
    if one('LCOMPRS') not in {'T','F'}:raise SchemaError('压缩标志无效')
    stride=2+len(species)*(2 if compressed else 1)
    count=periods*(npt+1)*stride;start=len(chunks)-count
    if start<=ncom+8:raise SchemaError('浓度数据被截断')
    arrays={s:[] for s in species};source_arrays={name:{s:[] for s in species} for name in names}
    pos=start;times=[]
    for h in range(periods):
        expected_begin=begin+timedelta(hours=h);expected_end=expected_begin+timedelta(hours=1)
        stamp_expected=(expected_begin.year,int(expected_begin.strftime('%j')),expected_begin.hour,0,expected_end.year,int(expected_end.strftime('%j')),expected_end.hour,0)
        hourly=[]
        for name in names+['TOTAL']:
            if len(chunks[pos])!=32 or struct.unpack('<8i',chunks[pos])!=stamp_expected:raise SchemaError('二进制时段与控制不一致')
            if len(chunks[pos+1])!=32 or chunks[pos+1][8:24].strip().decode('ascii')!=name:raise SchemaError('浓度源标签顺序不符')
            pos+=2;group={}
            for s,label in zip(species,labels):
                packed=None
                if compressed:
                    if len(chunks[pos])!=4:raise SchemaError('缺压缩计数')
                    packed=struct.unpack('<i',chunks[pos])[0];pos+=1
                    if not 1<=packed<=nrec:raise SchemaError('压缩计数无效')
                payload=chunks[pos];pos+=1
                if payload[:15]!=label or (len(payload)-15)%4 or (packed is not None and len(payload)!=15+4*packed):raise SchemaError('浓度标签/长度不符')
                group[s]=_expand(payload[15:],nrec,compressed)
                (arrays if name=='TOTAL' else source_arrays[name])[s].append(group[s])
            hourly.append(group)
        for s in species:
            if not np.allclose(np.sum([g[s].astype('f8') for g in hourly[:-1]],axis=0),hourly[-1][s],rtol=3e-6,atol=1e-35):raise SchemaError('分源贡献与总量不一致')
        times.append(expected_end.isoformat())
    if pos!=len(chunks):raise SchemaError('未解析尾记录')
    return dict(times=times,species=species,source_names=names,coordinates=coords,values_g_m3={s:np.stack(v) for s,v in arrays.items()},source_values_g_m3={n:{s:np.stack(v) for s,v in d.items()} for n,d in source_arrays.items()},compressed=compressed)
