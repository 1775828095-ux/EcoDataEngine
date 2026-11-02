"""CALPUFF5.8 CONC.DAT2.1 reader, based on WROUT1/OUTPUT/COMPRS/WRDAT.

Total-source gridded and discrete receptor output; CTSG is unsupported.
Native concentrations are g/m3; exported values explicitly convert to ug/m3.
"""
from datetime import datetime,timedelta,timezone
import struct
import numpy as np
from ecodata.core.exceptions import SchemaError


def records(raw):
    if not raw or len(raw)>128*1024**2:
        raise SchemaError('CALPUFF浓度文件为空或超过128 MiB。')
    offset=0
    while offset<len(raw):
        if offset+8>len(raw):raise SchemaError('CALPUFF记录截断。')
        size=struct.unpack_from('<i',raw,offset)[0]
        if size<1 or offset+size+8>len(raw) or struct.unpack_from('<i',raw,offset+size+4)[0]!=size:
            raise SchemaError('CALPUFF Fortran记录长度或尾标记不一致。')
        yield raw[offset+4:offset+4+size]
        offset+=size+8


def _expand(raw,count,compressed):
    values=np.frombuffer(raw,dtype='<f4')
    if not np.isfinite(values).all():raise SchemaError('CALPUFF浓度包含非有限数值。')
    if not compressed:
        if len(values)!=count or np.any(values<0):raise SchemaError('浓度数组数量或符号错误。')
        return values.copy()
    result=np.zeros(count,dtype='float32');position=0
    for value in values:
        if value<0:
            length=int(-value)
            if length<1 or position+length>count:raise SchemaError('CALPUFF零段压缩长度无效。')
            position+=length
        else:
            if position>=count:raise SchemaError('CALPUFF压缩数据超出网格。')
            result[position]=value;position+=1
    if position!=count:raise SchemaError('CALPUFF压缩数据未覆盖完整网格。')
    return result


def read_concentrations(raw):
    chunks=list(records(raw))
    if len(chunks)<6 or len(chunks[0])!=96 or chunks[0][:16].strip()!=b'CONC.DAT' or chunks[0][16:32].strip()!=b'2.1':
        raise SchemaError('只支持CALPUFF CONC.DAT2.1。')
    if len(chunks[1])!=4:raise SchemaError('浓度文件注释计数无效。')
    ncom=struct.unpack('<i',chunks[1])[0]
    if not 0<=ncom<=10000 or ncom+5>len(chunks):raise SchemaError('浓度文件注释或头被截断。')
    header=chunks[ncom+2]
    if len(header)!=320 or header[:12].strip()!=b'CALPUFF':raise SchemaError('不支持的CALPUFF运行头。')
    ints=np.frombuffer(header[36:224],dtype='<i4')
    floats=np.frombuffer(header[36:224],dtype='<f4')
    hours,nx,ny=int(ints[5]),int(ints[8]),int(ints[9])
    ib,jb,ie,je,mesh=map(int,ints[20:25])
    msource,nrec,nctrec,lsamp,nspecies,compressed=map(int,ints[33:39])
    if msource!=0 or not 0<=nrec<=11000 or nctrec or not lsamp:
        raise SchemaError('浓度读回仅支持总源贡献网格及离散受体，不支持复杂地形受体。')
    if not (1<=hours<=168 and 1<=nspecies<=100 and 1<=ib<=ie<=nx<=265 and 1<=jb<=je<=ny<=265 and 1<=mesh<=10):
        raise SchemaError('浓度头的时间/网格/物种超出支持范围。')
    sx,sy=mesh*(ie-ib)+1,mesh*(je-jb)+1
    if sx*sy>1_000_000:raise SchemaError('采样网格超出读回上限。')
    dx,dy,x0,y0=map(float,[floats[10],floats[11],floats[13],floats[14]])
    zone_hours=float(floats[4])
    if not np.isfinite([dx,dy,x0,y0,zone_hours]).all() or min(dx,dy)<=0 or abs(zone_hours)>12:
        raise SchemaError('浓度文件网格间距或时区无效。')
    species_record=chunks[ncom+4]
    if len(species_record)!=15*nspecies:raise SchemaError('物种列表长度无效。')
    labels=[species_record[i*15:(i+1)*15] for i in range(nspecies)]
    try:names=[label[:12].decode('ascii').strip() for label in labels]
    except UnicodeError as exc:raise SchemaError('浓度文件物种名称无效。') from exc
    if len(set(names))!=nspecies:raise SchemaError('浓度物种名称重复。')
    position=ncom+5
    receptor_coordinates=None
    if nrec:
        if position>=len(chunks) or len(chunks[position])!=12*nrec:raise SchemaError('离散受体位置头长度无效。')
        receptor_coordinates=np.frombuffer(chunks[position],dtype='<f4').reshape(3,nrec).copy();position+=1
        if not np.isfinite(receptor_coordinates).all():raise SchemaError('离散受体位置含非有限值。')
    for count in ints[25:33]:
        if count>0:position+=1
    if position>len(chunks):raise SchemaError('浓度源记录头被截断。')
    arrays={name:[] for name in names};discrete={name:[] for name in names};times=[]
    def read_field(label,count):
        nonlocal position
        encoded=None
        if compressed:
            if position>=len(chunks) or len(chunks[position])!=4:raise SchemaError('缺少浓度压缩计数。')
            encoded=struct.unpack('<i',chunks[position])[0];position+=1
            if not 1<=encoded<=count:raise SchemaError('浓度压缩计数越界。')
        if position>=len(chunks):raise SchemaError('缺少浓度场。')
        payload=chunks[position];position+=1
        if payload[:15]!=label or (len(payload)-15)%4 or (encoded is not None and len(payload)!=15+4*encoded):raise SchemaError('浓度场标签或压缩长度不匹配。')
        return _expand(payload[15:],count,bool(compressed))
    tz=timezone(timedelta(hours=-zone_hours))
    for step in range(hours):
        if position+2>len(chunks) or len(chunks[position])!=32 or len(chunks[position+1])!=32:
            raise SchemaError('浓度时段/源记录被截断。')
        stamps=struct.unpack('<8i',chunks[position]);position+=2
        try:
            begin=datetime(stamps[0],1,1,tzinfo=tz)+timedelta(days=stamps[1]-1,hours=stamps[2],seconds=stamps[3])
            end=datetime(stamps[4],1,1,tzinfo=tz)+timedelta(days=stamps[5]-1,hours=stamps[6],seconds=stamps[7])
        except ValueError as exc:raise SchemaError('浓度文件时段日期无效。') from exc
        if end-begin!=timedelta(hours=1) or (times and begin.isoformat()!=times[-1]):
            raise SchemaError('浓度文件不是连续1小时时段。')
        times.append(end.isoformat())
        for name,label in zip(names,labels):
            arrays[name].append(read_field(label,sx*sy).reshape(sy,sx))
            if nrec:discrete[name].append(read_field(label,nrec))
    if position!=len(chunks):raise SchemaError('浓度文件含未解析尾记录。')
    # SETCOM/SAMPL: the first sampling point is IBSAMP-0.5 from grid edge.
    x=x0+(ib-0.5+np.arange(sx)/mesh)*dx
    y=y0+(jb-0.5+np.arange(sy)/mesh)*dy
    return dict(times=times,species=names,x_km=x,y_km=y,values_g_m3={n:np.stack(v) for n,v in arrays.items()},
                discrete_coordinates_km_m=receptor_coordinates,discrete_values_g_m3={n:np.stack(v) for n,v in discrete.items()} if nrec else {},
                projection=header[224:232].decode('ascii').strip(),datum=header[236:244].decode('ascii').strip(),
                utm_zone=int(ints[40]),hemisphere=header[232:236].decode('ascii').strip(),compressed=bool(compressed),
                period_start_local=(datetime(int(ints[0]),1,1,tzinfo=tz)+timedelta(days=int(ints[1])-1,hours=int(ints[2]),seconds=int(ints[3]))).isoformat())
