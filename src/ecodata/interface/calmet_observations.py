"""Bounded decoding of the registered 1990 demo's original weather records.

CD144/SAMSON positions follow SMERGE RDWRITS/CLOUDS/PCODES; FSL follows
READ62; TD3240 dry-interval semantics follow PMERGE GETPREC/PREAD.
No user files or scripts are executed. Other dates/encodings are unsupported.
"""
from datetime import datetime,timedelta,timezone
import bisect
import hashlib
import math
import zipfile
import pandas as pd
from ecodata.core.exceptions import SchemaError
from ecodata.models.adapters.calmet_surface import FIELDS

STATIONS=[14606,14611,14745,14742,14764]
PRECIP_STATIONS=[170273,176905,177325,178641,270741,270998,272842,273182,274732,274808,275639,275780,276234,276818,278885,437054]
ZONE=timezone(timedelta(hours=-5))
START=datetime(1990,1,9,5,tzinfo=ZONE)
TIMES=[START+timedelta(hours=i) for i in range(4)]


def surface_config():
    return dict(version='CALMET-5.8.5-SURF-2.0',timezone_west_hours=5,time_label='hour_end',station_ids=STATIONS,
                missing_policy='preserve',columns={f:f for f in FIELDS},
                units=dict(zip(FIELDS,['m/s','degree','hundred_ft','tenths','K','%','hPa','code'])))


def _member(path,basename):
    with zipfile.ZipFile(path) as archive:
        matches=[n for n in archive.namelist() if n.replace('\\','/').split('/')[-1].casefold()==basename.casefold()]
        if len(matches)!=1:raise SchemaError('登记原始归档缺失或重复文件：'+basename)
        return archive.read(matches[0])


def _number(value):return float(value) if value.strip() else float('nan')
def _ceiling(value):return 999 if value=='---' else _number(value)
def _cover(value):return 10 if value=='-' else _number(value)
def _pcode(digits):
    if len(digits)!=5:raise SchemaError('CD144降水天气码长度错误。')
    return min([i*9+int(v) for i,v in enumerate(digits) if int(v)] or [0])


def decode_surface(archive):
    rows=[];sources=[]
    for name,station in zip(['BANGOR.144','BRUNSWK.144','CONCORD.144','BURLING.144','PORTLND.CDR'],STATIONS):
        raw=_member(archive,name);selected=[]
        for line in raw.decode('ascii').splitlines():
            samson=name.endswith('.CDR')
            if samson and (line.startswith('~') or not line.strip()):continue
            if samson:
                wanted=line[1:3]=='90' and int(line[4:6])==1 and int(line[7:9])==9 and 5<=int(line[10:12])<=8
            else:
                wanted=line[5:11]=='900109' and 5<=int(line[11:13])<=8
            if not wanted:continue
            if samson:
                stamp=datetime(1900+int(line[1:3]),int(line[4:6]),int(line[7:9]),tzinfo=ZONE)+timedelta(hours=int(line[10:12]))
                cover=line[53]
                if line[52]=='1':cover='-'
                elif line[52]=='9':cover=' '
                temp,rh,pressure,direction,speed=map(_number,[line[54:60],line[66:70],line[70:75],line[75:79],line[79:85]])
                if direction==999:speed=direction=float('nan')
                elif speed==0:direction=0
                if temp==9999:temp=float('nan')
                if rh==999:rh=float('nan')
                ceil=int(line[92:99])
                ceil=999 if ceil==77777 else 888 if ceil==88888 else float('nan') if ceil==99999 else int(ceil*3.2808399/100)
                digits=''.join(str((int(v)+1)%10) for v in line[101:106])
                code=float('nan') if int(line[13])==9 and not any(int(v) for v in digits) else _pcode(digits)
                values=[speed,direction,ceil,_cover(cover),temp+273.15,rh,pressure,code]
            else:
                if int(line[:5])!=station:raise SchemaError('CD144站点编号不匹配。')
                stamp=datetime(1900+int(line[5:7]),int(line[7:9]),int(line[9:11]),int(line[11:13]),tzinfo=ZONE)
                speed=_number(line[40:42]);direction=_number(line[38:40])*10
                if not math.isfinite(direction) or (speed>0 and direction<=0):speed=direction=float('nan')
                elif speed==0:direction=0
                code=_pcode(line[24:29]) if line[23:31].strip() else float('nan')
                values=[speed*.5144444444444445,direction,_ceiling(line[13:16]),_cover(line[78:79]),
                        (_number(line[46:49])-32)*5/9+273.15,_number(line[52:55]),_number(line[42:46])*.01*33.864,code]
            selected.append(line)
            rows.append(dict(station_id=station,time=stamp.isoformat(),**dict(zip(FIELDS,values))))
        if len(selected)!=4:raise SchemaError('原始地面观测未完整覆盖登记案例4小时：'+name)
        sources.append(dict(file=name,sha256=hashlib.sha256(raw).hexdigest(),selected_original_records=selected))
    frame=pd.DataFrame(rows).sort_values(['time','station_id'])
    if len(frame)!=20 or frame.duplicated(['time','station_id']).any():raise SchemaError('地面观测时站结构不完整。')
    return frame,surface_config(),sources


def decode_upper(archive,short):
    if short not in ('PWM','CHH'):raise SchemaError('仅支持登记的PWM和CHH原始探空。')
    raw=_member(archive,'90010100.'+short)
    wanted={datetime(1990,1,9,0,tzinfo=timezone.utc),datetime(1990,1,9,12,tzinfo=timezone.utc),datetime(1990,1,10,0,tzinfo=timezone.utc)}
    months={s:i+1 for i,s in enumerate(['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])}
    profiles={};current=None;metadata=None
    keys=['pressure','height','temperature','wind_direction','wind_speed']
    for line in raw.decode('ascii').splitlines():
        parts=line.split()
        if not parts:continue
        kind=int(parts[0])
        if kind==254:
            current=datetime(int(parts[4]),months[parts[3]],int(parts[2]),int(parts[1]),tzinfo=timezone.utc)
            if current in wanted:profiles[current]=[]
        elif current in wanted:
            if kind==1:
                meta=dict(id=int(parts[1]),name=short,latitude=float(parts[3]),longitude=-float(parts[4]),elevation_m=float(parts[5]))
                if metadata is not None and metadata!=meta:raise SchemaError('原始探空站点元数据变化。')
                metadata=meta
            if kind in (4,5,6,7,8,9):
                values=list(map(int,parts[1:]))
                if len(values)!=6:raise SchemaError('原始FSL层记录结构错误。')
                if values[0]==32767 or values[0]>=500:
                    profiles[current].append([values[0],values[1],values[2],values[4],values[5]])
    if set(profiles)!=wanted or metadata is None:raise SchemaError('探空未完整覆盖登记案例3个时刻。')
    rows=[]
    for stamp,levels in sorted(profiles.items()):
        for index,values in enumerate(levels,1):
            converted={key:float('nan') if val==32767 else val*(.1 if key in ('temperature','wind_speed') else 1) for key,val in zip(keys,values)}
            rows.append(dict(station_id=metadata['id'],time=stamp.isoformat(),level=index,**converted))
    config=dict(mode='upper_air',version='CALMET-5.8.5-UP-2.0',projection='NONE',height_reference='MSL',
                wind_direction_reference='meteorological_from_true_north',sounding_interval_hours=12,missing_values=[],stations=[metadata],
                columns={k:k for k in keys},units=dict(zip(keys,['hPa','m','degC','degree','m/s'])),missing_policy='preserve',top_pressure_hpa=500)
    return pd.DataFrame(rows),config,dict(file='90010100.'+short,sha256=hashlib.sha256(raw).hexdigest(),reported_levels=len(rows),
                scope='all reported levels at pressure>=500hPa; missing variables preserved; no vertical interpolation')


def decode_precipitation(archive):
    raw=_member(archive,'TD3240.DAT');by_station={s:[] for s in PRECIP_STATIONS}
    for line in raw.decode('ascii').splitlines():
        if not line.strip():continue
        if len(line)!=42:raise SchemaError('登记TD3240记录需42个字符。')
        station=int(line[3:9])
        if station not in by_station:continue
        if line[11:17]!='HPCPHI':raise SchemaError('仅支持登记TD3240小时降水百百分英寸编码。')
        hour=int(line[30:32])
        if not 1<=hour<=24:raise SchemaError('TD3240小时无效。')
        stamp=datetime(int(line[17:21]),int(line[21:23]),int(line[23:27]),tzinfo=ZONE)+timedelta(hours=hour)
        by_station[station].append(dict(time=stamp,value=int(line[34:40]),flag1=line[40],flag2=line[41],line=line))
    rows=[];evidence=[]
    normalize=lambda f:{'[':'M',']':'M','{':'D','}':'D'}.get(f,f)
    regular=lambda r:r['flag1'] in (' ','g') and r['flag2']==' ' and 0<=r['value']<99999
    for station,source in by_station.items():
        if not source or any(a['time']>=b['time'] for a,b in zip(source,source[1:])):raise SchemaError('TD3240站点缺失或时序重复。')
        closed={};i=0
        while i<len(source) and source[i]['time']<=max(TIMES):
            if source[i]['flag1'] in ('M','D','[',']','{','}'):
                if i+1>=len(source) or normalize(source[i]['flag1'])!=normalize(source[i+1]['flag1']):raise SchemaError('TD3240缺测区间未闭合。')
                if any(source[i]['time']<=t<=source[i+1]['time'] for t in TIMES):raise SchemaError('所选小时跨越TD3240缺测区间。')
                closed[i+1]=source[i];i+=2
            else:i+=1
        stamps=[r['time'] for r in source]
        for stamp in TIMES:
            index=bisect.bisect_left(stamps,stamp)
            if index<len(source) and source[index]['time']==stamp:
                record=source[index]
                if not regular(record):raise SchemaError('所选TD3240小时带有不支持的缺测/质量标记。')
                amount=record['value']*.254;basis='explicit hourly observation';support=[record]
            else:
                if not 0<index<len(source):raise SchemaError('所选TD3240小时没有双侧资料覆盖。')
                before,after=source[index-1],source[index]
                if not regular(after):raise SchemaError('TD3240后边界不能证明该小时无雨。')
                if index-1 in closed:
                    support=[closed[index-1],before,after]
                elif regular(before):support=[before,after]
                else:raise SchemaError('TD3240前边界不能证明该小时无雨。')
                amount=0.;basis='format-defined dry hour between valid bounds, outside paired missing intervals'
            rows.append(dict(station_id=station,time=stamp.isoformat(),hourly_mm=amount))
            evidence.append(dict(station=station,time=stamp.isoformat(),mm=amount,basis=basis,supporting_original_records=[r['line'] for r in support]))
    config=dict(mode='precipitation',version='CALMET-5.8.5-PRECIP-2.0',time_label='hour_end',timezone_west_hours=5,
                station_ids=PRECIP_STATIONS,fields=dict(station='station_id',time='time',precipitation='hourly_mm'),
                missing_values=[],source_units='mm',quantity='one_hour_amount',accumulation_seconds=3600)
    return pd.DataFrame(rows),config,dict(file='TD3240.DAT',sha256=hashlib.sha256(raw).hexdigest(),records=evidence,
                scope='16 stations, four selected dry hours; no positive rainfall validation by this case')
