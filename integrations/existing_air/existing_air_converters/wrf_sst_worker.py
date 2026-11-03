"""WPS SST supplement: preserve dates and unknowns; no WRF model execution."""
from pathlib import Path
import hashlib, io, json, struct, sys, zipfile, subprocess
import numpy as np
import pandas as pd
import xarray as xr
from ecodata.models.adapters.wps_sst import _text, _record, export_sst
from ecodata.interface.wrf_assembly import read_fields


def linux_path(path):
    p=Path(path).resolve()
    if len(p.drive)!=2 or p.drive[1]!=':':raise ValueError('WSL路径需要本机盘符绝对路径')
    return '/mnt/'+p.drive[0].lower()+p.as_posix()[2:]


def forcing(job,out):
    deps=job.get('dependencies',[]);gribs=[d for d in deps if d['role']=='raw_atmospheric_forcing']
    if not gribs:return None
    controls=[d for d in deps if d['role']=='engineering_configuration']
    if len(controls)!=1:raise ValueError('原始FNL转换需要唯一的工程namelist日期与间隔')
    from ecodata.interface.wrf_external_project import _groups
    groups=_groups(Path(controls[0]['path']).read_bytes())
    tc=groups.get('time_control',{})
    def stamp(prefix):
        values=[]
        for key in ('year','month','day','hour','minute','second'):
            tokens=[x.strip() for x in tc.get(prefix+'_'+key,'0' if key in ('minute','second') else '').split(',') if x.strip()]
            if not tokens or len(set(tokens))!=1:raise ValueError('嵌套域的日期不同或缺失，不能猜测共同FNL转换时段')
            values.append(int(tokens[0]))
        from datetime import datetime
        return datetime(*values).strftime('%Y-%m-%d_%H:%M:%S')
    interval=int(tc.get('interval_seconds','0'))
    if not 0<interval<=86400:raise ValueError('驱动资料间隔缺失或不在支持范围')
    target=out/'atmosphere';target.mkdir()
    request=dict(start=stamp('start'),end=stamp('end'),interval_seconds=interval,
        inputs=[dict(path=linux_path(d['path']),sha256=d['sha256']) for d in gribs],output=linux_path(target))
    path=out/'ungrib_request.json';path.write_text(json.dumps(request),encoding='utf8')
    command=['wsl.exe','--exec','python3',linux_path(Path(__file__).with_name('wrf_ungrib_linux.py')),linux_path(path)]
    with (out/'ungrib_dispatch.log').open('wb') as log:
        p=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=660,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
    if p.returncode:raise ValueError('官方WPS解码失败：'+(out/'ungrib_dispatch.log').read_text(encoding='utf8',errors='replace')[-2000:])
    report=json.loads((target/'ungrib_receipt.json').read_text(encoding='utf8'))
    checks=[]
    for record in report['files']:
        path=target/record['path'];raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=record['sha256']:raise ValueError('大气转换文件哈希变化')
        fields=read_fields(raw)
        if fields[0].time[:13] != record['path'][5:-4]:raise ValueError('大气文件名与内部时间不一致')
        present={f.key[1] for f in fields}
        if not {'TT','UU','VV','GHT','RH','PSFC'}<=present:raise ValueError('大气驱动缺少必要配套物理场：'+str(present))
        if any(not np.isfinite(np.frombuffer(f.parts[4],dtype='>f4')).all() for f in fields):raise ValueError('WPS场含无穷或非有限编码')
        checks.append(dict(file=record['path'],time=fields[0].time,record_count=len(fields),fields=sorted(present),sha256=record['sha256']))
    report['independent_reader_checks']=checks
    report['input_dependencies']=deps
    return report


def main(request):
    job=json.loads(Path(request).read_text(encoding='utf8'));out=Path(job['output'])
    payload=Path(job['product']).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=job['product_sha256']:raise ValueError('处理产品哈希不一致')
    with xr.open_dataset(io.BytesIO(payload)) as source: ds=source.load()
    if ds.sst.attrs.get('units')!='K' or ds.attrs.get('time_basis')!='UTC':raise ValueError('SST单位/时区不符')
    a=ds.sst.transpose('time','lat','lon');times=pd.DatetimeIndex(ds.time.values)
    for axis in ('lat','lon'):
        v=ds[axis].values
        if v.ndim!=1 or len(v)<2 or not np.isfinite(v).all() or not np.all(np.diff(v)>0) or not np.allclose(np.diff(v),np.diff(v)[0],rtol=0,atol=1e-6):
            raise ValueError('WPS SST经纬度必须单调、规则且至少两点')
    if times.hasnans or not times.is_unique or not times.is_monotonic_increasing:raise ValueError('SST日期无效')
    if len(times)>1 and not np.all(np.diff(times.asi8)==86400*10**9):raise ValueError('非连续逐日分析，不自动插值')
    reports=[];native=out/'native';native.mkdir()
    for index,t in enumerate(times):
        data=a.isel(time=index).values.astype('float64');known=np.isfinite(data)
        if not known.any() or np.isinf(data).any() or np.any((data[known]<250)|(data[known]>330)):raise ValueError('SST有效范围失败')
        if t.minute or t.second or t.microsecond or t.nanosecond:raise ValueError('WPS命名不支持非整点分析')
        if known.all():
            files,_=export_sst(ds.isel(time=[index]),dict(version='WPS-4.6-intermediate-v5-SST',time_basis='UTC',crs='EPSG:4326'))
            raw=next(value for name,value in files.items() if name.endswith('.wps'))
        else:
            # Same legacy WPS record helpers. -1e30 is the native METGRID.TBL
            # missing_value for SST; it is a mask, never a manufactured temperature.
            ny,nx=data.shape
            header=(_text(t.strftime('%Y-%m-%d_%H:%M:%S'),24)+struct.pack('>f',0)+_text('NOAA OISST daily analysis',32)+_text('SST',9)+_text('K',25)+_text('Daily SST analysis; original missing mask',46)+struct.pack('>fiii',200100.,nx,ny,0))
            projection=_text('SWCORNER',8)+struct.pack('>5f',float(ds.lat[0]),float(ds.lon[0]),float(ds.lat[1]-ds.lat[0]),float(ds.lon[1]-ds.lon[0]),6367.47)
            slab=np.where(known,data,-1e30).astype('>f4').tobytes()
            raw=b''.join(_record(p) for p in (struct.pack('>i',5),header,projection,struct.pack('>i',0),slab))
        # Independent existing reader, not the writer's own return flag.
        fields=read_fields(raw)
        if len(fields)!=1:raise ValueError('WPS SST应每时刻一条记录')
        field=fields[0];back=np.frombuffer(field.parts[4],dtype='>f4').reshape(data.shape)
        if field.name!='SST' or field.unit!='K' or field.time!=t.strftime('%Y-%m-%d_%H:%M:%S'):raise ValueError('WPS时间/变量/单位读回错误')
        expected_grid=[float(ds.lat[0]),float(ds.lon[0]),float(ds.lat[1]-ds.lat[0]),float(ds.lon[1]-ds.lon[0])]
        if not np.allclose(field.grid[:4],expected_grid,rtol=0,atol=1e-6) or field.level!=200100.:
            raise ValueError('WPS海温空间网格或层次读回不一致')
        if not np.array_equal(back<-1e29,~known):raise ValueError('WPS缺失掩码不一致')
        error=float(np.max(np.abs(back[known].astype(float)-data[known])))
        if error>3e-5:raise ValueError('WPS浮点编码误差超限')
        name='SST_'+t.strftime('%Y-%m-%d_%H')+'.wps';(native/name).write_bytes(raw)
        reports.append(dict(file=name,time=field.time,shape=list(back.shape),valid_count=int(known.sum()),missing_count=int((~known).sum()),max_encoding_error_K=error,sha256=hashlib.sha256(raw).hexdigest()))
    note=('WPS intermediate-v5 daily SST SUPPLEMENT ONLY.\n'
        'Original daily analysis dates and missing masks are preserved. No temporal interpolation or constant snapshot.\n'
        'Rename SST_YYYY-MM-DD_HH.wps to SST:YYYY-MM-DD_HH on a WPS host.\n'
        'SST missing_value=-1.E30 is used; no zeros or land/lake temperatures were invented.\n'
        'Do not rely on METGRID.TBL fill_missing=0 to replace unknown ocean temperatures.\n'
        'Subdaily forcing, missing coastal coverage, lake temperatures and updating wrfinput require an explicit reviewed policy.\n'
        'This export does NOT update the supplied WRF project, does not provide full model input, and does not run WRF.\n')
    (native/'README_SST.txt').write_text(note,encoding='ascii')
    atmosphere=forcing(job,out)
    validation=dict(scope='daily SST supplement and supplied FNL atmospheric WPS conversion',source_product=job['product'],source_product_sha256=job['product_sha256'],
        records=reports,independent_reader='ecodata.interface.wrf_assembly.read_fields',
        missing_value_policy='preserve original unknowns as native -1e30; never zero fill',
        atmosphere=atmosphere,native_model_executed=False,full_model_ready=False,engineering_updated=False,accuracy_evaluated=False)
    (out/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf8')
    with zipfile.ZipFile(out/'model-inputs.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in native.iterdir():z.write(p,'native/'+p.name)
        if atmosphere:
            for p in (out/'atmosphere').iterdir():
                if p.is_file():z.write(p,'atmosphere/'+p.name)
    result=dict(status='PASS',files=['model-inputs.zip','validation.json','conversion.log'],validation=validation)
    (out/'converter_result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('Daily SST native conversion and independent source/mask/time readback passed',flush=True)


if __name__=='__main__':main(sys.argv[1])
