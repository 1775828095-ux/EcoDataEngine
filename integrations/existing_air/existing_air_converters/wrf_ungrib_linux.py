"""Controlled existing official ungrib.exe, invoked by the platform converter."""
from pathlib import Path
from datetime import datetime,timedelta
import hashlib,json,subprocess,sys,tempfile,shutil

ENGINE=Path('/opt/ecodata-wrf/WPS-4.7.0/ungrib/src/ungrib.exe')
VTABLE=Path('/opt/ecodata-wrf/WPS-4.7.0/ungrib/Variable_Tables/Vtable.GFS')
PINS={'engine':'7fc3831efcc8f38c9e896aeed5c83382164445aeab0a4d20a301ab0fb0276ec7',
      'vtable':'1d196a00bf15bc01c87c4897fd2817b1a046a87ec156701f0d15430a7af1ac4a'}

def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):value.update(block)
    return value.hexdigest()

def main(path):
    job=json.loads(Path(path).read_text());dest=Path(job['output']);dest.mkdir(exist_ok=True)
    if digest(ENGINE)!=PINS['engine'] or digest(VTABLE)!=PINS['vtable']:raise ValueError('Installed WPS decoder or Vtable hash changed')
    for item in job['inputs']:
        if digest(Path(item['path']))!=item['sha256']:raise ValueError('Input GRIB hash mismatch')
    with tempfile.TemporaryDirectory(prefix='ecodata_ungrib_') as temp:
        run=Path(temp)
        shutil.copyfile(VTABLE,run/'Vtable')
        for i,item in enumerate(job['inputs']):
            if i>=26**3:raise ValueError('Too many GRIB files')
            code=''.join(chr(65+(i//power)%26) for power in (676,26,1))
            (run/('GRIBFILE.'+code)).symlink_to(Path(item['path']))
        text="&share\n wrf_core='ARW', max_dom=1,\n start_date='"+job['start']+"',\n end_date='"+job['end']+"',\n interval_seconds="+str(job['interval_seconds'])+",\n/\n&ungrib\n out_format='WPS', prefix='FILE',\n/\n"
        (run/'namelist.wps').write_text(text)
        with (dest/'ungrib.log').open('wb') as log:
            p=subprocess.run([str(ENGINE)],cwd=run,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        log=(dest/'ungrib.log').read_text(errors='replace')
        if p.returncode or 'Successful completion' not in log:raise ValueError('Official ungrib failed: '+log[-2000:])
        expected=[];t=datetime.strptime(job['start'],'%Y-%m-%d_%H:%M:%S');end=datetime.strptime(job['end'],'%Y-%m-%d_%H:%M:%S')
        while t<=end:expected.append('FILE:'+t.strftime('%Y-%m-%d_%H'));t+=timedelta(seconds=job['interval_seconds'])
        actual=sorted(p.name for p in run.glob('FILE:*'))
        if actual!=expected:raise ValueError('GRIB coverage does not match supplied engineering dates: '+str(actual))
        files=[]
        for name in actual:
            target=dest/(name.replace(':','_')+'.wps');shutil.copyfile(run/name,target)
            files.append(dict(path=target.name,sha256=digest(target),bytes=target.stat().st_size))
        result=dict(returncode=p.returncode,engine=str(ENGINE),engine_sha256=PINS['engine'],vtable=str(VTABLE),vtable_sha256=PINS['vtable'],
            start=job['start'],end=job['end'],interval_seconds=job['interval_seconds'],command=[str(ENGINE)],files=files,
            scope='Original supplied FNL -> official WPS intermediate atmospheric fields; no model integration or science setting changes')
        (dest/'ungrib_receipt.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result))

if __name__=='__main__':main(sys.argv[1])
