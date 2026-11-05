"""Trusted Linux-side runner for an isolated official WRFDA job directory."""
from pathlib import Path
import hashlib,json,re,shutil,subprocess,sys,time


def run(job_id):
    if not re.fullmatch('[a-f0-9]{32}',job_id):raise ValueError('Invalid job identity')
    base=Path('/opt/ecodata');job=base/'jobs'/job_id
    if not job.is_dir():raise ValueError('Prepared native job does not exist')
    runtime=json.loads((base/'runtime.json').read_text())
    executable=Path(runtime['executable'])
    if hashlib.sha256(executable.read_bytes()).hexdigest()!=runtime['sha256']:
        raise ValueError('Installed solver fingerprint changed')
    inputs={'fg','be.dat','ob.ascii','namelist.input'}
    for name in inputs:
        if not (job/name).is_file() or (job/name).is_symlink():raise ValueError('Missing or linked scientific input: '+name)
    for table_dir in [base/'WRF/run',base/'WRF/var/run']:
        for table in table_dir.iterdir():
            if table.is_file() and table.name not in inputs and table.suffix not in {'.exe','.sh','.csh','.pl','.py'}:
                shutil.copyfile(table,job/table.name)
    before={p.name for p in job.iterdir()};started=time.monotonic()
    try:
        with (job/'wrfda.log').open('wb') as log:
            result=subprocess.run([str(executable)],cwd=job,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        code=result.returncode;timed_out=False
    except subprocess.TimeoutExpired:
        code=None;timed_out=True
    outputs=[dict(name=p.name,bytes=p.stat().st_size) for p in sorted(job.iterdir())
             if p.is_file() and not p.is_symlink() and p.name not in before]
    report=dict(schema='ecodata.wrfda-execution.v1',job_id=job_id,runtime=runtime,returncode=code,
                timed_out=timed_out,seconds=time.monotonic()-started,outputs=outputs,
                native_program_executed=True,wrf_forecast_executed=False)
    (job/'native_execution.json').write_text(json.dumps(report,indent=2))
    return 0 if code==0 else 1


if __name__=='__main__':sys.exit(run(sys.argv[1]))
