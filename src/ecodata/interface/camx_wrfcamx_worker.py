"""Only a trusted pinned local WRFCAMx binary, fixed explicit control."""
from pathlib import Path
import sys,json,subprocess,hashlib
c=json.loads(Path(sys.argv[1]).read_text());folder=Path(c['linux_folder']);exe=Path('/mnt/d/EcoDataEngine/work/camx_tools_20261021/wrfcamx_v5.2/src/wrfcamx')
assert hashlib.sha256(exe.read_bytes()).hexdigest()=='3c3b136871f7185bc345ed2f71b9a1dcb5b0d524f62378e817be5f706e9ed252'
control='\n'.join(['Run note           |Independent2016 GFS; original same-case snow-age initial state','CAMx nested grid   |F','Diagnostic fields  |T','Seaice adjustment  |F','KV Method          |MYJ','Minimum Kv         |0.1','Project            |LAMBERT','Subgrid convecton  |DIAG','Subgrid stratiform |T','Start/end date     |16061000 16061006','WRF output freq    |60','Grid time zone     |0','CAMx grid size     |158, 91, 20','CAMx Grid spacing  |36. 36.','CAMx orig & params |-2556., -1872., -97., 40., 33., 45.','Layer mapping      |'+','.join(map(str,c['layer_mapping'])),f'CAMx LU file       |{folder}/landuse.nc',f'CAMx 3D file       |{folder}/met3d.nc',f'CAMx 2D file       |{folder}/met2d.nc',f'CAMx Kv file       |{folder}/kv.nc',f'Input snow age     |{folder}/snow_age_initial.bin','Initial snow age   |','Output snow age    |',f'WRF filename       |{c["linux_wrf"]}',''])
(folder/'control.txt').write_text(control)
with (folder/'console.log').open('wb') as log:r=subprocess.run([str(exe)],cwd=folder,input=control.encode(),stdout=log,stderr=subprocess.STDOUT,timeout=300)
text=(folder/'console.log').read_text(errors='replace');result=dict(returncode=r.returncode,finished='RUN FINISHED' in text,actual_snow_age_file_read='Reading input snow age file' in text,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
(folder/'native_result.json').write_text(json.dumps(result,indent=2));print(result,flush=True);assert r.returncode==0 and result['finished'] and result['actual_snow_age_file_read']
