"""Register a built local Linux runtime, checking its pinned SSH and solver identity."""
from pathlib import Path
import argparse,json,os,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ecodata.models.wrfda_backend import connect

parser=argparse.ArgumentParser()
parser.add_argument('connection',type=Path)
args=parser.parse_args()
cfg=json.loads(args.connection.read_text(encoding='utf-8'))
if cfg['host'] not in {'localhost','127.0.0.1','::1'}:raise ValueError('Only local Linux backends are enabled')
client=connect(cfg)
try:
    sftp=client.open_sftp()
    with sftp.open('/opt/ecodata/runtime.json') as stream:runtime=json.load(stream)
    if runtime['commit']!='f52c197ed39d12e087d02c50f412d90d418f6186':raise ValueError('Unvalidated official source version')
    sftp.put(str(Path(__file__).with_name('run_wrfda_job.py')),'/opt/ecodata/run_wrfda_job.py')
finally:client.close()
cfg.update(schema='ecodata.wrfda-backend.v1',solver_sha256=runtime['sha256'],source_commit=runtime['commit'])
base=args.connection.resolve().parent
if (base/'qemu_args.json').is_file():cfg.update(qemu=str(base/'qemu/qemu-system-x86_64.exe'),qemu_args=str(base/'qemu_args.json'))
destination=Path(os.environ.get('LOCALAPPDATA',Path.home()/'.local/share'))/'EcoDataEngine/wrfda_backend.json'
destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
print('Installed WRFDA backend registered:',destination)
