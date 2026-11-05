"""Create local cloud-init media; keys remain outside the repository and logs."""
from pathlib import Path
import io,json,subprocess,hashlib,os,argparse
import pycdlib

parser=argparse.ArgumentParser(description='Prepare local isolated Linux backend from verified QEMU and Ubuntu downloads')
parser.add_argument('--runtime-dir',type=Path,default=Path(os.environ.get('LOCALAPPDATA',Path.home()))/'EcoDataEngine/wrfda-runtime')
args=parser.parse_args()
base=args.runtime_dir.resolve();repo=Path(__file__).resolve().parents[1]
base.mkdir(parents=True,exist_ok=True)
keys=base/'keys';keys.mkdir(exist_ok=True)
for name in ['client','host']:
    if not (keys/name).exists():
        subprocess.run(['ssh-keygen','-t','ed25519','-N','','-C','ecodata-local-wrfda','-f',str(keys/name)],check=True,capture_output=True)
image=base/'downloads/ubuntu-noble.img'
with image.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
assert sha=='bc9237bfdfa42b5f9a4404b6c6cab595d3e57599c6498f6664ad00e6192f9595'
disk=base/'wrfda-linux.qcow2'
if not disk.exists():
    subprocess.run([str(base/'qemu/qemu-img.exe'),'create','-f','qcow2','-F','qcow2','-b',str(image),str(disk),'24G'],check=True,capture_output=True)
config=dict(hostname='ecodata-wrfda',manage_etc_hosts=True,ssh_pwauth=False,disable_root=True,
 users=[dict(name='ecodata',groups=['sudo'],shell='/bin/bash',sudo='ALL=(ALL) NOPASSWD:ALL',lock_passwd=True,
             ssh_authorized_keys=[(keys/'client.pub').read_text().strip()])],
 ssh_keys=dict(ed25519_private=(keys/'host').read_text(),ed25519_public=(keys/'host.pub').read_text().strip()),
 write_files=[dict(path='/opt/ecodata/bootstrap.sh',permissions='0755',content=(repo/'tools/bootstrap_wrfda_linux.sh').read_text(encoding='utf-8'))],
 runcmd=[['env','ECODATA_ISOLATED_VM=1','bash','/opt/ecodata/bootstrap.sh']])
user=('#cloud-config\n'+json.dumps(config,indent=2)).encode()
meta=json.dumps({'instance-id':'ecodata-wrfda-local-v1','local-hostname':'ecodata-wrfda'}).encode()
seed=base/'seed.iso'
if not seed.exists():
    iso=pycdlib.PyCdlib();iso.new(interchange_level=3,joliet=3,rock_ridge='1.09',vol_ident='CIDATA')
    for filename,payload,iso_name in [('user-data',user,'USERDATA.;1'),('meta-data',meta,'METADATA.;1')]:
        iso.add_fp(io.BytesIO(payload),len(payload),iso_path='/'+iso_name,joliet_path='/'+filename,rr_name=filename)
    iso.write(str(seed));iso.close()
args=['-L',str(base/'qemu/share'),'-machine','q35','-accel','whpx','-cpu','qemu64,-svm','-m','4096','-smp','2',
 '-drive','file='+str(disk).replace('\\','/')+',if=virtio,format=qcow2',
 '-drive','file='+str(seed).replace('\\','/')+',media=cdrom,readonly=on',
 '-netdev','user,id=n0,hostfwd=tcp:127.0.0.1:22384-:22','-device','virtio-net-pci,netdev=n0,romfile=',
 '-display','none','-serial','file:'+str(base/'serial.log').replace('\\','/'),'-monitor','none',
 '-smbios','type=1,serial=ds=nocloud','-qmp','tcp:127.0.0.1:22385,server=on,wait=off']
(base/'qemu_args.json').write_text(json.dumps(args,indent=2),encoding='utf-8')
public=(keys/'host.pub').read_text().split()
(base/'connection.json').write_text(json.dumps(dict(host='127.0.0.1',port=22384,username='ecodata',
 key_file=str(keys/'client'),host_key_type=public[0],host_key=public[1],runtime_root='/opt/ecodata'),indent=2),encoding='utf-8')
print('Created isolated Linux overlay and seed; private key material was not printed')
