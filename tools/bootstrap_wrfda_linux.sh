#!/usr/bin/env bash
# Build the unmodified official WRFDA 3DVAR solver inside an isolated Linux backend.
set -euo pipefail
mkdir -p /opt/ecodata
exec > >(tee -a /opt/ecodata/bootstrap.log) 2>&1
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends build-essential gfortran libnetcdf-dev libnetcdff-dev csh git m4 ca-certificates python3-pexpect
mkdir -p /opt/ecodata/netcdf
ln -sfn /usr/include /opt/ecodata/netcdf/include
ln -sfn /usr/lib/x86_64-linux-gnu /opt/ecodata/netcdf/lib
ln -sfn /usr/bin /opt/ecodata/netcdf/bin
export NETCDF=/opt/ecodata/netcdf
export WRFIO_NCD_LARGE_FILE_SUPPORT=1
cd /opt/ecodata
if [[ ! -d WRF/.git ]]; then
    git -c http.version=HTTP/1.1 clone --depth 1 --branch v4.7.1 https://github.com/wrf-model/WRF.git WRF
fi
cd WRF
test "$(git rev-parse HEAD)" = f52c197ed39d12e087d02c50f412d90d418f6186
git -c http.version=HTTP/1.1 submodule update --init --recursive
python3 - <<'PY'
import pexpect,re,sys
child=pexpect.spawn('./configure',['wrfda'],encoding='utf-8',timeout=120)
child.logfile=sys.stdout
child.expect('Enter selection')
text=child.before
hits=[int(m.group(1)) for m in re.finditer(r'(\d+)\.\s*\(serial\)[^\n]*(?:gfortran|GNU)',text,re.I)]
if len(hits)!=1:raise RuntimeError('No unique official GNU serial build option: '+repr(hits))
child.sendline(str(hits[0]))
while True:
    index=child.expect([pexpect.EOF,r'Compile for nesting',r'Enter selection'],timeout=120)
    if index==0:break
    child.sendline('0')
child.close()
if child.exitstatus!=0:raise RuntimeError('Official WRFDA configure failed')
PY
./compile -j 2 all_wrfvar > /opt/ecodata/compile.log 2>&1
test -x var/build/da_wrfvar.exe
python3 - <<'PY'
from pathlib import Path
import hashlib,json,subprocess
root=Path('/opt/ecodata/WRF');exe=root/'var/build/da_wrfvar.exe'
record=dict(schema='ecodata.wrfda-runtime.v1',version='4.7.1',source='https://github.com/wrf-model/WRF',
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 executable=str(exe),sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
 compiler=subprocess.check_output(['gfortran','--version'],text=True).splitlines()[0],
 scope='official serial 3DVAR; installation is not an assimilation success',source_modified=False)
Path('/opt/ecodata/runtime.json').write_text(json.dumps(record,indent=2))
PY
chown -R ecodata:ecodata /opt/ecodata
if [[ "${ECODATA_ISOLATED_VM:-0}" == 1 ]]; then touch /etc/cloud/cloud-init.disabled; fi
echo WRFDA_BUILD_READY
