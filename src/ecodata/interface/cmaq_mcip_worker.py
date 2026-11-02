"""Standard-library worker for fixed installed MCIP; never execute uploaded code."""
from pathlib import Path
import sys,json,hashlib,subprocess,os,re

if __name__=='__main__':
    c=json.loads(Path(sys.argv[1]).read_text());folder=Path(c['folder']);exe=Path(c['exe'])
    if hashlib.sha256(exe.read_bytes()).hexdigest()!=c['pin']:raise ValueError('MCIP hash mismatch')
    (folder/'fort.8').symlink_to(folder/'namelist.mcip');(folder/'fort.4').symlink_to(folder/'GRIDDESC')
    for i,path in enumerate(c['wrf']):(folder/f'fort.{10+i}').symlink_to(path)
    env=dict(os.environ,IOAPI_CHECK_HEADERS='T',EXECUTION_ID='UPLOADED_LISTOS_WRF_MCID',IOAPI_ISPH='20',IOAPI_LOG_WRITE='F')
    names={'GRID_BDY_2D':'GRIDBDY2D','GRID_CRO_2D':'GRIDCRO2D','GRID_DOT_2D':'GRIDDOT2D','MET_BDY_3D':'METBDY3D','MET_CRO_2D':'METCRO2D','MET_CRO_3D':'METCRO3D','MET_DOT_3D':'METDOT3D','LUFRAC_CRO':'LUFRAC_CRO','SOI_CRO':'SOI_CRO','MOSAIC_CRO':'MOSAIC_CRO'}
    for key,name in names.items():env[key]=str(folder/(name+'.nc'))
    with (folder/'console.log').open('wb') as log:p=subprocess.run([str(exe)],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=180)
    text=(folder/'console.log').read_text(errors='replace')
    result=dict(returncode=p.returncode,normal_termination=bool(re.search(r'(?m)^\s*(?:\*+\s*)?NORMAL TERMINATION\b',text)),executable_sha256=c['pin'])
    (folder/'native_status.json').write_text(json.dumps(result,indent=2))
    sys.exit(0 if p.returncode==0 and result['normal_termination'] else 1)
