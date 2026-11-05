"""Transport to the operator-installed, fingerprinted local Linux WRFDA backend."""
from pathlib import Path
import base64,json,os,shlex,socket,subprocess,time,uuid


class BackendUnavailable(RuntimeError):pass


def configuration():
    path=Path(os.environ.get('ECODATA_WRFDA_CONFIG',
        str(Path(os.environ.get('LOCALAPPDATA',Path.home()/'.local/share'))/'EcoDataEngine/wrfda_backend.json')))
    if not path.is_file():return None
    value=json.loads(path.read_text(encoding='utf-8'))
    if value.get('schema')!='ecodata.wrfda-backend.v1':raise BackendUnavailable('WRFDA后端配置版本不符')
    if value.get('host') not in {'127.0.0.1','localhost','::1'}:
        raise BackendUnavailable('当前安装仅支持本机隔离Linux后端，不会把输入自动上传外部服务器')
    return value


def connect(cfg):
    import paramiko
    client=paramiko.SSHClient()
    host=f"[{cfg['host']}]:{cfg['port']}"
    client.get_host_keys().add(host,'ssh-ed25519',paramiko.Ed25519Key(data=base64.b64decode(cfg['host_key'])))
    def attempt():
        client.connect(cfg['host'],port=cfg['port'],username=cfg['username'],key_filename=cfg['key_file'],
            allow_agent=False,look_for_keys=False,timeout=5,banner_timeout=10,auth_timeout=10)
    try:attempt()
    except (OSError,paramiko.SSHException):
        if not cfg.get('qemu_args'):raise BackendUnavailable('本机WRFDA Linux后端未启动或SSH身份不符')
        try:
            with socket.create_connection((cfg['host'],cfg['port']),timeout=2):pass
        except OSError:
            args=json.loads(Path(cfg['qemu_args']).read_text(encoding='utf-8'))
            root=Path(cfg['qemu']).parent.parent
            flags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'CREATE_NEW_PROCESS_GROUP',0)
            with (root/'autostart_stdout.log').open('ab') as out,(root/'autostart_stderr.log').open('ab') as err:
                proc=subprocess.Popen([cfg['qemu'],*args],cwd=root,stdout=out,stderr=err,creationflags=flags)
            (root/'qemu.pid').write_text(str(proc.pid))
        for _ in range(30):
            try:attempt();break
            except (OSError,paramiko.SSHException):time.sleep(2)
        else:raise BackendUnavailable('WRFDA隔离后端启动未就绪，保留启动日志，未执行同化')
    return client


def execute(request,folder,cfg,progress=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    client=connect(cfg);job=uuid.uuid4().hex;remote='/opt/ecodata/jobs/'+job
    started=time.monotonic()
    try:
        sftp=client.open_sftp()
        with sftp.open('/opt/ecodata/runtime.json') as stream:runtime=json.load(stream)
        if runtime.get('sha256')!=cfg['solver_sha256'] or runtime.get('commit')!=cfg['source_commit']:
            raise BackendUnavailable('WRFDA程序指纹或官方源码版本与已验收安装不符')
        _,out,err=client.exec_command('mkdir -p '+shlex.quote(remote))
        if out.channel.recv_exit_status()!=0:raise BackendUnavailable('WRFDA后端无法创建隔离任务目录')
        for native_name,path in request.items():
            if native_name not in {'fg','be.dat','ob.ascii','namelist.input'}:raise ValueError('未知WRFDA输入端口')
            sftp.put(str(path),remote+'/'+native_name)
        if progress:progress('正在运行官方WRFDA 3DVAR；输入质量标记由原生程序处理')
        _,out,err=client.exec_command('python3 /opt/ecodata/run_wrfda_job.py '+job)
        deadline=time.monotonic()+1900;last=0
        while not out.channel.exit_status_ready():
            if time.monotonic()>deadline:raise BackendUnavailable('WRFDA后端执行超时；任务日志保留，未交付分析场')
            if progress and time.monotonic()-last>15:
                progress('官方WRFDA 3DVAR仍在计算，尚未通过分析场验收');last=time.monotonic()
            time.sleep(.25)
        transport_code=out.channel.recv_exit_status()
        sftp.get(remote+'/native_execution.json',str(folder/'native_execution.json'))
        report=json.loads((folder/'native_execution.json').read_text(encoding='utf-8'))
        for item in report['outputs']:
            name=item['name']
            if Path(name).name!=name or name in {'.','..'}:raise ValueError('原生后端输出路径非法')
            sftp.get(remote+'/'+name,str(folder/name))
        report.update(transport_returncode=transport_code,host_elapsed_seconds=time.monotonic()-started)
        (folder/'native_execution.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        return report
    finally:client.close()
