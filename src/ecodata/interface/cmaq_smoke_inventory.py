"""Audited EPA NEI oil/gas sector -> six native SMOKE steps, fixed LISTOS case.

This does not claim all emission sectors or arbitrary inventories are supported.
Uploaded data are never executed. Reviewed local environments are hash-pinned.
"""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
from netCDF4 import Dataset


def contract(sector='oilgas'):
    names={'oilgas':'cmaq_smoke_listos.json','non_egu':'cmaq_smoke_non_egu.json','cmv':'cmaq_smoke_cmv.json'}
    if sector not in names:raise ValueError('未知已核验排放部门')
    return json.loads(Path(__file__).with_name(names[sector]).read_text(encoding='utf-8'))


def validate(payload,sector='oilgas'):
    if not isinstance(payload,bytes) or hashlib.sha256(payload).hexdigest()!=contract(sector)['raw_sha256']:
        raise ValueError('当前SMOKE入口只验收所选部门已核验的2018NEI东北子集；不将其他清单替换为登记数据')


def validate_hourly(payload,sector='cmv'):
    expected=contract(sector).get('hourly_raw_sha256')
    if expected is None or not isinstance(payload,bytes) or hashlib.sha256(payload).hexdigest()!=expected:
        raise ValueError('船舶小时清单必须与已核验年度子集匹配；不补缺小时，不回退登记小时值')


def build(folder,payload,evidence_root,sector='oilgas',hourly_payload=None):
    validate(payload,sector)
    if contract(sector).get('hourly_raw_sha256'):validate_hourly(hourly_payload,sector)
    elif hourly_payload is not None:raise ValueError('所选部门不接受此小时清单')
    from ecodata.interface.independent_cases import linux
    folder=Path(folder);base=Path(evidence_root);c=contract(sector);reference=base/c['reference']
    folder.mkdir(parents=True,exist_ok=False)
    raw=folder/'raw_inventory.csv';raw.write_bytes(payload)
    mapping={linux(reference):linux(folder),linux(base/c['raw']):linux(raw)}
    if hourly_payload is not None:
        hourly=folder/'raw_hourly.csv';hourly.write_bytes(hourly_payload)
        mapping[linux(base/c['hourly_raw'])]=linux(hourly)
        (folder/'pthour_files.txt').write_text('#LIST\n'+linux(hourly)+'\n',encoding='utf-8')
    for step in c['steps']:
        if step['subdir']!='.':
            (folder/step['name']).mkdir()
            mapping[linux(reference/step['subdir'])]=linux(folder/step['name'])
    (folder/(c.get('inventory_name','northeast_oilgas2018')+'_dat')).mkdir()
    report=dict(raw_sha256=c['raw_sha256'],steps=[],full_regional_inventory=False,
                sector=sector,scope=c.get('scope','2018 NEI northeast oil/gas point sector; original audited temporal/speciation/gridding settings. Other sectors retained separately; not all HAP species modeled.'))
    if hourly_payload is not None:report['hourly_raw_sha256']=c['hourly_raw_sha256']
    for step in c['steps']:
        old=reference/step['subdir'];target=folder if step['subdir']=='.' else folder/step['name']
        cfgpath=old/'environment.json';exe=base/'tools/smoke_official'/step['exe']
        if hashlib.sha256(cfgpath.read_bytes()).hexdigest()!=step['environment_sha256']:raise ValueError('SMOKE登记环境已变：'+step['name'])
        if hashlib.sha256(exe.read_bytes()).hexdigest()!=step['exe_sha256']:raise ValueError('SMOKE程序已变：'+step['name'])
        cfg=json.loads(cfgpath.read_text(encoding='utf8'))
        for key,value in cfg.items():
            for src,dst in sorted(mapping.items(),key=lambda x:-len(x[0])):value=value.replace(src,dst)
            cfg[key]=value
        if step['name']=='temporal':
            (target/'procdates.txt').write_bytes((old/'procdates.txt').read_bytes())
        (target/'environment.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
        with (target/'console.log').open('wb') as log:
            p=subprocess.run(['wsl.exe','-d','EcoData-WRF-Ubuntu2204','-u','root','--','env',*[k+'='+v for k,v in cfg.items()],linux(exe)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        report['steps'].append(dict(name=step['name'],returncode=p.returncode,engine_sha256=step['exe_sha256']))
        (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        if p.returncode:raise ValueError('SMOKE失败：'+step['name']+'，保留日志')
    report['products']={}
    for label,path in [('stack_groups',folder/'elevpoint/stack_groups.ncf'),('inline',folder/'merge/inline.ncf')]:
        ref=reference/next(s['subdir'] for s in c['steps'] if s['name']==('elevpoint' if label=='stack_groups' else 'merge'))/path.name
        with Dataset('new',memory=path.read_bytes()) as d,Dataset('audited',memory=ref.read_bytes()) as r:
            if set(d.variables)!=set(r.variables):raise ValueError('SMOKE产品物种/结构与已审计案例不符')
            for name,var in d.variables.items():
                a=np.ma.asarray(var[:])
                if np.ma.getmaskarray(a).any() or not np.isfinite(a).all():raise ValueError('SMOKE缺测或非有限值：'+name)
                if not np.array_equal(a,r[name][:]):raise ValueError('同一清单重建后科学值不一致：'+name)
        report['products'][label]=dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    report['success']=True
    (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return report
