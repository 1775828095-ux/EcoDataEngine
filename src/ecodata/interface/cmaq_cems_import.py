"""Native CEMS import stage. Not a complete CCTM new-build adapter.

Retains all original source bytes. Uses reviewed SMOKE configuration/executable;
does not interpolate missing hours or rewrite missing-value sentinels.
"""
from pathlib import Path
import hashlib,json,subprocess,csv,math

def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def check_point_requirements(annual):
    """List missing source fields separately from unsupported release types."""
    missing=[];unsupported=[];seen=set()
    with Path(annual).open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(line for line in f if not line.startswith('#')):
            identity=tuple(row.get(k,'') for k in ('region_cd','facility_id','unit_id','rel_point_id','process_id'))
            if identity in seen:continue
            seen.add(identity)
            item={'source':dict(zip(('region','facility','unit','release','process'),identity))}
            kind=row.get('erptype','').strip()
            if kind!='2':
                unsupported.append(dict(item,release_type=kind,reason='当前烟囱排放路线不支持此类来源；请提供已由相应工具处理好的完整工程，或等待该类型支持。不是缺信息，不能通过补烟囱参数代替。'))
                continue
            fields=[]
            for key,label in [('longitude','经度'),('latitude','纬度'),('stkhgt','排放口高度'),('stkdiam','排放口直径'),('stktemp','排出气体温度'),('stkvel','排出速度')]:
                try:
                    v=float(row.get(key,''))
                    valid=math.isfinite(v)
                    if key in ('stkhgt','stkdiam','stkvel'):valid=valid and v>0
                    elif key=='stktemp':valid=valid and v>-459.67
                    elif key=='longitude':valid=valid and -180<=v<=180
                    elif key=='latitude':valid=valid and -90<=v<=90
                except (ValueError,TypeError):valid=False
                if not valid:fields.append(label+'（'+key+'）')
            if fields:missing.append(dict(item,required=fields,action='请补充或更正原始资料后重新提交；平台不自动填值。'))
    return {'source_count':len(seen),'missing_information':missing,'unsupported_sources':unsupported,'ready_for_point_route':bool(seen) and not missing and not unsupported,'scope':'只检查点源必要字段与释放类型；通过后仍需年度/小时对应、日期、单位与完整工程检查。'}

def import_original(folder, annual, hourly, summary, evidence_root, *, diagnostic_only=False):
    from ecodata.interface.cmaq_smoke_inventory import contract
    from ecodata.interface.independent_cases import linux
    folder=Path(folder);base=Path(evidence_root);c=contract('cmv')
    step=c['steps'][0];reference=base/c['reference']
    cfgpath=reference/'environment.json';exe=base/'tools/smoke_official'/step['exe']
    if _sha(cfgpath)!=step['environment_sha256'] or _sha(exe)!=step['exe_sha256']:
        raise ValueError('Reviewed native configuration or engine changed')
    annual=Path(annual);summary=Path(summary);hourly=list(map(Path,hourly))
    if not hourly:raise ValueError('Original hourly CEMS inputs required')
    head=annual.read_bytes()[:4096]
    if b'#FORMAT=FF10_POINT' not in head or b'#YEAR=2018' not in head:
        raise ValueError('This stage requires original 2018 FF10 point inventory')
    requirements=check_point_requirements(annual)
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'input_requirements.json').write_text(json.dumps(requirements,ensure_ascii=False,indent=2),encoding='utf-8')
    if not diagnostic_only and not requirements['ready_for_point_route']:
        raise ValueError('资料检查未通过：缺必要信息的来源 '+str(len(requirements['missing_information']))+' 个；当前路线不支持的来源 '+str(len(requirements['unsupported_sources']))+' 个。详见input_requirements.json；未启动计算。')
    report={'stage':'native_CEMS_inventory_import_only','success':False,
            'full_raw_construction':False,'inputs':[],'diagnostic_only':diagnostic_only,
            'science_policy':'Original values unchanged. Native CEM allocation and missing-value branches must be audited before outputs can be integrated. No CCTM acceptance implied.'}
    saved=[]
    for i,p in enumerate([annual,summary]+hourly):
        target=folder/('original_'+str(i)+p.suffix);target.write_bytes(p.read_bytes());saved.append(target)
        report['inputs'].append({'source':str(p),'saved':str(target),'bytes':target.stat().st_size,'sha256':_sha(target)})
    name='original_egucems2018'
    mapping={linux(reference):linux(folder),'northeast_cmv2018':name}
    env=json.loads(cfgpath.read_text(encoding='utf-8'))
    for k,v in env.items():
        for a,b in sorted(mapping.items(),key=lambda x:-len(x[0])):v=v.replace(a,b)
        env[k]=v
    (folder/(name+'_dat')).mkdir()
    listing=folder/'pthour_files.txt'
    listing.write_text('#LIST CEM\n'+'\n'.join(linux(p) for p in saved[2:])+'\n',encoding='utf-8')
    # Same reviewed native type-1 handling as the non-EGU fugitive trial.
    # Zero stack fields encode "not a chimney" here, not a physical release
    # height of zero. Fugitive geometry still needs its own model adapter.
    env.update(PTINV=linux(saved[0]),CEMSUM=linux(saved[1]),PTHOUR=linux(listing),HOUR_SPECIFIC_YN='Y',PHOUR=linux(folder/'hour_inventory.ncf'),FLOW_RATE_FACTOR='0',OUTPUT_LOCAL_TIME='N',HOURLY_TO_PROFILE='N',HOURLY_TO_DAILY='N',OUTZONE='0',NO_STACK_REPLACE_FUGITIVE='Y')
    report['explicit_processing_choices']={'NO_STACK_REPLACE_FUGITIVE':'Y','basis':'Reviewed fixstk.f skips type-1 chimney replacement; non-EGU fugitive trial uses same option','limitation':'Zero chimney fields are non-chimney placeholders. Not an implementation of physical fugitive height/width/length; integration remains unapproved.'}
    (folder/'environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
    (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    try:
        with (folder/'console.log').open('wb') as log:
            p=subprocess.run(['wsl.exe','-d','EcoData-WRF-Ubuntu2204','-u','root','--','env',*[k+'='+v for k,v in env.items()],linux(exe)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    except Exception as exc:
        report.update(error=type(exc).__name__+': '+str(exc),integration_approved=False)
        (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        raise
    report.update(returncode=p.returncode,engine_sha256=_sha(exe))
    report['success']=p.returncode==0 and (folder/'hour_inventory.ncf').exists()
    report['integration_approved']=False
    (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

def process_imported(folder,evidence_root):
    """Continue reviewed fixed-LISTOS preprocessing; no CCTM substitution."""
    from ecodata.interface.cmaq_smoke_inventory import contract
    from ecodata.interface.independent_cases import linux
    folder=Path(folder);base=Path(evidence_root);c=contract('cmv');reference=base/c['reference']
    imported=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    if not imported['success']:raise ValueError('Native hourly import has not succeeded')
    downloads=json.loads((base/'sources/CMAQ_additional_sector_downloads.json').read_text(encoding='utf-8'))
    tables={}
    for label,prefix in [('GSPRO','gspro_ptegu_'),('GSREF','gsref_ptegu_')]:
        item=next(x for x in downloads if Path(x['file']).name.startswith(prefix));p=Path(item['file'])
        if _sha(p)!=item['sha256']:raise ValueError('EGU speciation reference changed')
        tables[label]=linux(p)
    report={'success':False,'integration_approved':False,'steps':[],
            'scope':'Original EGU CEMS inventory, fixed LISTOS grid and audited EGU speciation tables. Native missing/stack changes require separate audit; no full regional or CCTM claim.'}
    mapping={linux(reference):linux(folder),'northeast_cmv2018':'original_egucems2018'}
    for s in c['steps'][1:]:
        old=reference/s['subdir'];target=folder/s['name'];target.mkdir(exist_ok=False)
        if _sha(old/'environment.json')!=s['environment_sha256']:raise ValueError('Reviewed configuration changed')
        exe=base/'tools/smoke_official'/s['exe']
        if _sha(exe)!=s['exe_sha256']:raise ValueError('Reviewed engine changed')
        env=json.loads((old/'environment.json').read_text(encoding='utf-8'))
        for k,v in env.items():
            for a,b in sorted(mapping.items(),key=lambda x:-len(x[0])):v=v.replace(a,b)
            env[k]=v
        env['PTINV']=linux(folder/'original_0.csv')
        if s['name']=='speciation':env.update(tables)
        if s['name']=='temporal':(target/'procdates.txt').write_bytes((old/'procdates.txt').read_bytes())
        (target/'environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
        with (target/'console.log').open('wb') as log:
            p=subprocess.run(['wsl.exe','-d','EcoData-WRF-Ubuntu2204','-u','root','--','env',*[k+'='+v for k,v in env.items()],linux(exe)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        report['steps'].append({'name':s['name'],'returncode':p.returncode,'engine_sha256':_sha(exe)})
        (folder/'processing_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if p.returncode:return report
    report['success']=True
    (folder/'processing_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
