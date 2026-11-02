"""Legacy experimental WASP text package; not used by the ordinary workflow.

This is a text writer, never a WIF binary reader/writer. A generated INP is
not native-verified until an independent WASP8 import/export is checked.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ecodata.core.exceptions import SchemaError
from ecodata.models.adapters.standardized import ModelAdapter

REFERENCE='https://github.com/respec/BASINS/blob/master/atcWASPInpWriter/atcWASPProject.vb'


class WASPInpWriter:
    def write(self, config):
        required=['module_type','start','end','state_variables','segments','flow_paths',
                  'boundaries','loads','time_functions','kinetic_parameters','solver']
        missing=[key for key in required if key not in config]
        if missing:raise SchemaError('WASP 配置缺少：'+', '.join(missing))
        if not config['segments'] or not config['state_variables']:
            raise SchemaError('WASP 必须提供实际分段和 state variable')
        if config['kinetic_parameters'] is None:
            raise SchemaError('kinetic parameters 必须明确配置，不能自动补值')
        if not config['kinetic_parameters'] and not config.get('transport_only'):
            raise SchemaError('缺少 kinetic parameters；空常数仅允许明确的 transport_only 工程')
        start,end=pd.Timestamp(config['start']),pd.Timestamp(config['end'])
        if end<=start:raise SchemaError('WASP end 必须晚于 start')
        segments=config['segments']; states=config['state_variables']
        state_names={s['name'] for s in states}
        if len(state_names)!=len(states):raise SchemaError('state variable 重复')
        if any(s.get('unit')!='mg/L' for s in states):raise SchemaError('当前浓度 state variable 必须明确 mg/L 单位')
        if any(p['field'] not in range(1,7) for p in config['flow_paths']):raise SchemaError('flow field 必须为1至6')
        for role,expected in [('flow_paths','m^3/s'),('boundaries','mg/L'),('loads','kg/day')]:
            for item in config[role]:
                if item.get('unit')!=expected:raise SchemaError(f'{role} 必须先统一到 {expected}')
                if role!='flow_paths' and item['variable'] not in state_names:
                    raise SchemaError('时间序列变量没有对应 state variable，禁止丢弃')
        if config.get('hydrodynamics_file'):
            raise SchemaError('外部 HYD 链接 card 尚未验证，不能冒充已自动链接')
        ids={s['id'] for s in segments}
        if ids!=set(range(1,len(segments)+1)) or len(ids)!=len(segments):
            raise SchemaError('segment id 必须唯一且连续，从1开始')
        lines=[]
        def section(name):lines.append(' ======================== '+name+' ===================================')
        def number(value,label):lines.append(f'{value:5}               {label}')
        def pairs(series):
            if not series:raise SchemaError('缺少真实时间序列，不填零')
            times=pd.DatetimeIndex([x['time'] for x in series])
            values=np.asarray([x['value'] for x in series],dtype=float)
            if not np.isfinite(values).all() or not times.is_monotonic_increasing or times.has_duplicates:
                raise SchemaError('时间序列存在非有限值、重复或倒序')
            if times[0]>start or times[-1]<end:raise SchemaError('时间序列未覆盖工程时间，禁止外推')
            number(len(series),'Number of Time Pairs')
            for time,value in zip(times,values):
                lines.append(f'{(time-start).total_seconds()/86400:16.9f} {value:20.12g}')
        lines.append(f"{config['module_type']:5}{3:7}          Module Type, Version")
        section('Model Parameterization')
        for label in ['Start date and time','Skip date and time']:
            lines.append(f'{start.month:5}{start.day:5}{start.year:5}{start.hour:5}{start.minute:5}{start.second:5}     '+label)
        lines.append(f'{(end-start).total_seconds()/86400:10.5f}          Julian end time')
        number(len(states),'Number of Systems')
        solver=config['solver']
        for key,label in [('mass_balance','Mass Balance Table Output'),('solution','Solution Technique Option'),
                          ('negative','Negative Solution Option'),('restart','Restart Option'),
                          ('time_optimization','Time Optimization Option'),('wq_linkage','WQ Module Linkage Option')]:
            number(solver[key],label)
        lines.extend([f"{solver['topt']:10.6f}          TOPT Factor",
                      f"{solver['min_step']:10.6f} {solver['max_step']:10.6f}     Min and Max Timestep",
                      f"{solver['print_interval']:10.6f}     Time and Print Interval"])
        for state in states:lines.append(f"{state['tag']:5}     {state['name']:<40}")
        section('Segment Information');number(len(segments),'Number of Segments')
        number(config['bed_volume_option'],'Bed Volume Option')
        lines.extend([f"{config['bed_compaction_step']:10.6f}          Bed Compaction Time Step",'   1.000   1.000    Volume Scale & Conversion Factor','Segment   SegName'])
        for s in segments:lines.append(f"{s['id']:5}     {s['name']:<40}")
        geometry=['volume','volume_multiplier','volume_exponent','depth_multiplier','depth_exponent',
                  'length','slope','width','roughness','depth_q0','surface_elevation','bottom_elevation',
                  'side_slope','initial_depth','weir_height','weir_type']
        lines.append('Seg BotSeg iType IQOPT     Volume VMult Vexp DMult Dexp Length Slope Width Rough Depth_Q0 SurfElev SegBotElev SegSideSlope SegInitDepth WeirHeight WeirType')
        for s in segments:
            if not all(np.isfinite(s[k]) for k in geometry) or s['volume']<=0:raise SchemaError('segment geometry 必须有限，volume 大于0')
            lines.append(''.join(f'{s[k]:5}' for k in ['id','bottom_segment','type','flow_option'])+''.join(f'{s[k]:15.5f}' for k in geometry))
        section('Flow Path Information');lines.append('               Flow Pathways');number(6,'Number of flow fields')
        paths=config['flow_paths']
        for field in range(1,7):
            selected=[x for x in paths if x['field']==field]
            lines.append(f'     Flow Field   {field}');number(len(selected),'Number of Flow Functions for Flow Field')
            for p in selected:
                lines.append('  '+p['name']);number(len(p['routes']),'Number of routes')
                for route in p['routes']:
                    if route['from'] not in ids|{0} or route['to'] not in ids|{0}:raise SchemaError('flow segment mapping 缺失')
                    lines.append(f"{route['from']:4}{route['to']:4}{route['fraction']:15.8f}")
        section('Flow Time Series');number(6,'Number of Flow Fields')
        for field in range(1,7):
            selected=[x for x in paths if x['field']==field]
            lines.append(f' Flow Field {field}');number(len(selected),'Number of inflows')
            if selected:lines.append('      1.000000      1.000000  Flow Scale & Conversion Factors')
            for p in selected:lines.append(p['name']);pairs(p['series'])
        section('Segment Dispersion/Exhange');number(0,'Number of Exchange Fields')
        if config.get('exchanges'):raise SchemaError('当前最小 Writer 尚未实现 exchange cards，不能静默忽略')
        section('Boundary Information');number(len(states),'Number of Systems')
        for state in states:
            selected=[x for x in config['boundaries'] if x['variable']==state['name']]
            lines.append('  '+state['name']);number(len(selected),'Number of boundaries')
            if selected:lines.append('      1.000000      1.000000  Boundary Scale & Conversion Factors')
            for x in selected:
                if x['segment_id'] not in ids:raise SchemaError('boundary segment mapping 缺失')
                number(x['segment_id'],'Boundary Segment Number');pairs(x['series'])
        section('Load Information');number(0,'NPS Input Option (0=No, 1=Yes)');number(len(states),'Number of Systems')
        for state in states:
            selected=[x for x in config['loads'] if x['variable']==state['name']]
            lines.append('  '+state['name']);number(len(selected),'Number of Loadings')
            if selected:lines.append('      1.000000      1.000000  Loading Scale & Conversion Factors')
            for x in selected:
                if x['segment_id'] not in ids:raise SchemaError('load segment mapping 缺失')
                number(x['segment_id'],'Loading Segment Number');pairs(x['series'])
        section('Time Functions');number(len(config['time_functions']),'Number of Time Functions')
        for x in config['time_functions']:
            lines.append(f"{x['id']:8}       1     {x['name']}");pairs(x['series'])
        section('Segment Parameters');number(len(segments),'Number of Segments');number(0,'Number of Segment Parameters')
        lines.append(' ===============================================================================')
        # Non-empty kinetic cards require a verified module-specific exported template.
        if config['kinetic_parameters']:
            raise SchemaError('非空 kinetic parameters 尚需对应模块的官方导出模板，不猜常数编号')
        section('Constants');number(0,'Number of Constants')
        section('Initial Conditions');number(len(states),'');number(len(segments),'')
        for state in states:
            lines.extend(['     Initial conditions for system '+state['name'],f"{state['solids_field']:5}               Solids Transport Field",f"{state['solids_density']:15.6f}     Solids Density, g/mL",f"{state['max_concentration']:15.6f}     Maximum Allowed Concentration",'  Seg   Conc   DissF'])
            for s in segments:
                initial=state['initial'].get(str(s['id']))
                if initial is None:raise SchemaError('initial condition 缺失，不填零')
                if not np.isfinite(initial) or initial<0:raise SchemaError('initial condition 非有限或负值')
                lines.append(f"{s['id']:5} {initial:16.9g} {state['dissolved_fraction']:12.8f}")
        return ('\r\n'.join(lines)+'\r\n').encode('ascii')


class WASPAdapter(ModelAdapter):
    def package(self, config, sources=None):
        files={}; issues=[]
        try:files['model.inp']=WASPInpWriter().write(config)
        except (SchemaError,KeyError,ValueError) as exc:issues.append(str(exc))
        for role in ['flow_paths','boundaries','loads','time_functions']:
            rows=[]
            for item in config.get(role,[]):
                for pair in item.get('series',[]):rows.append({**{k:v for k,v in item.items() if k not in ['series','routes']},**pair})
            if rows:
                filename={'flow_paths':'tributary_flow','boundaries':'boundary_concentration','loads':'point_source_load','time_functions':'environmental_functions'}[role]
                value_name={'flow_paths':'flow','boundaries':'concentration','loads':'load','time_functions':'value'}[role]
                files[f'timeseries/{filename}.csv']=pd.DataFrame(rows).rename(columns={'value':value_name}).to_csv(index=False).encode('utf8')
        report=dict(status='FAIL' if issues else 'WARN',issues=issues,standardized_preprocessing=bool(files.get('timeseries/tributary_flow.csv')),
                    model_format_matched=False,native_model_input_verified=False,
                    warnings=['INP 是待 WASP8 实测的候选文本，不是已验证原生输入','CSV 是可审查侧车，不声称已自动建立 ODBC/WRDB 外部链接'])
        manifest=dict(model='WASP',version='8.5',writer_reference=REFERENCE,sources=sources or [],
                      files={k:dict(size=len(v),sha256=hashlib.sha256(v).hexdigest()) for k,v in files.items()},
                      configuration=config,acceptance=report)
        files['manifest.json']=json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf8')
        files['validation_report.json']=json.dumps(report,ensure_ascii=False,indent=2).encode('utf8')
        return files

    def save_package(self, directory, config, sources=None):
        directory=Path(directory)
        if directory.exists() and any(directory.iterdir()):raise SchemaError('输出目录非空，禁止覆盖旧验收证据')
        for name,content in self.package(config,sources).items():
            path=directory/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
        return directory
