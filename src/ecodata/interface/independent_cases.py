"""Explicitly registered independent cases, never arbitrary uploaded programs."""
from pathlib import Path
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(os.environ.get('ECODATA_INDEPENDENT_CASES', r'C:\Users\ASUS\Documents\ChatGPT\测试重\完整融合案例_20261028'))
RUNS = ROOT / 'outputs/independent_case_jobs'
LOCK = threading.Lock()
ACTIVE = None
PROFILES = {
    'wrf': dict(label='Colorado 2021-03-13（新登记案例）', folder='WRF_Colorado', mode='pressure_fields',
                raw='original_pressure_fields.nc', converted='platform_converted_pressure.zip',
                scope='49×49点、12km、48层，3小时积分。新建重跑geogrid/ungrib；已有复用地理和完整配套场。区域与物理方案固定。'),
    'calpuff': dict(label='Cambridge 2016-01-01（新登记案例）', folder='CALPUFF_Cambridge', mode='receptors',
                    raw='source_receptors_and_control.zip', converted='platform_conversion.zip',
                    scope='CALMET6.5与公开改版CALPUFF7.2.1串行构建；原6受体、示范NOX点源。重建仅重算CALMET，地理/已预处理观测复用，不是原始GIS全链。'),
    'camx': dict(label='2020-11-02 CB05（新登记案例）', folder='CAMx_202011', mode='gridded_emissions',
                 raw='original_NO.csv', converted='platform_conversion.zip',
                 scope='历史6.00/CB05-CF一小时场景，非作者6.10复现。平台7.10中间NO经二进制接入；30/31清单物种映射，NR语义未确认；不是完整原始清单新建。'),
    'cmaq': dict(label='LISTOS 2018-08-05（新登记案例）', folder='CMAQ_LISTOS', mode='gridded_emissions',
                 raw='original_NO.nc', converted=None,
                 scope='CMAQ5.5/CB6R3一小时。复用MCIP/ICBC与4类点源；BEIS/MEGAN/闪电/双向氨按已验证场景关闭。重组非完整原始清单新建。'),
}


def directory(ident):
    if not re.fullmatch(r'[0-9a-f]{32}', ident):
        raise ValueError('无效独立案例任务号')
    return RUNS / ident


def status(ident):
    return json.loads((directory(ident) / 'status.json').read_text(encoding='utf-8'))


def save(folder, record):
    temp = folder / 'status.tmp'
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    # Windows readers may briefly hold the previous status file while a page
    # refreshes. Retry only this sharing violation, never rewrite scientific data.
    for attempt in range(50):
        try:
            temp.replace(folder / 'status.json')
            break
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(.02)


def linux(path):
    p = Path(path).resolve()
    return '/mnt/' + p.drive[0].lower() + '/' + p.as_posix()[3:]


def members(payload):
    if len(payload) > 128 * 1024**2:
        raise ValueError('转换包超过128MiB')
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        items = z.infolist()
        if len(items) > 100 or sum(i.file_size for i in items) > 128 * 1024**2:
            raise ValueError('转换包解压大小/数量超限')
        result = {}
        for item in items:
            name = item.filename
            if item.is_dir() or '/' in name or '\\' in name or ':' in name or name in ('.', '..'):
                raise ValueError('转换包只接受平铺数据文件')
            if name.casefold() in result:
                raise ValueError('转换包文件重名')
            result[name.casefold()] = (name, z.read(item))
        return dict(result.values())


def scientific_equal(name, current, reference):
    if not name.endswith('.nc'):
        return current == reference
    import numpy as np
    from netCDF4 import Dataset
    # I/O API creation/write clock metadata are not model time (TFLAG/SDATE/STIME).
    volatile = {'CDATE','CTIME','WDATE','WTIME'}
    with Dataset('current', memory=current) as a, Dataset('reference', memory=reference) as b:
        if set(a.dimensions) != set(b.dimensions) or set(a.variables) != set(b.variables):
            return False
        if any((len(a.dimensions[k]),a.dimensions[k].isunlimited()) != (len(b.dimensions[k]),b.dimensions[k].isunlimited()) for k in a.dimensions):
            return False
        attrs_a=set(a.ncattrs())-volatile; attrs_b=set(b.ncattrs())-volatile
        if attrs_a != attrs_b or any(not np.array_equal(a.getncattr(k),b.getncattr(k)) for k in attrs_a):
            return False
        for k in a.variables:
            x,y=a[k],b[k]
            if x.dimensions != y.dimensions or x.dtype != y.dtype or set(x.ncattrs()) != set(y.ncattrs()):
                return False
            if any(not np.array_equal(x.getncattr(n),y.getncattr(n)) for n in x.ncattrs()):
                return False
            if not np.array_equal(x[:],y[:]):
                return False
    return True


def start(model, route, converted_zip=None, mode=None, chemical_profile=None, meteorology_files=None, accept_reference_defaults=False, raw_oilgas_inventory=None, raw_non_egu_inventory=None, raw_cmv_inventory=None, raw_cmv_hourly=None):
    global ACTIVE
    if model not in PROFILES or route not in ('existing', 'rebuild'):
        raise ValueError('未知案例或路线')
    if chemical_profile is not None and (model != 'cmaq' or route != 'rebuild'):
        raise ValueError('完整化学剖面目前仅用于CMAQ新建中的化学初边界生成')
    if accept_reference_defaults and (model!='cmaq' or route!='rebuild' or chemical_profile is None):
        raise ValueError('参考默认必须显式提供CMAQ新建化学剖面')
    if meteorology_files is not None and (model != 'cmaq' or route != 'rebuild'):
        raise ValueError('上传WRF新建MCIP目前仅用于CMAQ新建路线')
    if raw_oilgas_inventory is not None and (model!='cmaq' or route!='rebuild'):
        raise ValueError('原始NEI清单目前仅用于CMAQ新建路线')
    if raw_non_egu_inventory is not None and (model!='cmaq' or route!='rebuild'):
        raise ValueError('原始非电力NEI清单目前仅用于CMAQ新建路线')
    if raw_cmv_inventory is not None or raw_cmv_hourly is not None:
        if model!='cmaq' or route!='rebuild':raise ValueError('原始船舶清单目前仅用于CMAQ新建路线')
        from ecodata.interface.cmaq_smoke_inventory import validate,validate_hourly
        validate(raw_cmv_inventory,'cmv');validate_hourly(raw_cmv_hourly)
    profile = PROFILES[model]
    if mode is not None and mode != profile['mode']:
        raise ValueError('此新登记案例尚未验证该转换分类；不回退旧示例')
    with LOCK:
        if ACTIVE is not None:
            raise ValueError('独立案例正在运行；四模型原生任务串行执行')
        ACTIVE = uuid.uuid4().hex
        ident = ACTIVE
    folder = directory(ident)
    try:
        folder.mkdir(parents=True, exist_ok=False)
        if raw_cmv_inventory is not None:
            (folder/'raw_cmv_inventory.csv').write_bytes(raw_cmv_inventory)
            (folder/'raw_cmv_hourly.csv').write_bytes(raw_cmv_hourly)
        if raw_non_egu_inventory is not None:
            from ecodata.interface.cmaq_smoke_inventory import validate
            validate(raw_non_egu_inventory,'non_egu')
            (folder/'raw_non_egu_inventory.csv').write_bytes(raw_non_egu_inventory)
        if raw_oilgas_inventory is not None:
            from ecodata.interface.cmaq_smoke_inventory import validate
            validate(raw_oilgas_inventory)
            (folder/'raw_oilgas_inventory.csv').write_bytes(raw_oilgas_inventory)
        if meteorology_files is not None:
            from ecodata.interface.cmaq_listos_meteorology import validate_inputs
            meteorology_report = validate_inputs(meteorology_files)
            for i,content in enumerate(meteorology_files):
                (folder/f'uploaded_WRF_{i:02d}.nc').write_bytes(content)
        if chemical_profile is not None:
            from ecodata.interface.cmaq_chemical_profiles import required_species, profile_contract
            reference = SOURCE/'data/CMAQ_LISTOS/icbc/cb6r3_ae7_aq/ICON_v54_2018_12Listos_regrid_20180805.nc'
            chemical_contract = required_species(reference)
            _,chemical_report = profile_contract(chemical_profile, chemical_contract,accept_reference_defaults)
            (folder/'chemical_profile.csv').write_bytes(chemical_profile)
            (folder/'chemical_contract.json').write_text(json.dumps(chemical_contract,indent=2),encoding='utf-8')
        inputs = SOURCE / 'platform_inputs' / profile['folder']
        from ecodata.round1 import convert
        config = json.loads((inputs / 'config.json').read_text(encoding='utf-8'))
        files, report = convert(model, (inputs / profile['raw']).read_bytes(), config)
        # Only scientific data members must match. Reports can contain timestamps.
        expected = {n: b for n, b in files.items() if n.endswith(('.wps', '.nc', '.INP'))}
        if not expected:
            raise ValueError('转换没有生成可接入的科学输入文件')
        if converted_zip is not None:
            uploaded = members(converted_zip)
            selected = {n: b for n, b in uploaded.items() if n.endswith(('.wps', '.nc', '.INP'))}
            if set(selected) != set(expected) or any(not scientific_equal(n,selected[n],expected[n]) for n in expected):
                raise ValueError('本次转换与所选新登记案例不匹配；不替换为登记数据')
            files = uploaded
        for name, content in files.items():
            if '/' in name or '\\' in name:
                raise ValueError('新登记案例转换必须为平铺文件')
            (folder / name).write_bytes(content)
        # Keep an explicit current conversion record, separate from native output.
        (folder / 'conversion.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        record = dict(id=ident, model=model, route=route, state='running', stage='准备新登记案例',
                      source=str(SOURCE), scope=profile['scope'], accuracy_evaluated=False,
                      scientific_input_sha256={n: hashlib.sha256(files[n]).hexdigest() for n in expected},
                      equivalence_policy='Exact scientific values, axes, units and attributes; NetCDF CDATE/CTIME/WDATE/WTIME only excluded. WPS/control bytes exact.',
                      supplied_conversion=converted_zip is not None)
        if chemical_profile is not None:
            record['chemical_profile'] = chemical_report
            record['accept_reference_defaults'] = bool(accept_reference_defaults)
            record['scope'] += ('；显式采用已核验官方参考剖面生成初边界，五项缺字段由原生采用最小值并单列，不代表完整化学背景。' if accept_reference_defaults else '；显式采用上传完整剖面生成全部化学初边界。')+'均匀静态背景，非原三维化学场复现；本次是否通过以实际任务状态为准。'
        if meteorology_files is not None:
            record['meteorology_inputs'] = meteorology_report
            record['scope'] += '；本次上传WRF实际生成MCIP。固定LISTOS网格/时段/35层，非任意区域，也不是从FNL重跑WRF；其余化学与排放资料按所选路线明确复用。'
        if raw_oilgas_inventory is not None:
            record['raw_oilgas_inventory_sha256']=hashlib.sha256(raw_oilgas_inventory).hexdigest()
            record['scope']+='；本次原始NEI油气部门经六步SMOKE重建，替换第三点源部门；面源与未单独选择的其余部门保留，不是全区域排放新建。'
        if raw_non_egu_inventory is not None:
            record['raw_non_egu_inventory_sha256']=hashlib.sha256(raw_non_egu_inventory).hexdigest()
            record['scope']+='；本次原始NEI非电力部门正烟囱子集经六步SMOKE重建，替换第一点源部门；33个颗粒物资料矛盾/不全的整源已隔离，逸散源因该分支不处理而未纳入，不应称为坏数据。不是完整区域。该部门启用PMC差分和明确的零排放免分配；其他未选择部门仍复用。'
        if raw_cmv_inventory is not None:
            record['raw_cmv_inventory_sha256']=hashlib.sha256(raw_cmv_inventory).hexdigest()
            record['raw_cmv_hourly_sha256']=hashlib.sha256(raw_cmv_hourly).hexdigest()
            record['scope']+='；船舶225个完整三天逐小时源子集经六步SMOKE重建第四点源流，PMC由原生同源差分一次；不补缺小时。不是全区域船舶排放，也不是实测小时排放。未选择部门、MCIP及初边界按各自选择复用。'
        save(folder, record)
        threading.Thread(target=work, args=(folder, record), daemon=True).start()
        return ident
    except Exception as exc:
        save(folder,dict(id=ident,model=model,route=route,state='failed',stage='完整化学剖面校验' if chemical_profile is not None else '转换输入校验',error=str(exc),native_started=False))
        with LOCK:
            ACTIVE = None
        raise


def work(folder, record):
    global ACTIVE
    try:
        if record.get('raw_cmv_inventory_sha256'):
            from ecodata.interface.cmaq_smoke_inventory import build as build_emissions
            record['stage']='原始船舶年度及小时子集经六步SMOKE生成输入';save(folder,record)
            record['new_smoke_cmv']=build_emissions(folder/'raw_cmv',(folder/'raw_cmv_inventory.csv').read_bytes(),SOURCE.parent/'双任务推进_20261030',sector='cmv',hourly_payload=(folder/'raw_cmv_hourly.csv').read_bytes())
            save(folder,record)
        if record.get('raw_non_egu_inventory_sha256'):
            from ecodata.interface.cmaq_smoke_inventory import build as build_emissions
            record['stage']='原始NEI非电力子集经六步SMOKE生成点源输入';save(folder,record)
            record['new_smoke_non_egu']=build_emissions(folder/'raw_non_egu',(folder/'raw_non_egu_inventory.csv').read_bytes(),SOURCE.parent/'双任务推进_20261030',sector='non_egu')
            save(folder,record)
        if record.get('raw_oilgas_inventory_sha256'):
            from ecodata.interface.cmaq_smoke_inventory import build as build_emissions
            record['stage']='原始NEI油气清单经六步SMOKE生成点源输入';save(folder,record)
            record['new_smoke']=build_emissions(folder/'raw_emissions',(folder/'raw_oilgas_inventory.csv').read_bytes(),SOURCE.parent/'双任务推进_20261030')
            save(folder,record)
        if record.get('meteorology_inputs'):
            from ecodata.interface.cmaq_listos_meteorology import build
            record['stage'] = '本次上传WRF生成MCIP并核验目标网格'
            save(folder,record)
            record['new_mcip'] = build(folder/'raw_meteorology',
                [p.read_bytes() for p in sorted(folder.glob('uploaded_WRF_*.nc'))],
                SOURCE/'data/CMAQ_LISTOS/met/mcip')
            save(folder,record)
        if record.get('chemical_profile'):
            from ecodata.interface.cmaq_chemical_profiles import build_pair
            record['stage'] = '完整化学剖面生成ICON和BCON'
            save(folder,record)
            record['chemical_preprocessing'] = build_pair(folder/'chemical_preprocessing',
                (folder/'chemical_profile.csv').read_bytes(),
                json.loads((folder/'chemical_contract.json').read_text()),
                folder/'raw_meteorology' if record.get('new_mcip') else SOURCE/'data/CMAQ_LISTOS/met/mcip',
                '2018_12Listos',folder/'raw_meteorology/GRIDDESC' if record.get('new_mcip') else SOURCE/'data/CMAQ_LISTOS/met/mcip/GRIDDESC',
                accept_reference_defaults=record.get('accept_reference_defaults',False))
            save(folder,record)
        if record['model'] == 'calpuff':
            original = json.loads((SOURCE / 'evidence/Cambridge_CALMET_latest.json').read_text())
            if record['route'] == 'rebuild':
                stage = folder / 'calmet'; stage.mkdir()
                source = SOURCE / 'data/CALPUFF_Cambridge/calmet'
                for name in ('calmet_test.inp', 'geo_camb_100m.dat', 'surf_bostonf.dat', 'up_14684_74494_chh_2016-2020.dat'):
                    shutil.copyfile(source / name, stage / name)
                exe = Path(original['engine'])
                if hashlib.sha256(exe.read_bytes()).hexdigest() != original['engine_sha256']:
                    raise ValueError('CALMET程序哈希变化')
                result = subprocess.run([str(exe), 'calmet_test.inp'], cwd=stage, input=b'', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
                (stage / 'console.log').write_bytes(result.stdout)
                if result.returncode or b'ENTERING TERMINATION PHASE' not in result.stdout:
                    raise RuntimeError('CALMET重建失败；保留日志')
                record['new_calmet'] = str(stage)
            else:
                record['new_calmet'] = original['folder']
            save(folder, record)
        science = SOURCE.parent / '四模型页面路线验收_20261029/linux_science_modules'
        command = ['wsl.exe', '-d', 'EcoData-WRF-Ubuntu2204', '-u', 'root', '--',
                   'env', 'PYTHONPATH='+linux(science), 'OPENBLAS_NUM_THREADS=1', 'python3',
                   linux(Path(__file__).with_name('independent_case_worker.py')), linux(folder), linux(SOURCE)]
        with (folder / 'runner.log').open('wb') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=5400)
        if process.returncode:
            raise RuntimeError(f'原生案例返回{process.returncode}，请查看本次runner.log')
        native = json.loads((folder / 'native_result.json').read_text())
        if not native['success']:
            raise RuntimeError('原生完成标记或数值检查未通过')
        plot(folder, record['model'], native)
        record.update(state='succeeded', stage='原生运行与数值出图完成', native=native)
    except Exception as exc:
        record.update(state='failed', error=str(exc))
    finally:
        try:
            save(folder, record)
        finally:
            with LOCK:
                ACTIVE = None


def plot(folder, model, native):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    arrays = np.load(folder / 'plot_data.npz')
    figure = Figure(figsize=(8, 5)); FigureCanvasAgg(figure); ax = figure.subplots()
    a = arrays['value']
    if model == 'calpuff':
        for i in range(a.shape[1]):
            ax.plot(arrays['time'], a[:, i], 'o-', label=f'Receptor {i+1}')
        ax.legend(ncol=3); ax.set(xlabel='Local hour ending (UTC-5)', ylabel='NOX (ug/m3)')
        table = np.column_stack([np.repeat(arrays['time'], a.shape[1]), np.tile(np.arange(1, a.shape[1]+1), a.shape[0]), a.ravel()])
        header = 'local_hour_ending,receptor,NOX_ug_m3'
    else:
        image = ax.pcolormesh(arrays['x'], arrays['y'], a, shading='auto', cmap='viridis')
        figure.colorbar(image, ax=ax, label=native['plot_units'])
        ax.set(xlabel=native['x_label'], ylabel=native['y_label'])
        x, y = np.meshgrid(arrays['x'], arrays['y']) if arrays['x'].ndim == 1 else (arrays['x'], arrays['y'])
        table = np.column_stack([x.ravel(), y.ravel(), a.ravel()]); header = 'x,y,value'
    ax.set_title(native['plot_title'] + '\nNative acceptance scenario; accuracy not assessed', fontsize=11)
    figure.tight_layout(); figure.savefig(folder / 'result.png', dpi=150)
    np.savetxt(folder / 'result.csv', table, delimiter=',', header=header, comments='')


def render(model, converted_zip=None, mode=None):
    import streamlit as st
    p = PROFILES[model]
    options=['原登记案例',p['label']]
    if model=='wrf':options.append('自定义原始GFS/FNL新建工程（试验）')
    if model=='calpuff':options.append('WRF驱动新建 / 上传已有工程（试验）')
    if model=='cmaq':options.append('上传已有工程生效配置（本机试验）')
    if model=='camx':options.append('接入本机完整已有工程（试验）')
    selected = st.selectbox('验证案例', options, key=model+'_independent_case')
    if model=='wrf' and selected==options[-1]:
        from ecodata.interface.wrf_project import render as render_project
        return render_project(converted_zip,mode)
    if model=='calpuff' and selected==options[-1]:
        from ecodata.interface.calpuff_project import render as render_project
        return render_project(converted_zip,mode)
    if model=='cmaq' and selected==options[-1]:
        from ecodata.interface.cmaq_external_project import render
        return render(converted_zip,mode)
    if model=='camx' and selected==options[-1]:
        from ecodata.interface.camx_external_project import render
        return render(converted_zip,mode)
    if selected == '原登记案例':
        return False
    st.header(p['label'])
    st.warning(p['scope'])
    st.caption('仅开放此案例已经验证的转换类别：'+p['mode']+'。有入口不代表任意区域/版本通用。')
    if mode is not None and mode != p['mode']:
        st.error('上方转换分类与所选新案例不匹配。请换回对应分类，不会回退旧案例。')
        return True
    labels = ['接入已有工程', '从原始GFS及登记地理库新建'] if model == 'wrf' else ['接入已有工程', '重建登记配套工程（部分成品复用）']
    choice = st.radio('新案例工程路线', labels, key=model+'_independent_route')
    route = 'existing' if choice == labels[0] else 'rebuild'
    pre = ('geogrid → ungrib → 原始格点裁剪 → 本次气压场接入' if route == 'rebuild' else '复用geo_em和完整配套场 → 本次气压场接入') if model == 'wrf' else ('按已登记配套资料重建；原始全链未实现' if route == 'rebuild' else '保留工程配置，接入本次转换')
    st.write('转换 → '+pre+' → 原生计算 → 数值检查 → 本次图表/CSV')
    graph='digraph { rankdir=LR; node [shape=box]; a[label="本次转换与校验"]; b[label="'+pre+'"]; c[label="原生运行"]; d[label="数值检查与图表"]; a->b->c->d; '
    if model in ('camx','cmaq') and route=='rebuild':
        graph+='x[label="原始清单分配/独立ICBC构建：未实现",style=dashed]; x->b[style=dashed]; '
    graph+='}'
    if model!='cmaq' or route!='rebuild':st.graphviz_chart(graph)
    st.caption('配套资料位置：'+str(SOURCE/'platform_inputs'/p['folder']))
    if model in ('camx', 'cmaq') and route == 'rebuild':
        st.caption('完整区域原始清单链尚未完成；CMAQ下方可分别选择已验收油气、非电力、船舶部门子集、新气象与参考化学背景，不能把部分部门代表全部排放。' if model=='cmaq' else '原始清单时空/化学分配及独立ICBC构建仍未实现；本按钮不代表这部分已经完成。')
    chemical_profile = None
    meteorology_files = None
    use_wrf = False
    accept_reference_defaults = False
    raw_oilgas_inventory = None
    raw_non_egu_inventory = None
    raw_cmv_inventory = None
    raw_cmv_hourly = None
    use_cmv = False
    use_non_egu = False
    use_nei = False
    if model == 'cmaq' and route == 'rebuild':
        use_nei=st.checkbox('试验：原始NEI油气部门经SMOKE重新生成点源输入',key='cmaq_raw_oilgas_enabled')
        if use_nei:
            st.warning('仅验收2018NEI东北油气子集。重跑清单读取、化学分配、网格、时间分配、高架源和合并；只替换第三点源部门，面源及未单独选择的其余部门仍复用。不是全区域排放从零。')
            nei=st.file_uploader('已核验原始NEI油气子集CSV',type=['csv'],key='cmaq_raw_oilgas_upload')
            if nei is not None:raw_oilgas_inventory=nei.getvalue()
        use_non_egu=st.checkbox('试验：原始NEI非电力部门经SMOKE重新生成点源输入',key='cmaq_raw_non_egu_enabled')
        if use_non_egu:
            st.warning('仅接受已核验2018NEI非电力质量筛选子集，替换第一点源部门。当前只验收完整正烟囱参数的源；类型1逸散源未纳入，不能把它们视为错误资料或假设其没有排放。另33个PM10/PM2.5矛盾或缺项的整源已隔离留档。显式采用PMC=PM10-PM2.5及官方零排放免分配，不补值、不裁负值。其他未选择部门继续复用，不是完整区域新建。')
            non_egu=st.file_uploader('已核验原始NEI非电力子集CSV',type=['csv'],key='cmaq_raw_non_egu_upload')
            if non_egu is not None:raw_non_egu_inventory=non_egu.getvalue()
        use_cmv=st.checkbox('试验：原始船舶年度与小时清单经SMOKE重新生成输入',key='cmaq_raw_cmv_enabled')
        if use_cmv:
            st.warning('仅验收225个源、8月4—6日七污染物完整小时子集，替换第四点源流；不是完整区域船舶清单。剔除缺源日或日总量矛盾的整源，不补缺小时，不把未纳入源当作没有排放。PMC仅由原生差分一次；已核验时区、年排放、逐小时原生读取和时间分配。源资料是由2017清单投影的2018估算，不是实测。')
            cmv=st.file_uploader('已核验船舶年度子集CSV',type=['csv'],key='cmaq_raw_cmv_upload')
            cmv_hourly=st.file_uploader('匹配的船舶逐小时子集CSV',type=['csv'],key='cmaq_raw_cmv_hourly_upload')
            if cmv is not None:raw_cmv_inventory=cmv.getvalue()
            if cmv_hourly is not None:raw_cmv_hourly=cmv_hourly.getvalue()
        use_wrf = st.checkbox('试验：上传同案例WRF，重新生成MCIP气象输入',key='cmaq_uploaded_wrf_enabled')
        if use_wrf:
            st.warning('限定LISTOS：12公里、35层，须覆盖2018-08-04 23时至08-05 01时UTC。起点是已生成的WRF产品，不包含原始FNL到WRF；排放仍使用登记配套资料，化学背景按下方选择。不会将不匹配的区域强接本工程。')
            wrf_uploads = st.file_uploader('同案例WRF NetCDF文件（可多选）',type=['nc','ncf'],accept_multiple_files=True,key='cmaq_uploaded_wrf_files')
            if wrf_uploads:
                meteorology_files = [item.getvalue() for item in wrf_uploads]
        use_profile = st.checkbox('试验：用完整化学剖面新生成初始及边界文件',key='cmaq_complete_profile_enabled')
        if use_profile:
            st.warning('完整剖面入口已实现校验与ICON/BCON调用；全部字段齐全的独立剖面尚未完成CCTM验收。下方另有已通过限定案例的官方参考背景情景，须单独接受五项原生默认。剖面形成水平一致、时间不变背景，改变原三维化学分布，不代表当地观测。气象及排放按各自选项处理。')
            uploaded = st.file_uploader('完整化学剖面CSV（全部所需物种、单位、ZH与PRES；不是单独O3）',type=['csv'],key='cmaq_complete_profile_upload')
            if uploaded is not None:
                chemical_profile = uploaded.getvalue()
            accept_reference_defaults = st.checkbox('我选择参考背景情景：仅对已核验官方剖面，接受APNCOMI、APNCOMJ、APOCI、APOCJ、IEPOXP由CCTM原生设为1e-30；这五项不是资料实测值，也不是完整化学输入',key='cmaq_accept_reference_defaults')
            st.caption('不勾选则要求全部字段齐全；勾选也不补写字段、不接受任意残缺CSV。参考剖面为2016远洋年均模拟背景，改变的是化学情景，不代表2018当地背景。')
        # Report server receipt, rather than relying on the browser file-name list.
        receipt = []
        for enabled, label, payload in [
            (use_nei, '油气年度清单', raw_oilgas_inventory),
            (use_non_egu, '非电力年度清单', raw_non_egu_inventory),
            (use_cmv, '船舶年度清单', raw_cmv_inventory),
            (use_cmv, '船舶小时清单', raw_cmv_hourly),
            (use_profile, '化学剖面', chemical_profile),
        ]:
            if enabled:
                receipt.append(label + ('：后台已收到 ' + str(len(payload)) + ' 字节' if payload is not None else '：后台尚未收到，请重新上传'))
        if use_wrf:
            receipt.append('WRF：后台已收到 ' + str(len(meteorology_files or [])) + ' 个文件')
        if receipt:
            st.info('本次输入接收状态（以后台为准）\n\n' + '\n\n'.join(receipt))
        st.graphviz_chart('digraph { rankdir=LR; node [shape=box]; a[label="平台转换与输入校验"]; b[label="'+('上传WRF→新MCIP' if use_wrf else '复用配套MCIP')+'"]; c[label="'+('原始NEI油气→六步SMOKE' if use_nei else '复用油气点源')+'"]; n[label="'+('原始非电力子集→六步SMOKE' if use_non_egu else '复用非电力点源')+'"]; d[label="'+('剖面→新ICON/BCON；参考默认单列' if use_profile else '复用配套初边界')+'"]; e[label="CCTM原生运行→读取核验→出图"]; f[label="面源及其他未选部门：复用"]; x[label="全区域原始清单链：未完成",style=dashed]; a->b; a->c; a->n; a->d; b->e; c->e; n->e; d->e; f->e; x->f[style=dashed]; }')
        if use_cmv:st.graphviz_chart('digraph { rankdir=LR; a[label="匹配船舶年度＋原始逐小时子集"]; b[label="校验→六步SMOKE→逐小时已审计合同"]; c[label="替换第四点源流→CCTM→出图"]; x[label="全区域船舶清单：未完成",style=dashed]; a->b->c; x->a[style=dashed]; }')
    st.info('当前转换不匹配会拒绝。未提供转换包时，将从此新案例的原始转换输入实际重新转换。')
    if st.button('运行所选新案例', key=model+'_independent_start', type='primary'):
        try:
            extra = {}
            if use_cmv:
                if raw_cmv_inventory is None or raw_cmv_hourly is None:raise ValueError('船舶路线必须同时上传匹配的年度及小时清单；不会回退原船舶部门')
                extra['raw_cmv_inventory']=raw_cmv_inventory
                extra['raw_cmv_hourly']=raw_cmv_hourly
            if use_non_egu:
                if raw_non_egu_inventory is None:raise ValueError('已选择非电力原始NEI路线，必须上传对应清单；不会回退原部门')
                extra['raw_non_egu_inventory']=raw_non_egu_inventory
            if use_nei:
                if raw_oilgas_inventory is None:raise ValueError('已选择原始NEI路线，必须上传清单；不会回退原点源部门')
                extra['raw_oilgas_inventory']=raw_oilgas_inventory
            if use_wrf:
                if not meteorology_files:
                    raise ValueError('已选择上传WRF路线，必须上传WRF文件；不会回退配套气象')
                extra['meteorology_files'] = meteorology_files
            if model == 'cmaq' and route == 'rebuild' and use_profile:
                if chemical_profile is None:
                    raise ValueError('已选择完整化学剖面路线，必须上传完整CSV；不会回退配套初边界')
                extra['chemical_profile'] = chemical_profile
                if accept_reference_defaults:
                    extra['accept_reference_defaults'] = True
            ident = start(model, route, converted_zip, mode, **extra)
            st.session_state[model+'_independent_job'] = ident
        except Exception as exc:
            st.error(str(exc))
            return True
    ident = st.session_state.get(model+'_independent_job')
    if not ident:
        return True
    @st.fragment(run_every=3)
    def show():
        record = status(ident); folder = directory(ident)
        st.write('本次任务：'+ident+'；实际路线：'+record['route'])
        if record['state'] == 'running':
            st.info('本次原生任务运行中；四模型串行执行。')
            stages=folder/'stages.json'
            if stages.is_file():
                try:
                    st.dataframe(json.loads(stages.read_text()),hide_index=True)
                except (OSError,ValueError):
                    st.caption('阶段记录更新中。')
        elif record['state'] == 'failed':
            st.error(record['error'])
        else:
            st.success('所选新案例原生运行及输出读取完成；适用范围见上方说明。')
            if record.get('accept_reference_defaults'):
                st.warning('本次为显式参考背景情景，五字段使用原生最小值；不代表完整背景资料已齐或当地精度通过。')
                if (folder/'native_defaults.json').is_file():
                    st.json(json.loads((folder/'native_defaults.json').read_text()))
            if record.get('native',{}).get('full_inventory_adaptation') is False:
                native=record['native'];coverage=native.get('inventory_coverage',{})
                missing=sorted(set(n for kind in ('Area','Point') for n in coverage.get(kind,{}).get('not_modeled',[]))) or native.get('unmapped',[])
                st.warning('原生运行通过，但完整清单适配仍未验收；未建模/未确认的物种：'+', '.join(missing)+'。不能把此结果写成全部清单物种已通过。')
            st.image(str(folder / 'result.png'))
            st.download_button('下载本次结果CSV', (folder/'result.csv').read_bytes(), model+'_independent.csv', key=model+'_independent_csv')
            st.download_button('下载本次图表PNG',(folder/'result.png').read_bytes(),model+'_independent.png',key=model+'_independent_png')
        with st.expander('本次状态、来源与范围'):
            st.json(record)
        if (folder/'runner.log').exists():
            with st.expander('本次原生日志'):
                st.code((folder/'runner.log').read_text(errors='replace')[-10000:])
    show()
    return True
