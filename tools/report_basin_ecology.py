"""Evidence-based delivery; no post-test change to selection or algorithms."""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
from pathlib import Path
from collections import Counter
from importlib.metadata import version
import json,hashlib,shutil,zipfile
import numpy as np
import xarray as xr
from run_basin_ecology import ROOT,OUT,DATA,SEEDS,load
from evaluate_basin_ecology import labels
from ecodata.algorithms.numerical_search import save


def main():
    audit=load(OUT/'independent_audit.json');assert audit['status']=='PASS'
    effects=load(OUT/'effect_summary.json');single=load(OUT/'single_method_comparison.json');assert single['independent_replay']=='PASS'
    qa=load(DATA/'quality_and_units.json');costs=load(OUT/'execution_costs.json');controls=load(OUT/'controls_frozen.json')
    names=dict(random='平台随机变异',none='同初始化、不变异',development_fixed='开发共同最强固定',prior='仅训练期FVC气候态',
        nearest='固定最近邻恢复',linear='固定线性恢复',forest_all='固定全协变量森林')
    precision=[]
    for entry in load(OUT/'predictions_frozen.json'):
        with xr.open_dataset(entry['path']) as f:
            reference=labels(entry['case'],'test');prediction=f.fvc.sel(time=reference.time,lat=reference.lat,lon=reference.lon).values.astype('float64')
        value=float(np.sqrt(np.mean((prediction-reference.values.astype('float64'))**2)))
        original=next(r for r in load(OUT/'test_metrics.json') if r['case']==entry['case'] and r['method']==entry['method'])['objective']
        precision.append(dict(case=entry['case'],method=entry['method'],float64_RMSE=value,reported_RMSE=original,difference=value-original))
    save(OUT/'float64_precision_recheck.json',precision)
    structures=[];operators=Counter();algorithms=Counter();forest_origins=Counter()
    for folder in (OUT/'runs/validation').iterdir():
        meta=load(folder/'metadata.json');summary=load(folder/'summary.json');records=load(folder/'graphs.json');seen=set()
        def visit(key):
            if key in seen:return
            seen.add(key)
            for p in records[key]['parents']:visit(p)
        visit(summary['best']['key'])
        ops=Counter(records[k]['operation'] for k in seen if records[k]['operation']!='source')
        feature_sets=[records[k]['parameters']['grid_predictor_features'] for k in seen if 'grid_predictor_features' in records[k]['parameters']]
        structures.append(dict(**meta,selected_nodes=sum(ops.values()),operations=dict(ops),selected_feature_sets=feature_sets))
        if meta['control']=='random':
            for t in load(folder/'mutation_trace.json'):operators[t['family']]+=1
        origins={r['key']:r['origin'] for line in (folder/'attempts.jsonl').read_text().splitlines() if (r:=json.loads(line))}
        for e in load(folder/'total_cost.json')['events']:
            algorithms[e['algorithm']]+=1
            if 'prior_grid_forest' in e['algorithm']:forest_origins[origins.get(e['graph'],'unresolved')]+=1
    save(OUT/'selected_structure_audit.json',structures)
    lines=['# 小流域七源生态重建实验交付（2026-10-28）','',
      '本轮由平台公共PortraitMutation入口自行筛选、连接、随机变异、真实执行和选图；没有向主搜索输入手写完整候选池、指定胜出路线或人工指定协变量搭配。固定路线仅用于独立对照。','',
      f"事实：39项定向回归、6次真实开发入口验收通过；18次正式运行、33份主预测及9份单方法对照完成。独立重放、输入/源码哈希、训练目标隔离、单位与类别比例检查通过。",'',
      '## 实际效果','',
      '主指标为2017年被遮的同一批FVC参考产品像元RMSE，单位为0—1比例；越低越好。','',
      '| 方法 | RMSE | MAE | Bias |', '|---|---:|---:|---:|']
    for key,row in effects['results'].items():lines.append(f"| {names[key]} | {row['RMSE']:.8f} | {row['MAE']:.8f} | {row['Bias']:.8f} |")
    lines+=['',f"最佳已发现单方法对照RMSE={single['single_mean_RMSE']:.8f}；随机最终图相对它改善{single['improvement_percent']:.6f}%（负数为退步）。此对照只从各次已经真实执行的零/单科学节点图中按2016验证误差选择，并在测试前封存，不增加主搜索预算。",'',
      '| 随机变异相对对照 | RMSE改善 | 赢空间块/3 |','|---|---:|---:|']
    for key,row in effects['comparisons'].items():lines.append(f"| {names[key]} | {row['improvement_percent']:.6f}% | {row['block_wins']} |")
    lines+=['',
      '判断边界：对不变异的改善只说明搜索是否增益；对固定单方法和最佳已发现单节点的改善才提供更直接的组合/选优证据。不能把特征子集变化或图更长本身叫作多算法协同。极小末位差异按基本持平解释，不当作有实际意义的提升。',
      f"独立float64精度敏感性复算与原指标的最大绝对差为{max(abs(r['difference']) for r in precision):.3g}；原搜索和选图保留原样，没有据此重新挑图。完整明细见float64_precision_recheck.json。",'',
      '## 数据、单位与质量','',
      '研究区为黄山附近HydroBASINS出口4120774820的完整上游集水区，6个第12级子区共1056.6平方公里。出口局部区仅204.2平方公里，未拿局部区冒充完整流域。',
      '公开数据共同年份为2013—2017。已下载740份原生卫星/高程窗口、5年CLCD安徽省图和4份气候空间子集。原始请求、版本、QC、网格变换和校验均保留；首版签名请求触发429的日志保留，已复用官方SDK的容器级令牌缓存，未更换来源或伪造缺失。','',
      '| 数据 | 原生尺度 | 单位/处理 |','|---|---|---|',
      '| FVC代理 | Landsat 30m | QA去云影雪填充及饱和；至少2次有效观测，5—10月NDVI中位数；端元仅2013—2014拟合，结果限定0—1 |',
      '| 土地覆盖 | CLCD 30m | 类别代码单位1；自动转为类别面积比例，不求代码均值 |',
      '| DEM/坡度 | NASADEM约30m | m/度；先在30m米制网格算坡度，再聚合 |',
      '| LAI | Terra MOD15A2H V061，500m/8日 | m²/m²；SCF_QC 0/1且云状态清晰，取5—10月合格观测均值 |',
      '| NPP | Terra MOD17A3HGF V061，500m/年 | kg C m⁻² yr⁻¹；不冒充总碳量，保留gap-fill百分比 |',
      '| 降水/气温 | Peng中国1/120度逐月 | 源0.1mm/0.1℃按元数据换算；5—10月降水累加、气温按月天数加权 |','',
      f"目标为气候产品原生1/120度中心（约1km），不是把所有变量宣称为新增1km信息。流域内{qa['basin_cells']}格，五年均有合格FVC参考的{qa['common_finite_FVC_cells']}格。只按几何与质量选择3个互不重叠8×8块；每块64格，共192个测试参考格，搜索种子不扩增独立样本数。", 
      'FVC是Landsat派生代理，不是独立地面实测覆盖度。CLCD与FVC共享Landsat来源，NPP与LAI共享生产链；本轮是产品缺口恢复，不是独立传感器精度、云下真实FVC或生态变化归因实验。NPP为全年辅助量，与5—10月目标时段的差别保留，不是未来预测。', '',
      '## 平台自己做了什么','',
      '公共接入模块自动执行产品QA、单位和连续/分类语义准备；这些是共同测量契约，不是预设胜出路线。主搜索只接收遮挡后的FVC、训练期气候态、带独立单位的辅助变量和仅训练期非空的training_target。',
      '2013—2014拟合；2015开发与选固定；2016验证选图；2017最后测试。平台从实际注册算法接口发现候选，通过动态参数提议探索任意可用协变量子集；随机执行算法替换、参数、增删、重连和联合破坏—修复。每个完整候选执行真实数值管线后才能得到RMSE。无手写整图池、无最大图深度预设、无GOLEM/ALNS替换。',
      '栅格任务不能冒用逐日点序列响应探测；该探测明确UNKNOWN。保留三画像契约/状态筛选，真实选优依据隔离验证误差。任务契约明确为FVC产品重建，没有冒名套用WRF/SWAT需求，也没有声称学习了下游模型敏感性。',
      '森林在这里是把带各自单位的辅助数据估计为FVC的候选科学处理节点，不是另加一个算法排序模型来替换既定选择机制。CLCD是土地覆盖分类，不等于精细土地利用或管理措施。',
      f"关键负结果：正式运行中的辅助变量森林执行次数及阶段为{dict(forest_origins)}；9个随机最终图有{sum(r['control']=='random' and bool(r['selected_feature_sets']) for r in structures)}个保留了它。因此本轮小幅改善来自FVC处理组合，不能包装成七变量联合融合收益。数据和候选预测端口已经接通，但本次并未充分探索其后续跨源分支潜力。", 
      '部分胜出图仍含重复插值操作。精确状态去重不等于消除所有近似冗余，也没有逐节点证明其贡献；本轮保持原搜索和负结果，不在测试后修图追优势。',
      f"实际随机修改计数：{dict(operators)}。最终逐例算法、特征子集和科学节点见selected_structure_audit.json；原始边和执行轨迹见各run目录。",'',
      '## 成本、负结果和适用范围','',
      '| 组别 | 平均注册调用 | 平均秒 | 最长秒 | 训练能力调用 | 最终节点均值 |','|---|---:|---:|---:|---:|---:|']
    for control in ['random','none']:
        rows=[r for r in costs if r['control']==control]
        lines.append(f"| {names[control]} | {np.mean([r['calls'] for r in rows]):.2f} | {np.mean([r['seconds'] for r in rows]):.2f} | {max(r['seconds'] for r in rows):.2f} | {sum(r['training_capable_calls'] for r in rows)} | {np.mean([r['nodes'] for r in rows]):.2f} |")
    lines+=['',
      '每次统一上限180新增注册算法调用、40000提议和120秒软墙钟；缓存不算新调用，失败和部署均入账。已进入的内核无法中断，超过120秒的实际耗时公开保留，没有补种子或延长预算追优势。不变异实际调用更少，不能宣称两组实际耗时相同。一次RF注册调用内部包含中位数拟合和50棵树训练，耗时完整计入，但同调用数不等于同FLOPs。',
      f"开发固定对照从全部{controls['discovered_recipes']}个已发现配方跨三案例真实重放选出，不是全库穷举；另耗{controls['replay_adapter_calls']}次注册调用、{controls['seconds']:.2f}秒。最佳单方法部署另{single['deployment_calls']}次/{single['deployment_seconds']:.2f}秒。独立主审计另{audit['independent_replay_adapter_calls']}次/{audit['audit_seconds']:.2f}秒。公共数据下载、接入整理为共同一次性开销，未作精确统一计时，不拿它声明效率优势。", 
      '所有失败、中间状态、缓存、未胜出候选、低收益和不利案例保留。只有一个流域和一个测试年，三个空间块仍有相关性；不做强显著性或普遍泛化结论。没有穷举最优或全局最优认证。本轮也未做新生态任务网页表单的浏览器验收，已验证的是公共数据模块与公共文件/API入口。','',
      '## 交付与复用','',
      '- 原始数据和完整气候压缩包保存在data/basin_ecology_2013_2017；prepared_v1包含标准化七源数据、原生30m FVC/DEM/坡度、质量与单位清单。',
      '- output/basin_ecology_2013_2017保存配置、冻结源码、输入哈希、全部真实候选、预测、控制组和独立审计。',
      '- 公共新增适配器为raster_gap_prediction.py；任务契约为application_contract.py；公共数据接入为io/ecological_sources.py。原随机变异与图执行器复用，不向研究脚本搬一套搜索器。',
      '- 实验包含标准化/必要原生生态变量与实验全部证据；大体量原始Landsat反射率、全国气候压缩包、亚洲全量流域ZIP另存本机，不重复塞入交付ZIP，来源/哈希仍在清单。','',
      '来源：',
      '- HydroBASINS：https://www.hydrosheds.org/products/hydrobasins',
      '- Landsat：https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-reflectance',
      '- CLCD：https://zenodo.org/records/8176941',
      '- LAI：https://lpdaac.usgs.gov/documents/926/MOD15_User_Guide_V61.pdf',
      '- NPP：https://modis-land.gsfc.nasa.gov/pdf/MOD17C61UsersGuideV11Mar112021.pdf',
      '- 降水/气温：https://zenodo.org/records/3114194 、 https://zenodo.org/records/3185722',
      '- FVC代理方法参考：https://www.mdpi.com/2071-1050/16/23/10430','']
    report=ROOT/'docs/小流域七源生态重建_实验交付_20261028.md';report.write_text('\n'.join(lines),encoding='utf-8')
    packages=['numpy','pandas','xarray','scipy','scikit-learn','rasterio','geopandas','planetary-computer','pystac-client','h5netcdf']
    save(OUT/'environment.json',{name:version(name) for name in packages})
    requirements=ROOT/'requirements-basin-data.txt';requirements.write_text('# Additional public-data acquisition clients; existing project numerical dependencies remain required.\nplanetary-computer==1.0.0\npystac-client==0.9.1\npystac==1.15.2\n',encoding='utf-8')
    master=ROOT/'docs/EcoData Engine项目总交付报告.md';heading='## 第68章 小流域七源生态重建与自主随机变异'
    if heading not in master.read_text(encoding='utf-8'):
        shutil.copy2(master,OUT/'master_before_ch68.md');r=effects['results'];d=effects['comparisons']
        text=f"\n\n{heading}（2026-10-28）\n\n完成黄山附近HydroBASINS出口4120774820完整上游1056.6km²的2013—2017七类数据下载，740份卫星/高程原生窗口、5年CLCD和4份气候空间子集。公共接入核对0.1mm/0.1℃转换、云/饱和/LAI/NPP QC、坡度先细算与类别面积比例。FVC为Landsat训练端元代理，不是地面真实覆盖度。\n\n"
        text+='公共入口新增通用栅格缺口/先验协变量预测适配与明确应用任务契约，复用既有SciPy、feature_forest与随机变异/图去重/预算执行。主搜索没有手写完整候选池或固定特征组，节点、特征子集、参数、边和最终图由平台自己选择。栅格不适用点序列探测时保持UNKNOWN；不冒用WRF/SWAT敏感性。39项回归、6次开发和18次正式搜索通过。\n\n'
        text+=f"2013—2014拟合、2015开发、2016选图、2017测试；3个互不重叠8×8块、3种子。33份主预测和9份验证选出的最佳已发现单方法预测均先封存再测试。FVC RMSE：随机{r['random']['RMSE']:.8f}、不变异{r['none']['RMSE']:.8f}、开发固定{r['development_fixed']['RMSE']:.8f}、全协变量森林{r['forest_all']['RMSE']:.8f}；随机对不变异/固定改善{d['none']['improvement_percent']:.6f}%/{d['development_fixed']['improvement_percent']:.6f}%，对最佳已发现单方法改善{single['improvement_percent']:.6f}%（负数为退步）。不把末位数值噪声、选到单算法或网格对齐称为组合创新。\n\n"
        text+='全部候选真实执行，完整失败/负结果、成本和软墙钟超限保留；独立重放、源码/输入哈希、训练目标隔离和单位审计通过。单流域单测试年，不支持普遍优势或全局最优；未运行生态/水文原生模型。详见docs/小流域七源生态重建_实验交付_20261028.md及output/basin_ecology_2013_2017。\n'
        with master.open('a',encoding='utf-8') as f:f.write(text)
    marker='补充证据边界（本轮七变量）：'
    if marker not in master.read_text(encoding='utf-8'):
        with master.open('a',encoding='utf-8') as f:
            f.write('\n'+marker+f"辅助变量森林实际执行阶段/次数为{dict(forest_origins)}，9个随机最终图均未保留森林；不能把本次FVC处理组合改善叫作七变量联合融合收益，也未充分探索全部跨源分支。胜出图仍有重复插值操作，未逐节点证明贡献；保留原结果，不按测试再修图。\n")
    data_root=ROOT/'data/basin_ecology_2013_2017'
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_receipt.json' and not any(x.startswith('tests_temp') for x in p.parts)]
    files+=[p for p in data_root.rglob('*') if p.is_file() and p.suffix not in {'.zip','.part'} and not p.name.endswith('_temporary.nc')
            and not ('landsat' in p.parts and p.suffix=='.tif')]
    files+=[report,master,requirements,ROOT/'docs/小流域七源生态重建_冻结协议_20261028.md']
    files+=[ROOT/'tools'/n for n in ['acquire_basin_ecology.py','run_basin_ecology.py','evaluate_basin_ecology.py','audit_basin_ecology.py','basin_single_method_control.py','report_basin_ecology.py']]
    files+=[ROOT/'tests/test_raster_gap_application.py']
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    path=ROOT/'output/小流域七源生态重建_实验交付_20261028.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in hashes:z.write(ROOT/name,name)
        z.writestr('PACKAGE_SHA256.json',json.dumps(hashes,indent=2))
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in hashes.items())
    save(OUT/'delivery_receipt.json',dict(status='PASS',zip=str(path),bytes=path.stat().st_size,files=len(hashes),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),all_zip_members_verified=True,bulk_originals_remain_local=True))
    print('DELIVERY VERIFIED',path,len(hashes),flush=True)


if __name__=='__main__':main()
