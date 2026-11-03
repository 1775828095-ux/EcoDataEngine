"""Report audited results without turning exhaustive certification into mutation wins."""
from importlib.metadata import version
from pathlib import Path
import hashlib,json,shutil,xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/common_reference_20261026'


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    r=load(OUT/'results.json');audit=load(OUT/'independent_audit.json');entry=load(OUT/'delivery/acceptance.json');replay=load(OUT/'thread_replay_acceptance.json')
    assert audit['status']==entry['status']==replay['status']=='PASS'
    tests=set()
    for filename in ['common_reference_tests_20261026.xml','common_reference_compatibility_20261026.xml']:
        tree=ET.parse(ROOT/'output'/filename)
        for test in tree.iter('testcase'):
            assert test.find('failure') is None and test.find('error') is None
            tests.add((test.attrib.get('classname'),test.attrib['name']))
    names=dict(common='共同强基准＋ALNS',legacy='原协同加权＋ALNS',uniform_alns='随机初始化、无三路软评分ALNS',random='纯随机')
    lines=['# 共同强基准与变异预算实验交付（2026-10-26）','',
        '公共机制修改、真实执行与独立验收均已完成。区分初始选图、后续变异和独立穷举，不将三者混为算法优势。','',
        '## 修改内容','',
        '复用composition_scoring、Composition、ALNS和真实验证反馈。共同基准由开发冻结的最强单算法与最强固定组合构成。前者在原8个开发案例上真实执行线性时间插值、粗源双线性和PCHIP三个单源对照选择；后者复用原8开发案例、全1296图的独立审计结果。当前可见遮挡上选择这两个参照中较强者，对全部候选使用相同参照。','',
        '协同增益为共同参照RMSE减候选RMSE，并提供有界显示。数据/模型主评分各0.5；协同不再线性加分，只给出两个可见窗口是否均改善的判断，主评分相同时才参与比较。因此不能再用较弱自身分支的相对改善抵消较差主评分。未知维持区间；可见窗口一致改善不等于测试置信保证。','',
        '新增模式要求parallel_scoring.common_reference的path及sha256，并要求collaboration加权系数为0。旧机制仅保留作兼容和对照；交付的新请求已显式启用共同基准模式。模型项仍为有声明的工程输入偏好，未运行WRF。','',
        '## 冻结设计','',
        '4组新验证—测试配对，两个东部区域均未参与权重拟合，起始2000/2700及后续2070/2770，旧测试未冒充新测试；同源NCEP受控互补分辨率。每例33个六小时时次，细源5个48小时锚点，粗源全时次，28隐藏细网格时次。候选1296种，不是整个算法库所有可能图。','',
        '每法5种子，同一有界轨迹分析32/128/256次唯一真实执行前缀。最大4096提议/1800秒，失败及提前停止保留。前缀是嵌套比较，不是独立重复。所有前缀选图先冻结，再运行独立1296全域参照，之后才评价测试。','',
        '正式变异及随机运行全部exact_check=False；没有使用穷举补齐、穷举误差表或最优知情停止。前2次命中属于初筛，之后首次命中才记为变异。共同机制额外有72局部候选探测与4参照探测；不冒充零成本软规则。','',
        '## 验证最优命中率','',
        '| 方法 | 32次预算 | 128次预算 | 256次预算 |',
        '|---|---:|---:|---:|']
    for method,table in r['summary'].items():
        lines.append('| '+names[method]+' | '+' | '.join(f'{table[str(b)]["hits"]}/{table[str(b)]["runs"]}' for b in [32,128,256])+' |')
    lines+=['','256次预算的命中来源：','',
        '| 方法 | 初始两图命中 | 后续变异首次命中 | 未命中 | 测试平均RMSE/K |',
        '|---|---:|---:|---:|---:|']
    for method,table in r['summary'].items():
        s=table['256'];lines.append(f'| {names[method]} | {s["initial_hits"]} | {s["mutation_hits"]} | {s["runs"]-s["hits"]} | {s["mean_test_rmse"]:.6f} |')
    lines+=['',f'开发最强固定方案的测试平均RMSE：**{r["mean_fixed_rmse"]:.6f} K**。','',
        '如果共同机制在初始两图已命中，后续预算不能算作变异新收益。随机初始化ALNS的后续命中才提供更直接的变异搜索证据。每个案例5种子的分组命中见results.json；20次运行只覆盖4组时空案例，不代表普适收敛或无条件全局最优保证。','',
        '## 验收与复现','',
        f'- {len(tests)}项不同定向/兼容测试通过。',
        f'- 独立重算{audit["numeric_products"]}份开发之外的完整图数值误差与{audit["test_products"]}份不同测试产品。',
        f'- 独立核对{audit["reference_probes"]}份共同参照探测和{audit["common_scores"]}条候选评分，确认同请求参照一致，协同不抵消主评分。',
        '- 核对所有运行无exact_audit执行，逐前缀重算最优命中，并单列初筛与变异。',
        '- 文件、页面处理器及实际CLI无隐藏参考输出同一产品；不声明浏览器逐控件验收。','',
        '运行资源调整：发现每进程默认24个BLAS线程叠加案例并行，改用任务进程内单线程，并根据32逻辑处理器、约19GB空闲内存采用12个独立案例进程。未修改公式、候选、参数、预算或种子；6个已完成运行保留，未完成轨迹单独保留后按相同种子继续。6个原运行另做全轨迹重放，评价顺序、提议和选图一致，数值差异见thread_replay_acceptance.json。重放不计入正式命中样本。不会用混合资源下的墙钟宣称算法加速。','',
        '正式源代码按source_manifest.json冻结；全部产品、失败日志、runtime_partial及runtime_partial_workers保留。初次准备脚本的语法错误记录也保留，发生在科学协议冻结前。','',
        '无隐藏参考请求：delivery/request_no_reference.json；带独立验证反馈的256次预算请求：delivery/request_validated_search.json。','',
        '```powershell',"Set-Location -LiteralPath 'D:/codex/EcoDataEngine'","$env:PYTHONPATH = 'D:/codex/EcoDataEngine/src'","$env:OPENBLAS_NUM_THREADS = '1'","$env:OMP_NUM_THREADS = '1'",'./.venv/Scripts/python.exe -m ecodata.algorithms.platform_search output/common_reference_20261026/delivery/request_validated_search.json output/common_reference_new_request','```','',
        '完整覆盖的参照只能确认这1296种图的验证RMSE最优，不能证明未知测试最优。本轮不声称独立传感器融合、WRF响应或所有算法与数据特征已覆盖。','']
    lines+=['## 结论的限度','',
        '这批4组新案例×5种子中，128次唯一执行已使三个变异组均20/20命中；256次没有再提升命中或测试精度。共同基准组5次在初始两图命中、15次在后续变异命中，不能把20次全部归于变异。随机初始化ALNS的20次均来自后续变异，为增加预算后的实际搜索作用提供更直接证据。','',
        '共同基准修复了相对弱分支加分的逻辑，但本轮没有证明其优于旧公式：旧公式32预算16/20，共同基准15/20；128和256预算两者都20/20。不能把预算收益归功于新公式。','',
        '4个验证案例的穷举最优仍是同一开发固定图；128预算后测试追平固定，未超过固定。当前任务没有证明按案例变化的组合更优，也不证明任意新任务在128次预算内必然达到全局最优。','',
        '局部responses.constituent_rmse保留原分支诊断；最终候选routes.collaboration/common_gain会以共同参照重新计算。不会因这些诊断字段存在，就把旧自身基准奖励重新加回主评分。','']
    if audit.get('oracle_receipts'):
        lines+=['## 穷举复用账本','',
            '穷举参照在选择冻结之后，从相同输入哈希、数值需求和处理图的既有真实执行中复用产品，逐份验证文件哈希并重算误差，再单独执行尚未覆盖的图。覆盖全部候选的保证属于穷举，不回写或扩展正式变异轨迹。','',
            '| 案例 | 复用真实产品 | 仅穷举补齐 | 总覆盖 |','|---|---:|---:|---:|']
        for x in audit['oracle_receipts']:lines.append(f'| {x["case"]} | {x["reused_real_products"]} | {x["new_oracle_only_executions"]} | {x["total"]} |')
    report=ROOT/'docs/共同强基准与变异预算_实验交付_20261026.md';report.write_text('\n'.join(lines),encoding='utf-8')
    master=ROOT/'docs/EcoData Engine项目总交付报告.md';text=master.read_text(encoding='utf-8')
    title='## 第59章 共同强基准与变异预算（2026-10-26）';assert title not in text
    content='\n\n'+title+'\n\n共同强基准模式已接公共评分器：开发冻结最强单算法/固定组合，可见遮挡选择统一参照；协同不再线性加分，只作收益判断和同主评分下比较。\n\n'
    for method,table in r['summary'].items():
        content+=names[method]+'的32/128/256预算验证最优命中分别为'+ '/'.join(str(table[str(b)]['hits']) for b in [32,128,256])+'/20；256预算初筛/后续命中为'+str(table['256']['initial_hits'])+'/'+str(table['256']['mutation_hits'])+'。\n\n'
    content+=f'全程正式搜索无穷举补齐；4组新配对×5种子，测试隔离。{len(tests)}项测试及独立产品/评分审计通过。资源调整、未完成轨迹、重放与负结果保留，不使用穷举结果冒充变异优势。详见共同强基准与变异预算_实验交付_20261026.md及output/common_reference_20261026。\n'
    content+='128次预算后三个变异组均20/20，测试均追平开发最强固定0.828483K。新共同基准组5次初筛、15次变异；没有证明新公式优于旧公式，预算增加的收益不能归于新公式。4个新验证案例最优仍为同一固定图；不作普适收敛或个性化组合优越性宣称。\n'
    master.write_text(text+content,encoding='utf-8')
    released=[]
    paths=[Path(x['path']) for x in load(OUT/'source_manifest.json')]
    for x in load(OUT/'source_manifest.json'):assert sha(Path(x['path']))==x['sha256']
    paths += [ROOT/'tools'/n for n in ['audit_common_reference_study.py','accept_common_reference_entry.py','run_common_reference_workers.py','replay_common_thread_change.py','deliver_common_reference_study.py','build_common_reference_oracle.py']]
    paths += [ROOT/'tests/test_common_reference_scoring.py',ROOT/'src/ecodata/interface/search_page.py']
    for p in paths:
        q=OUT/'source_release'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);released.append(dict(path=str(p),sha256=sha(p)))
    evidence=[dict(path=str(OUT/name),sha256=sha(OUT/name)) for name in ['protocol.json','common_reference.json','frozen_selections.json','results.json','independent_audit.json','thread_replay_acceptance.json','delivery/acceptance.json','runtime_thread_change.json','runtime_worker_change.json']]
    save=dict(status='PASS',unique_tests=len(tests),released_sources=released,evidence=evidence,report=dict(path=str(report),sha256=sha(report)),
        numeric_products=audit['numeric_products'],test_products=audit['test_products'],
        scientific_sources_unchanged=True,mutation_excludes_exhaustive_completion=True,
        package_versions={p:version(p) for p in ['numpy','scipy','xarray','alns','h5netcdf','pytest']})
    (OUT/'final_assurance.json').write_text(json.dumps(save,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',report=str(report),tests=len(tests),summary=r['summary'])),flush=True)


if __name__=='__main__':main()
