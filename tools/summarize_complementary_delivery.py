"""Read back frozen evidence; do not run or tune scientific decisions."""
from pathlib import Path
import json,hashlib,statistics,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/complementary_20261024'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    result=load(OUT/'study_v3/results.json');audit=load(OUT/'study_v3/independent_audit.json');accept=load(OUT/'delivery/acceptance.json')
    assert audit['status']==accept['status']=='PASS'
    clients=[load(OUT/'delivery'/p/'summary.json') for p in ('file_entry','page_entry','cli_entry')]
    assert len({r['recommended_key'] for r in clients})==len({r['best']['numeric_fingerprint'] for r in clients})==1
    for row in clients:
        path=Path(row['best']['product']);assert hashlib.sha256(path.read_bytes()).hexdigest()==row['best']['product_sha256']
        assert not row['reference_used_for_selection'] and row['native_optimizer']['package']=='alns'
    testnames=set();checks=0
    for name in ('regression.xml','response_tests.xml','compatibility.xml'):
        suite=ET.parse(OUT/name).getroot()
        for item in suite.findall('.//testsuite'):assert int(item.attrib['errors'])==int(item.attrib['failures'])==0
        for case in suite.findall('.//testcase'):testnames.add((case.attrib['classname'],case.attrib['name']));checks+=1
    m=result['mean_rmse'];cost={}
    for backend in ('alns','basic','random','none'):
        rows=[r for r in result['runs'] if r['backend']==backend]
        cost[backend]=dict(mean_evaluations=statistics.mean(r['evaluations'] for r in rows),median_seconds=statistics.median(r['seconds'] for r in rows),all_pass=all(r['evaluations']==r['passes'] for r in rows))
    final=dict(status='PASS',unique_tests=len(testnames),test_executions=checks,rmse_K=m,cost=cost,
        improvement_over_original_score_percent=100*(1-m['alns']/m['original_score']),
        improvement_over_coarse_percent=100*(1-m['alns']/m['coarse_fast']),
        improvement_over_fixed_percent=100*(1-m['alns']/m['fixed']),
        audit=audit,public_clients_equal=True,reference_supplied=False,
        scope='4 held-out dates, 5 seeds each; controlled real-data downsampling; no WRF-response or mutation-superiority claim')
    (OUT/'final_summary.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
    document=f'''# 互补时空分辨率融合：最终实现与验收（2026-10-25）

已在平台公共入口完成：**全部合法候选三画像评分 → 筛出初始组合 → 原生变异/邻域搜索 → 真实数值执行及质量门 → 独立评价**。不调用GOLEM优化器，不以另一个黑箱学习排序器替代三画像。

## 完成与选型

取消原先随机抽16种再打分。当前任务56种结构全部检查和评分，筛选超限明确停止，不随机降级。已有逐源链、先后依赖、主辅角色、汇合前后阶段、时间窗和物理接口检查继续复用。

接入官方 [ALNS](https://github.com/N-Wouda/ALNS) 7.0.0（MIT）的自适应破坏—修复、AlphaUCB算子选择及HillClimbing接受；可按源、融合/后处理、时间窗和全局轴修改，修复仍依据三画像。开发期也真实比较了 [pymoo](https://github.com/anyoptimization/pymoo) 0.6.2 MixedVariableGA（Apache-2.0，含交叉/变异）、本地变异和随机扩展。两者均为官方包，许可证及版本已保存。

开发同分时按事前工程规则选择ALNS，**没有证据证明它是最新最好、精度优于其他搜索器，或有限预算达到全局最优**。完整实验请求显式指定ALNS；兼容旧请求未指定后端时仍保留basic，不把本任务选型强推给所有任务。

## 实验

真实NCEP温度序列受控降采样：H为12×12、48小时，C为6×6、6小时；恢复中间7个时刻的12×12场。全部源自同一再分析，原分辨率约2.5°，因此是相对细/粗的互补源机制实验，**不是两个独立传感器，也不是原生高分辨率产品验证**。

2013四组开发、两组验证用于初轮选型；2014四组原测试保留。发现原细节代理偏好过强注入后，仅以原四组开发真实执行结果拟合误差二阶矩，并冻结新增权重与新四组2014日期（1600、1960、2320、2680），每组5种子，未用旧/新测试拟合。总候选56，搜索预算12次唯一真实评价、48提议；全部56图另作事后参考，不混入变异收益。

三画像新增的expected_error是**有适用范围的开发期误差响应估计**，非新测试误差。对线性混合采用 `a²MHH+(1-a)²MCC+2a(1-a)MHC`，保留偏差及相关误差。数据画像核对物理量、网格、单位、源时距和目标时间位置；算法画像提供各预处理/融合族矩阵；模型偏好控制风险权重。未知或超范围处理链保留UNKNOWN。新增权重8及风险尺度{result['protocol']['parameters']['bad']:.6f}由开发网格选择，不是人工填造的WRF敏感度。

## 新测试结果

| 方案 | 平均RMSE（K） |
|---|---:|
| 稀疏细源线性时间插值 | {m['fine_slow']:.6f} |
| 稠密粗源双线性空间插值 | {m['coarse_fast']:.6f} |
| 原三画像代理评分 | {m['original_score']:.6f} |
| 校准评分＋ALNS | {m['alns']:.6f} |
| 校准评分＋basic／随机扩展／仅初筛 | {m['none']:.6f} |
| 开发期最强固定组合 | {m['fixed']:.6f} |
| 每任务事后池内真实误差最优 | {m['oracle']:.6f} |

相对粗源降低{final['improvement_over_coarse_percent']:.2f}%，相对原评分降低{final['improvement_over_original_score_percent']:.2f}%。推荐与开发期最强固定组合持平：线性时间处理＋PCHIP空间处理＋0.5强度粗观测一致投影。**组合有收益、原评分得到改进；数据自适应优于最强固定及变异额外收益均未证明。** 四日期不是大规模泛化证据，五种子不能当20个独立数据组。

随机对照与其他变异器共享三画像初始方案，因此检验的是初筛后的随机扩展，并非完全无评分的随机选择器；不据此声称三画像优于一切随机推荐。仅初筛已选到本轮最终方案，变异增加执行成本而未降低RMSE。逐策略成本见final_summary.json，保留所有实际轨迹。

原轮负结果不撤改：原评分测试均值1.507611 K，落后开发期最强固定1.211714 K。未修改原测试成绩或质量门，也未继续按新测试成绩调参。

## 验收与证据

- {len(testnames)}项不同测试通过（{checks}次测试执行），包含画像权重改变推荐、未知范围、标签扰动隔离、各原生后端和原接口兼容。
- 原轮独立审计170次决策、2192份数值产品；新轮{audit['runs']}次决策、{audit['products']}份产品及224份开发产品独立核验通过。重算二阶矩、面积观测、评分算术、推荐次序、RMSE和SHA，不仅检查测试状态。
- 文件函数、页面处理函数及实际CLI均不传真值，选图与数值指纹一致。验收的是页面公共处理链，未声称自动操作浏览器逐控件验收。
- 已安装依赖并把不可变属性、模型及响应画像发布到data目录；页面可读同版本。核验配置在`output/complementary_20261024/delivery/request_no_truth.json`，页面补充请求在同目录`page_request.json`。请求中没有隐藏参考文件。

从项目目录复用已安装环境运行（输出目录须为新的）：

```powershell
$env:PYTHONPATH = 'D:/codex/EcoDataEngine/src'
./.venv/Scripts/python.exe -m ecodata.algorithms.platform_search output/complementary_20261024/delivery/request_no_truth.json output/complementary_new_request
```

完整证据：`output/complementary_20261024/final_summary.json`、`study_v2/results.json`、`study_v3/results.json`、`study_v3/frozen_protocol.json`、`study_v3/independent_audit.json`、`delivery/acceptance.json`、`delivery/source`。原始失败日志与审计脚本修正记录均保留。GitHub审查指纹在`frozen_protocol.json`，不把检索范围宣称全领域SOTA。

仍未证明：独立多源迁移、真实WRF响应、通用数据条件下更优的自适应选择、变异额外价值。算法库注册项含适配器，56只是本任务有限语法结构数，不是算法库全部可能组合数。平台功能链路通过本协议验收，不代表所有科学问题均已解决。
'''
    path=ROOT/'docs/互补分辨率融合_最终交付_20261025.md';path.write_text(document,encoding='utf-8')
    master=ROOT/'docs/EcoData Engine项目总交付报告.md'
    chapter='''

## 第56章 全量三画像初筛、原生变异接入与互补时空融合验收（2026-10-25）

本章修正第55章初筛时随机抽16种的逻辑：当前为全部合法候选先三画像评分，再进入变异及真实执行；超过完整筛选上限明确拒绝，不随机截断。复用逐源、阶段、时段及协调公共入口，新增显式粗观测一致的线性残差投影融合。新请求接入官方ALNS7/AlphaUCB，并真实比较pymoo MixedVariableGA；不使用GOLEM优化器。开发同分依事前工程规则暂选ALNS，不构成算法优越性或全局最优证明。

NCEP真实温度受控降采样形成12×12/48小时与6×6/6小时互补源，恢复隐藏细场，56结构全部评分及真实执行穷举参考；非独立传感器或WRF验证。原轮评分推荐RMSE1.507611K落后开发最强固定1.211714K，完整保留。之后仅原四组开发真实执行数据校准算法画像误差二阶矩，继续三画像风险差联合评分，未知范围不填造，不另训替代排序器。冻结新四组2014日期及5种子后测试：校准推荐1.189751K，原评分1.361006K，粗源1.881679K，细源5.076219K；相对原评分降低12.58%、相对粗源降低36.77%。开发最强固定也为1.189751K，池内事后最优1.175612K；ALNS、basic、随机扩展及仅初筛相同。因此仅支持组合收益及本任务评分修正有效，未证明优于最强固定或变异额外收益。随机扩展共享画像初始方案，不冒充完全无评分随机推荐。

最终143项不同定向及兼容测试通过（145次执行）。原轮独立审计170次决策2192份产品；新轮96次决策1040份产品及224份开发产品二阶矩独立核验通过。文件、页面处理函数及实际CLI无参考真值选图/数值指纹相同；未宣称浏览器逐控件或原生下游模型运行验收。不可变画像已发布data目录，直接可运行配置与源码指纹在output/complementary_20261024/delivery。完整结论和边界见docs/互补分辨率融合_最终交付_20261025.md；冻结协议、全部负结果与审计均保留。第53章及更早冻结池证据限制不变。
'''
    text=master.read_text(encoding='utf-8')
    assert '## 第56章' not in text,'Do not overwrite an existing report chapter'
    master.write_text(text+chapter,encoding='utf-8')
    print(json.dumps(final,ensure_ascii=False))

if __name__=='__main__':main()
