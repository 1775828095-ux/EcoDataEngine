"""Three independent acceptance dimensions, with a bounded frozen generic pool."""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import xarray as xr

from ecodata.algorithms.registry import create_default_registry

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output/capability_final_acceptance"


def read(name):return json.loads((OUT/name).read_text(encoding="utf-8"))


def main():
    tests={}
    for name in ("targeted_tests.xml","aqmbc_official_tests.xml","full_regression.xml"):
        suites=ET.parse(OUT/name).findall(".//testsuite")
        item={k:sum(int(s.get(k,0)) for s in suites) for k in ("tests","failures","errors","skipped")}
        assert item["tests"] and not item["failures"] and not item["errors"], item
        item["passed"]=item["tests"]-item["skipped"]
        tests[name]=item
    model=read("model_execution_manifest.json")
    for case in model:
        path=OUT/case["output"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==case["sha256"]
        with xr.open_dataset(path,decode_times=False) as ds:
            assert ds.sizes
        for part in case["report"]["quality_control"].values():
            if "output" in part:assert part["output"]["status"] in {"PASS","WARN"}
            else:assert all(v["output"]["status"] in {"PASS","WARN"} for v in part.values())
    baseline=json.loads((ROOT/"output/algorithm_library_expansion/v2_delivery_check.json").read_text(encoding="utf-8"))["golem_sha256_unchanged"]
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in baseline.items())
    with (ROOT/"output/algorithm_library_expansion/v5/C1_C30_coverage.csv").open(encoding="utf-8-sig",newline="") as f:
        old=list(csv.DictReader(f))
    names=["连续场重构","分类重采样","类别面积比例","规则网格守恒","学习型重构","站点成场","地形气象处理","二维经纬度重采样","时间插值","时间统计聚合","累计量/速率区间","极值校正","不规则时间","垂直插值","土壤层深转换","物理范围门","适用积分守恒","粗尺度一致性","多源融合","偏差校正","观测残差融合","质量感知融合","辅助信息编码","排放空间分配","排放时间分配","化学机制物种映射","初始条件生成","边界条件生成","土壤/土地利用模型映射","作物/管理模型映射"]
    evidence={4:"regular_conservative_real_era5.nc;targeted_tests.xml",9:"instantaneous_time_real_era5.nc;targeted_tests.xml",
        11:"interval_rate_control.nc;targeted_tests.xml",16:"physical_violation_expected_block.json;targeted_tests.xml",
        17:"edgar_spatial.nc;edgar_hourly.nc;interval_rate_control.nc",18:"qp_reaggregation_control.nc;targeted_tests.xml",
        19:"weighted_real_pm25.nc;residual_real_pm25.nc;fusion_execution_manifest.json",21:"residual_real_pm25.nc;fusion_execution_manifest.json",
        22:"quality_aware_real_pm25.nc;fusion_execution_manifest.json;targeted_tests.xml",23:"galileo_real_terrain.nc;auxiliary_input_sensitivity.json;targeted_tests.xml",
        24:"edgar_spatial.nc;real_emissions_validation.json",25:"edgar_hourly.nc;real_emissions_validation.json",
        26:"s2s_execution.json;chemical_speciation.nc.validation.json",27:"icon_static_profile.nc.validation.json",28:"bcon_static_profile.nc.validation.json",
        29:"../algorithm_library_expansion/v5/swat_existing/soil/soils.sol;targeted_tests.xml",30:"../algorithm_library_expansion/v5/swat_existing/management/management.sch;targeted_tests.xml"}
    methods={24:"emiproc.remap_inventory",25:"emiproc.DailyProfile/create_scaling_factors_time_serie",26:"EPA S2S CB6R5_AE7 + emiproc.speciate",
        27:"PseudoNetCDF.icon_profile.toGriddedFile",28:"PseudoNetCDF.bcon_profile.toPerim + explicit IOAPI metadata"}
    limits={4:"曲线网格保守重网格仍缺ESMF；只验证规则网格",5:"ERA5气温固定倍率及1.5-4倍连续查询；不支持任意变量/区域",
        7:"气温/气压路径；非完整ForDs风/辐射/降水",19:"真实PM案例仅探索性：湿度定义/时间窗/坐标及相关误差限制仍存在",
        21:"只用40训练站；验证10站；最终测试封存，仪器QA未知",22:"真实数据无最终测试指标；独立测试防泄漏由受控样例验证",
        23:"真实USGS地形进入Galileo；仅特征，不是物理变量预测，非SRTM传感器等价性证明",
        24:"仅真实EDGAR CO 2014 Paris子集；不是完整模型原生排放包",25:"年均通量等效日+显式均匀小时比例，不是实测小时排放",
        26:"CB6R5_AE7/TOG/官方0000剖面；1kg参考基准；未验证SMOKE/CMAQ读取",
        27:"旧CB05/AERO5静态6层4物种参考；显式项目元数据而非真实MCIP；非CMAQ5.5完整输入",
        28:"静态剖面型边界；动态三维化学/气象边界与官方BCON未验证",
        29:"SWAT+62文件及独立Schema/参考比对；官方软件未读",30:"仅固定日历肥料/粪肥管理；缺作物操作/轮作/自动管理的完整路径与真实配置"}
    rows=[]
    for i,item in enumerate(old,1):
        assert item["Capability"]==f"C{i}"
        rows.append({"编号":f"C{i}","能力":names[i-1],"实际执行算法":methods.get(i,item["Candidates"]),
            "运行证据":evidence.get(i,item["Evidence"]),"功能覆盖状态":"PARTIAL" if i==30 else "COVERED",
            "算法运行证据":"PARTIAL" if i==30 else "VERIFIED",
            "原生模型状态":"NATIVE_UNVERIFIED" if i>=24 else "NOT_APPLICABLE",
            "剩余具体问题":limits.get(i,item["Limitations"]),"真实可执行路径":True})
    with (OUT/"C1_C30_final_acceptance.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Freeze verified kernels and the internal algorithms re-executed in this
    # acceptance, not training-heavy or catalog-only research descriptions.
    previous=json.loads((ROOT/"output/algorithm_library_expansion/algorithm_verification_evidence.json").read_text(encoding="utf-8"))
    allowed={r["algorithm"] for r in previous["algorithms"] if r["status"]=="VERIFIED"}
    allowed.update({"spatial.nearest","spatial.bilinear","spatial.conservative","temporal.semantic_resample","downscale.qp",
        "fusion.residual","fusion.uncertainty_weighted","fusion.quality_aware","features.galileo_nano","scale.semantic_stack"})
    registry=create_default_registry();frozen=[]
    for spec in registry.specs():
        if spec.id not in allowed or spec.status!="available" or spec.category=="model_preprocessing" or type(registry.get(spec.id)).__name__=="CatalogOnlyPlugin":continue
        entry=spec.as_dict()
        entry["scale_range"]=[v if math.isfinite(v) else None for v in entry["scale_range"]]
        frozen.append(entry)
    (OUT/"frozen_generic_pool.json").write_text(json.dumps({"scope":"Task-compatibility and quality-gated baseline only; not arbitrary-input executability",
        "formal_search_started":False,"count":len(frozen),"algorithms":frozen},ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    failures=[{"stage":"official source fetch","observed":"urllib IncompleteRead for GitHub raw/tree; bounded curl HTTP1.1 / git clone fallback succeeded"},
        {"stage":"aqmbc installation","observed":"PyPI aqmbc==0.3.1 not available; installed official git commit as version0.5.0 instead"},
        {"stage":"PNC toGriddedFile","observed":"cannot understand time for file; supplied explicit project time metadata"},
        {"stage":"PNC toPerimFile","observed":"interpSigma-created TFLAG incorrectly treated as concentration; used public toPerim at identical sigma with configured metadata constructor"},
        {"stage":"native unit QA","observed":"space-padded ppmV blocked strict string comparison; QA views trim whitespace only, native file preserved"},
        {"stage":"residual QC","observed":"unknown instrument QA not accepted as VALID; basic numeric/domain screening explicitly distinguished from UNKNOWN instrument QA"},
        {"stage":"targeted tests","observed":"UTF-8 JSON read with Windows default GBK failed; explicit UTF-8 corrected"}]
    (OUT/"failure_and_fix_evidence.json").write_text(json.dumps(failures,indent=2),encoding="utf-8")
    from importlib.metadata import version
    sources={"S2S":read("s2s_source_manifest.json"),"PseudoNetCDF":{"version":version("PseudoNetCDF"),"source":"https://pypi.org/project/PseudoNetCDF/3.4.1/"},
        "aqmbc":{"commit":subprocess.check_output(["git","-C",str(ROOT/"external/aqmbc"),"rev-parse","HEAD"],text=True).strip(),"license":"GPL-3.0","executed":"3 official options/expression tests; no dynamic data run"}}
    source_files=[ROOT/"examples/ai_round1/cmaq/emissions/A_EDGAR_CO_2014_Paris.nc",*list((OUT/"pnc_official_reference").iterdir()),*list((OUT/"package_sources").glob("*.whl"))]
    sources["local_files"]=[{"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_files if p.is_file()]
    (OUT/"source_manifest.json").write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding="utf-8")
    result={"FUNCTIONAL_COVERAGE":"30/30 paths (29 COVERED + 1 PARTIAL)","STRICT_VERIFIED_COVERAGE":"29/30 declared minimum contracts",
        "MODEL_NATIVE_VERIFIED":0,"MISSING_CAPABILITIES":[],"REMAINING_BLOCKERS":["C30 crop/rotation/automatic-management implementation and real project configuration",
            "C4b curvilinear conservative ESMF path","C27/C28 real MCIP+mechanism-matched dynamic chemistry and official CMAQ native validation",
            "Independent environmental accuracy and unresolved station datum/instrument QA"],"generic_pool_frozen_count":len(frozen),
        "model_preprocessing_is_separate":True,"golem_core_unchanged":True,"tests":tests,"formal_golem_experiment_started":False}
    (OUT/"final_acceptance.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":main()
