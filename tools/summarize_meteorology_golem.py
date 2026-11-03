"""Audit real run artefacts and append exactly one delivery-report section."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ecodata.evaluation.quality_gates import RawInputQualityGate
from ecodata.io.structures import read_structure
from ecodata.recognition import inspect_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/golem_real_meteorology_round1"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def markdown(frame):
    def cell(value):
        text = f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
        return text.replace("|", "\\|").replace("\n", " ")
    lines = ["|" + "|".join(map(str, frame.columns)) + "|", "|" + "|".join(["---"] * len(frame.columns)) + "|"]
    lines.extend("|" + "|".join(cell(value) for value in row) + "|" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def summary():
    config = load("experiment_config.json")
    split = pd.read_csv(OUT / "data/station_split.csv", dtype={"station": str})
    role_sets = {role: set(split.loc[split.role.eq(role), "station"]) for role in ("train", "validation", "test")}
    assert not (role_sets["train"] & role_sets["validation"] or role_sets["train"] & role_sets["test"] or role_sets["test"] & role_sets["validation"])
    original = pd.read_parquet(OUT / "data/regional_six_years.parquet")
    fit = original[original.station.isin(role_sets["train"]) & original.time.dt.year.le(2020)]
    for variable, norm in config["normalisation"].items():
        np.testing.assert_allclose([fit[variable].mean(), fit[variable].std(ddof=0)], [norm["mean"], norm["std"]], atol=1e-10)
    validation_data = pd.read_parquet(OUT / "data/validation.parquet")
    test_data = pd.read_parquet(OUT / "data/sealed_test.parquet")
    assert set(validation_data.time.dt.year) == {2021} and set(test_data.time.dt.year) == {2022}
    assert "test" not in set(validation_data.split) and "validation" not in set(test_data.split)
    # Reuse the project table Reader and recogniser without assigning source units/CRS.
    regional = OUT / "data/regional_six_years.parquet"
    recognized = read_structure(regional.name, regional.read_bytes())
    source_semantics = inspect_dataset(regional.name, recognized).as_dict()
    source_checks = {variable: RawInputQualityGate().validate(recognized, variable=variable,
                        intended_operation="conversion", requires_spatial=False) for variable in ("rh", "pressure")}
    (OUT / "source_semantics_and_quality.json").write_text(json.dumps({"recognition": source_semantics,
        "source_checks": source_checks, "physical_geospatial_ready": False,
        "numerical_plane_is_not_source_datum_confirmation": True}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    main, test = load("main_results.json"), load("final_test_results.json")
    assert len(main) == len(test) == 30
    frozen = load("frozen_main_selections.json")
    assert len(frozen) == 30
    assert (OUT / "frozen_main_selections.json").stat().st_mtime <= (OUT / "final_test_results.json").stat().st_mtime
    table = []
    for row in main:
        match = next(r for r in test if (r["variable"], r["mode"], r["seed"]) == (row["variable"], row["mode"], row["seed"]))
        metric = match["test_result"]
        assert metric["valid"] and not row["test_opened"]
        assert row["scored_objective_calls"] <= 40
        table.append({"variable": row["variable"], "mode": row["mode"], "seed": row["seed"],
            "validation_RMSE": row["best"]["rmse"], "validation_MAE": row["best"]["mae"], "validation_Bias": row["best"]["bias"],
            "test_RMSE": metric["rmse"], "test_MAE": metric["mae"], "test_Bias": metric["bias"],
            "validation_n": row["best"]["n"], "test_n": metric["n"], "objective_calls_scored": row["scored_objective_calls"],
            "objective_calls_all": row["counts"]["objective_calls"], "budget_deferred": row["counts"].get("budget_deferred_calls", 0),
            "numerical_executions": row["numerical_executions"], "cache_hits": row["cache_hits"],
            "selected_output_nonfinite_violations": 0,
            "absolute_physical_range_checked": False,
            "absolute_physical_range_violations": None,
            "valid_evaluations": row["valid_evaluations"], "invalid_evaluations": row["invalid_evaluations"],
            "mutation_proposals": row["counts"].get("mutation_proposals", 0),
            "invalid_proposals": row["counts"].get("invalid_proposals", 0),
            "posterior_rejections": row["counts"].get("posterior_rejections", 0),
            "legal_proposal_ratio": row["legal_proposal_ratio"],
            "success": row["qualified_graph_found"], "first_qualified_evaluation": row["first_qualified_evaluation"],
            "first_threshold_evaluation": row["first_threshold_evaluation"],
            "search_seconds": row["search_seconds"], "algorithm_seconds": row["algorithm_seconds"],
            "graph": " -> ".join(row["best"]["graph"])})
    frame = pd.DataFrame(table)
    frame.to_csv(OUT / "per_seed_comparison.csv", index=False)
    metrics = ["validation_RMSE", "validation_MAE", "test_RMSE", "test_MAE", "test_Bias", "legal_proposal_ratio",
               "success", "objective_calls_scored", "numerical_executions", "mutation_proposals", "posterior_rejections", "search_seconds", "algorithm_seconds"]
    aggregated = frame.groupby(["variable", "mode"])[metrics].agg(["mean", "std"])
    aggregated.columns = ["_".join(col) for col in aggregated.columns]
    aggregated.reset_index().to_csv(OUT / "aggregate_mean_sd.csv", index=False)
    paired = []
    for variable in frame.variable.unique():
        part = frame[frame.variable.eq(variable)]
        for reference in ("A", "B"):
            c = part[part["mode"].eq("C")].set_index("seed")
            other = part[part["mode"].eq(reference)].set_index("seed")
            for seed in sorted(c.index):
                paired.append({"variable": variable, "comparison": f"C-{reference}", "seed": int(seed),
                    **{metric: float(c.loc[seed, metric] - other.loc[seed, metric])
                       for metric in ("validation_RMSE", "test_RMSE", "legal_proposal_ratio", "search_seconds")}})
    pd.DataFrame(paired).to_csv(OUT / "paired_differences.csv", index=False)
    baseline = pd.DataFrame(load("exhaustive_baselines.json"))
    baseline.to_csv(OUT / "exhaustive_baselines.csv", index=False)
    ablations = load("ablation_results.json")
    pd.DataFrame([{"variable": r["variable"], "seed": r["seed"], "ablation": r["ablation"],
        "validation_RMSE": r["best"]["rmse"], "objective_calls": r["scored_objective_calls"],
        "seconds": r["search_seconds"], "graph": " -> ".join(r["best"]["graph"])} for r in ablations]).to_csv(OUT / "ablation_comparison.csv", index=False)
    sources = []
    audit_rows = []
    for year in range(2017, 2023):
        audit = load(f"data/audit_{year}.json")
        path = Path(audit["source"])
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual_hash == audit["sha256"]
        sources.append({"path": str(path), "sha256": actual_hash, "bytes": audit["bytes"]})
        audit_rows.append({"year": year, "all_rows": audit["rows"], "national_stations": len(audit["stations"]),
                           "day_labels": audit["unique_times"], "regional_rows": audit["selected_rectangle_rows"],
                           "pressure_missing": audit["missing"].get("pressure", 0)})
    # Every successful search output is independently reopened here, not only winners.
    roundtrips, failures, ancillary_warnings = 0, [], []
    for path in sorted(OUT.rglob("*.nc")):
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            assert {"x", "y", "time_label"}.issubset(ds.dims)
            assert "NOT geodetic" in ds.attrs["crs"]
            for variable in ds.data_vars:
                assert np.isfinite(ds[variable]).all()
                if variable in ("rh", "pressure"):
                    assert ds[variable].attrs["units"] == "unknown"
                elif variable == "kriging_variance":
                    # This is the official variance for z-scored inputs, not the
                    # original-unit target. Equal-unit assertions are incorrect.
                    ancillary_warnings.append({"file": str(path), "variable": variable,
                        "status": "WARN", "reason": "Standardised-space Kriging variance has no unit attribute; do not interpret as original-unit variance or independent accuracy"})
            roundtrips += 1
    for row in main:
        for item in row["objective_trace"]:
            if not item["valid"]:
                failures.append({"variable": row["variable"], "mode": row["mode"], "seed": row["seed"],
                                 "evaluation": item["evaluation"], "reason": item["reason"], "structure": item["structure"]})
    environment = {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "pyproj", "xarray", "pykrige", "metpy", "lxml", "openpyxl")},
        "golem": "0.5.0 loaded from existing isolated environment; unmodified official APIs"}
    source_code = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in
        (ROOT / "tools/run_meteorology_golem_numeric.py", ROOT / "tools/prepare_meteorology_golem.py",
         ROOT / "tools/run_golem_search.py", ROOT / "output/capability_final_acceptance/frozen_generic_pool.json")]
    report = {"actual_main_runs": len(main), "pilot_runs": len(load("pilot_results.json")),
        "ablation_runs": len(ablations), "exhaustive_candidates": len(baseline),
        "source_hashes_unchanged": True, "sources": sources, "code": source_code, "environment": environment,
        "roundtrip_files": roundtrips, "test_station_isolation": True,
        "ancillary_metadata_warnings": ancillary_warnings,
        "absolute_physical_units_verified": False, "coordinate_datum_verified": False,
        "prediction_accuracy_unit": "original workbook numerical units, not certified % or hPa",
        "invalid_objective_evaluations": len(failures), "scientific_status": "EXPLORATORY_NUMERICAL_BENCHMARK_PHYSICAL_VALIDATION_PARTIAL"}
    (OUT / "run_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "failure_records.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    text = ["\n\n## 17. GOLEM第一轮单源真实气象数值对照实验\n", "日期：2026-10-18。\n",
        "**已实际完成两变量A/B/C真实数值搜索及2022留站评价。物理单位/坐标基准仍未核实，所以是探索性数值基准，不是完成了严格物理与地理精度验证。**\n",
        "### 数据与证据边界\n",
        "只读取用户指定目录2017—2022六份Excel，未搜索/下载其他数据；2023—2024文件未使用。六份原件SHA256再次核对一致。\n",
        markdown(pd.DataFrame(audit_rows)),
        "\n表头分别为相对湿度24小时均值、海平面气压24小时均值。实际为日标签，不是小时序列；时区和日统计截止时刻未证实，全部按同源日标签匹配，不转换UTC。\n",
        "2018坐标是12231/5258一类未声明格式、海平面气压全缺；不猜DMS转换，未纳入空间实验。2017在固定数值矩形114—119、36—41内只有8月，2019只有6—12月，训练实际为这些有效记录加2020。完整六年原件及全国级逐年审计均保留。\n",
        "来源发布机构、仪器QA、%、hPa、坐标基准无文件内证据。没有根据范围0—100就认定百分比，没有将海平面气压混作站点气压，没有使用统一气压界限。算法使用训练站/训练年拟合的z-score无量纲值；输出还原原始数值但单位明确写unknown。坐标也只做训练站标准化，用LOCAL_CS明确标识非大地坐标、非米，未伪造WGS84或公里分辨率。\n",
        "跨年位置版本及站内变化保存data/coordinate_versions.csv；计算使用当日原始坐标，不把跨年修订自动当成真实搬站或强制固定旧坐标。\n",
        "### 固定划分与真实执行路径\n",
        f"按训练期站点坐标分4×4数值格、固定SHA256站号排序抽样；实际{config['station_counts']}。所有标准化仅用2017—2020输入站，验证/测试站连历史目标值也不参与拟合。\n",
        "验证2021、测试2022各固定1/2/4/5/7/8/10/11月15日；评价站按训练坐标确定的内接矩形筛选，只使用位置/时间元数据，不看目标值。每种方法验证32个配对样本，测试24个，输入允许使用输入站同一天观测，属于同期空间估计，不是未来天气预测。Kriging默认变差函数拟合也只使用当日输入站，不使用留出目标；它不是把2017—2020拟合的变差参数固定到未来。\n",
        "测试目标不进入Evaluator验证实例；30组主实验图先写frozen_main_selections.json，之后才打开sealed_test.parquet计算最终指标。相同变量/图的测试预测只执行一次后共享，不用测试成绩挑选方案。\n",
        "真实路径：六年Excel流式解析与派生Parquet→现有RawInputQualityGate→Registry中的PyKrige/MetPy→可选9×9到17×17真实重采样→现有ResultQualityGate→验证站误差→官方GOLEM Objective→冻结图→2022测试。源数据缺失的物理元数据没有被数值质量PASS覆盖。\n",
        "### 候选、GOLEM与评价预算\n",
        "冻结34候选不变。任务可行子空间是6种站点估计×[直接17×17、9×9后nearest、9×9后bilinear]共18图；无意义分支、反向依赖、重复拟合和其他不兼容节点统一拒绝。IDW不在冻结清单，未悄悄新增。\n",
        "A/B使用GOLEM0.5.0官方默认变异族tree_growth/single_add/single_change/single_drop/single_edge，只增加计数包装；B加后验类型契约过滤。C沿用EcoData的alpha×匹配+beta×缺口−gamma×成本评分，增加本任务的类型化站点→网格提议适配。没有修改第三方优化器，也没有将新实验适配称作既有实现逐字未变。\n",
        "三组共同初始图为Ordinary Kriging；同种子、同34节点、相同最终契约/罚值/预算/数据/缓存。C生成时仅使用8个类型适用节点，这正是待比较的提议引导；A/B保留34节点生成后付出拒绝成本。所有运算在同机CPU进行，部分定向测试/元数据诊断与主实验交叠，墙钟时间仅为本机探索性成本记录，不是严格隔离负载的速度基准。\n",
        "目标=Validation RMSE/训练标准差+0.01×预先声明成本+0.002×额外网格步；无效=1,000,000。成本采用Registry声明low=.25/medium=1等，不用不稳定的实测秒数改变选图；实测计算/搜索秒数另行完整记录。阈值RMSE/训练标准差≤.75在结果前固定。\n",
        "预实验6组各最多20次，均通过；主实验2变量×3方法×5种子，最多40次计入预算的Objective。重复图采用每轮独立缓存，缓存命中也计Objective预算。GOLEM可能因约束后的繁殖/多样性不足提前结束，B的不足40次如实保留，不补造评价；预算外回调计deferred并返统一罚值。\n",
        "### 每种子实际结果\n",
        markdown(frame[["variable", "mode", "seed", "validation_RMSE", "test_RMSE", "objective_calls_scored", "numerical_executions", "invalid_evaluations", "legal_proposal_ratio", "search_seconds", "graph"]]),
        "\n完整RMSE/MAE/Bias、变异提议/拒绝、全部Objective回调、缓存、首次合格/阈值评价序号和执行秒数见per_seed_comparison.csv与各run/search_result.json。最终输出非有限值违例为0；因单位不明，绝对物理范围未检查，其违例记NA而不是伪报0。误差单位为原始数值单位，不能将表中rh误差擅自写成百分比、pressure写成hPa。\n",
        "### 均值、样本标准差及配对差异\n",
        markdown(aggregated[["validation_RMSE_mean", "validation_RMSE_std", "test_RMSE_mean", "test_RMSE_std", "legal_proposal_ratio_mean", "search_seconds_mean"]].reset_index()),
        "\n相同seed的C−A、C−B逐项差保存paired_differences.csv。五个种子重复同一批站点日期，衡量的是搜索随机性，不是五个独立环境数据集；本轮仅作描述性配对分析，不据此宣称统计显著或通用优势。\n",
        "### 穷举、单算法与消融\n",
        "两变量18个可行图均实际执行，含6个直接单算法；exhaustive_baselines.csv保留所有成功/失败及验证分数。该小空间可以穷举，GOLEM并非必要最优工具；若最优是单个插值器，不能包装成复杂算法组合创新。\n",
        markdown(pd.DataFrame([{"variable": var, "best_validation_graph": baseline[(baseline.variable == var) & baseline.valid].sort_values("score").iloc[0]["graph"],
                       "RMSE": baseline[(baseline.variable == var) & baseline.valid].sort_values("score").iloc[0]["rmse"]} for var in ("rh", "pressure")])),
        "\n对主任务rh另实跑5种子×[关闭缺口、关闭软成本、仅硬类型契约]共15组，数据/预算保持一致；pressure未追加消融，优先完成两变量主实验。\n",
        markdown(pd.DataFrame([{"ablation": r["ablation"], "seed": r["seed"], "validation_RMSE": r["best"]["rmse"], "graph": r["best"]["graph"]} for r in ablations])),
        "\n缺口项在共同初始图已满足站点估计时通常为0，关闭它的相同结果不能被解释为该机制已获支持。类型契约减少无效提议与预测精度改善是两回事；本数据没有完成%/hPa范围的物理约束消融。\n",
        "### 问题、回读与测试\n",
        f"全部{roundtrips}份实际生成NetCDF已再回读，维度x/y/time_label、有限值、目标变量unknown单位及非大地坐标声明均检查。另有{len(ancillary_warnings)}份输出的kriging_variance辅助变量没有单位属性，保留WARN：它是标准化输入空间的模型方差，不能当成原始物理方差或独立精度。未改写原始输出掩盖该问题。源数据完整，失败图与拒绝计数保留，没有删除不利种子。\n",
        "实际修复：六年拼接中空2018区域表导致time列object，已显式按已有日期转datetime；2018全缺压力的Infinity审计界限改为JSON null；最终审计原先错误地要求辅助方差与目标变量同单位，已分开检查并记录上述缺失属性警告；这些修复不填补数据、不改变算法数学。Windows主环境复用已有隔离GOLEM包路径，未重新安装模型或修改第三方源码。\n",
        "结果状态：数值GOLEM端到端已运行；绝对物理单位、地理坐标和原始发布/QA证据仍PARTIAL。不能据此宣称‘物理语义引导在真实气象任务上普遍优于普通/后验GOLEM’。\n",
        "关键代码：tools/prepare_meteorology_golem.py、tools/run_meteorology_golem_numeric.py、tools/summarize_meteorology_golem.py；所有原始/派生审计、配置、输出、图JSON、CSV、失败记录及环境信息均在output/golem_real_meteorology_round1/。\n"]
    text.append("\n### 复现命令\n\n使用新的实验目录，勿覆盖本轮输出。主实验已有逐run JSON时支持中断续跑；配置、预实验和最终测试已存在时会阻止静默重写。以下命令均在项目根目录执行：\n\n"
        "```powershell\n"
        "$env:PYTHONIOENCODING='utf-8'\n"
        "$meteoRun='output/golem_real_meteorology_reproduce_new'\n"
        ".venv\\Scripts\\python.exe tools/prepare_meteorology_golem.py --data 'D:\\codex\\数据\\气压、湿度数据' --output \"$meteoRun/data\"\n"
        ".venv\\Scripts\\python.exe tools/run_meteorology_golem_numeric.py --output $meteoRun --phase prepare\n"
        ".venv\\Scripts\\python.exe tools/run_meteorology_golem_numeric.py --output $meteoRun --phase pilot\n"
        ".venv\\Scripts\\python.exe tools/run_meteorology_golem_numeric.py --output $meteoRun --phase main\n"
        ".venv\\Scripts\\python.exe tools/run_meteorology_golem_numeric.py --output $meteoRun --phase baselines\n"
        ".venv\\Scripts\\python.exe tools/run_meteorology_golem_numeric.py --output $meteoRun --phase ablation --variables rh\n"
        ".venv\\Scripts\\python.exe tools/run_meteorology_golem_numeric.py --output $meteoRun --phase test\n"
        ".venv\\Scripts\\python.exe -m pytest -p no:cacheprovider --basetemp=output/pytest_meteorology_reproduce_new\n"
        "```\n\n随机与NumPy种子固定，输出中的每个图可按已冻结计划重放；这不意味着不同硬件/库版本上的墙钟耗时或浮点末位完全一致。\n")
    full_xml = OUT / "full_regression.xml"
    if full_xml.exists():
        suite = ET.parse(full_xml).getroot().find("testsuite")
        assert int(suite.attrib["failures"]) == int(suite.attrib["errors"]) == 0
        passed = int(suite.attrib["tests"]) - int(suite.attrib["skipped"])
        text.append(f"\n本轮全量回归：{passed}通过、{suite.attrib['skipped']}跳过、0失败/错误；{float(suite.attrib['time']):.2f}秒。XML在同一输出目录。\n")
    else:
        raise ValueError("Full regression evidence required before report append")
    text.append("\n### 对六个研究问题的直接回答\n")
    for variable in ("rh", "pressure"):
        means = frame[frame.variable.eq(variable)].groupby("mode").mean(numeric_only=True)
        text.append(f"\n{variable}：C的合法变异比例{means.loc['C','legal_proposal_ratio']:.2%}，A为{means.loc['A','legal_proposal_ratio']:.2%}，B为{means.loc['B','legal_proposal_ratio']:.2%}，因此确实减少无效提议。"
            f"验证RMSE均值A={means.loc['A','validation_RMSE']:.6f}，B={means.loc['B','validation_RMSE']:.6f}，C={means.loc['C','validation_RMSE']:.6f}。"
            f"最终测试RMSE均值A={means.loc['A','test_RMSE']:.6f}，B={means.loc['B','test_RMSE']:.6f}，C={means.loc['C','test_RMSE']:.6f}。"
            f"搜索秒数均值A={means.loc['A','search_seconds']:.2f}，B={means.loc['B','search_seconds']:.2f}，C={means.loc['C','search_seconds']:.2f}。\n")
    text.extend(["\n首次合格图/首次预设阈值各组均由共同初始化达到，不能证明C更快；实际C平均墙钟耗时更长，更多有效执行不等于更低时间成本。成功找到合格图的主实验均为5/5，也不构成C成功率提高的证据。\n",
        "两个变量均未显示C优于B的验证精度；相同最终图的测试预测也完全相同。相对湿度最终测试C均值略差于A，海平面气压最终测试C均值优于A，因此不能说C在独立测试上一致占优。C相对A更稳定降低验证误差，不能证明新的插值算法精度或普遍泛化。\n",
        "本轮贡献主要可归于类型契约约束的提议机制。缺口引导在已满足任务的初始图上不活跃；软成本和硬类型消融只能按已执行表解读，不能声称全部引导分量均有效。%/hPa、真实坐标及严格气象物理约束的优势仍未验证。\n",
        "后续应取得单位/坐标/发布与QA证据，并在更大但确有处理必要性的任务空间和更多独立站点日期复验；不得加无意义节点、故意劣化基线或筛选数据制造C优势。\n"])
    report_path = ROOT / "docs/EcoData Engine项目总交付报告.md"
    old = report_path.read_text(encoding="utf-8")
    if "## 17. GOLEM第一轮单源真实气象数值对照实验" in old:
        raise ValueError("Section already exists; review rather than silently overwrite")
    report_path.write_text(old + "\n".join(text), encoding="utf-8")
    print(aggregated[["validation_RMSE_mean", "test_RMSE_mean", "legal_proposal_ratio_mean"]].to_string())
    print("VERIFIED", report["actual_main_runs"], "runs,", roundtrips, "roundtrips")


if __name__ == "__main__":
    summary()
