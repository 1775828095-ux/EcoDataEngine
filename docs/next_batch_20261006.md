# 2026-10-06 增量：两个专用输入片段

本批没有重打包便携版；`EcoDataEngine_Portable_20261005.zip` 仍是只读基线，不包含本批源码增量。下面两项都是供预处理程序使用的静态资料，未运行 SWAT+ 或 CMAQ。

## CMAQ 5.5 BCON/profile 的 BC_PROFILE

依据：[EPA CMAQ 5.5 的 BCON `prof_bcout.F`](https://github.com/USEPA/CMAQ/blob/CMAQv5.5_2Oct2024/PREP/bcon/src/profile/prof_bcout.F) 从 `BC_PROFILE` 读取注释、`VNAME,UNITS,LAYER..` 表头及每物种逐层数值，按 `ZH` 进行垂直插值。EPA [官方剖面样例](https://github.com/USEPA/CMAQ/blob/5.5/PREP/bcon/src/profile/avprofile_cb6r3m_ae7_kmtbr_hemi2016_v53beta2_m3dry_col051_row068.csv) 是 2016 年 CMAQ 5.3 beta2 模拟的远洋年平均值，44 层、251 变量；它是**既有模型输入**，并非实测上游原始浓度。本批把它独立解码成长表来检验映射与重编码，B2 仍是官方原文件，因此不能据此证明从观测数据到化学边界的完整转换。CMAQ 代码中的 EPA 授权声明允许衍生使用并要求按其声明保留权利信息；数据无另行标出的许可条款，应保留来源说明。

`mode=bcon_profile` 接受按配置映射的长表或宽表 CSV；明确层数、化学机制标签、每变量目标单位及缺测标记。支持同单位，以及 ppbV→ppmV、hPa→Pa、km→m 三种明确换算。生成 `BC_PROFILE_CMAQ55.csv`，不生成 BCON 的边界 NetCDF。静态剖面不含时间、经纬度或目标网格，后续 BCON 仍需实际 `MET_BDY_3D_FIN`、机制/物种匹配及运行配置。本机未运行 BCON，**程序实际读取仍待核实**。官方案例独立 CSV 读回 11,044 个值与 B2 数值完全一致；3 层合成宽表检验异构列名及单位换算。`BC_PROFILE` 数据本身没有小时/地理坐标，未补造。

## SWAT+ 62.0.0 的 aquifer.aqu 参数数据库

依据：[SWAT+ 62.0.0 的 `aqu_read.f90`](https://github.com/swat-model/swatplus/blob/62.0.0/src/aqu_read.f90) 以两行标题/表头后读取 id 和数据库记录；[`aquifer_module.f90`](https://github.com/swat-model/swatplus/blob/62.0.0/src/aquifer_module.f90) 定义两项名称与 15 项数值及单位。[Osu_1hru 的 `aquifer.aqu`](https://github.com/swat-model/swatplus/blob/62.0.0/refdata/Osu_1hru/aquifer.aqu) 有 21 条参考记录，但文件头标注 Editor 2.2.0、SWAT+ rev 60.5.4；本批核对 62.0.0 读入结构，**不把旧工程文件头等同于 62.0.0 整体工程验证**。SWAT+ 仓库许可为 [LGPL-2.1](https://github.com/swat-model/swatplus/blob/62.0.0/LICENSE)。

`mode=aquifer_parameters` 接受 ZIP：参数 CSV 和已有 `initial.aqu`，按配置映射 18 列并指定 15 项来源单位；校验 id 连续、名称唯一、直接 `init` 名称确实位于 `initial.aqu`。仅支持显式单位相同及 cm→mm、km→m、fraction→% 等少量确定换算。不填缺值/默认值。生成 `aquifer.aqu` 和范围说明；不生成 `aquifer.con`、HRU/河道连通、`file.cio`，也不验证 `initial.aqu` 内部更深层引用。官方 21 条模型输入独立解码成 CSV，并与官方 B2 按列数值逐项相同；合成 2 条异构字段和单位案例另行验证。官方 B2 不是现场原始地下水 A，因此只证明格式与映射过程，**不证明地下水实测参数正确**。

## 复现

从仓库根目录执行（使用项目 Python，`PYTHONPATH` 包含项目根和 `src`）：

```powershell
python -m ecodata.round1 --model cmaq --input examples/ai_round1/cmaq/bcon_profile/A_official_decoded_long.csv --config examples/ai_round1/cmaq/bcon_profile/config_official_long.json --output outputs/bcon_official
python -m ecodata.round1 --model cmaq --input examples/ai_round1/cmaq/bcon_profile/A_synthetic_wide.csv --config examples/ai_round1/cmaq/bcon_profile/config_synthetic_wide.json --output outputs/bcon_synthetic
python -m ecodata.round1 --model swat_plus --input examples/ai_round1/swat_plus/aquifer/A_official_decoded_bundle.zip --config examples/ai_round1/swat_plus/aquifer/config_official.json --output outputs/aquifer_official
python -m ecodata.round1 --model swat_plus --input examples/ai_round1/swat_plus/aquifer/A_synthetic_mapped_bundle.zip --config examples/ai_round1/swat_plus/aquifer/config_synthetic.json --output outputs/aquifer_synthetic
python -m pytest -q tests/test_cmaq_bcon_profile.py tests/test_swat_plus_aquifer.py tests/test_new_entries.py
```

界面：打开“首轮专项转换”，CMAQ 选“BCON静态边界剖面”，或 SWAT+ 选“含水层参数数据库”；分别上传示例 A 和对应 JSON，执行并下载全部转换文件。报告中的 `complete_model_input=false` 不是错误，而是明确的适用边界。
