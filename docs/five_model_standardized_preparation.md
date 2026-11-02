# 五模型标准化预处理说明

目标模型：CALPUFF、SWAT+、WRF、CAMx、CMAQ。

## 状态边界

- 模型整体状态仍为 `requirements_index`。
- 每个模型的 `preparation_scenario` 只表示可使用统一 `standardized` adapter 做标准化预处理。
- `native_model_input` 始终为 `false`。
- 没有官方工具读取或真实运行验收，不声称原生适配。

## 场景

| 模型 | 场景 | 已定义标准化输入 |
|---|---|---|
| CALPUFF | `calpuff_standardized_v1` | 控制参数、气象、地形、土地利用、排放、受体 |
| SWAT+ | `swat_plus_standardized_v1` | 控制、地形、土地利用、土壤、气象、流域对象 |
| WRF | `wrf_standardized_v1` | 静态地理、气象分析、气压、温度、湿度、风 |
| CAMx | `camx_standardized_v1` | 气象、排放、初始浓度、边界、化学参数 |
| CMAQ | 既有 `cmaq_pm25_auxiliary` | PM2.5 背景、气象、地形、观测、边界依赖 |

## 生成与复现

```powershell
$env:PYTHONPATH = "$PWD\src"
_runtime\python.exe tools\add_five_model_preparation_scenarios.py
_runtime\python.exe tools\run_real_case_preparation.py
_runtime\python.exe tools\build_real_case_manifests.py
```

真实案例、平台结果、对比报告和总报告位于用户交付目录
`C:\Users\ASUS\Documents\Codex\2026-10-02\wo\outputs`。
