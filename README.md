> 当前协作基准：`main` / `platform-baseline-20261126-r8`。使用 `launch_platform.ps1` 启动 PySide 桌面平台（20261126-r8）；页面、工具包与研究客户端复用 `src/ecodata` 公共服务。详见 `PLATFORM_BASELINE.json` 与桌面验收记录。真实推荐、变异、原生写回/回读及下载清理已验收；本案例最终保留背景场，不声称融合增益，CAMx模型求解未运行。密钥与本机运行环境不入库。

# EcoData Engine

P1 implements one reproducible file pipeline:

```text
irregular station CSV -> validation -> hourly interpolation -> NetCDF
-> re-read -> independent validation report
```

The current implementation intentionally supports linear interpolation only for
instantaneous continuous variables, never extrapolates, never converts units
implicitly, and marks generated samples as `INTERPOLATED`.

Run the example from the project directory:

```powershell
python examples/p1_csv_to_netcdf.py
```

Run tests with `pytest`.

P2 adds semantics-aware interval resampling, rectilinear nearest/bilinear/
conservative regridding, uncertainty-weighted multi-source fusion, residual
background-observation fusion, sequential transform scheduling, and independent
fidelity metrics.

Run the P2 known-truth experiment:

```powershell
python examples/p2_synthetic_experiment.py
```

Current scope is intentionally limited to rectilinear latitude/longitude grids.
Native CMAQ I/O API export is not implemented; only the explicitly scoped PM2.5
auxiliary-preparation scenario is available.

## EcoData Engine 1.0 统一转换平台

Windows 用户双击 `start_ecodata_1_0.bat`，浏览器将打开 Streamlit。界面保留
“数据转换”，新增“模型数据准备”（CMAQ 辅助场景，非原生输入），统一完成自动识别、真实结构转换、时间/空间处理、融合、
模型需求匹配和结果导出。架构见 `docs/统一转换平台架构说明.md`，格式边界见
`docs/支持格式列表.md`。

1. Select Nearest, Bilinear, Conservative, Constrained QP or Peak-aware QP.
2. Select a 2.5 km or 5 km target resolution and an hour.
3. Click **Run conversion**.
4. Compare coarse input, transformed output and known reference truth.
5. Review errors, peak metrics and runtime below the figure.
6. Download the transformed NetCDF file or JSON evaluation report.

平台提供模型适配检查和自定义转换两种模式；P3 QP 对照实验保留在独立实验页。界面调用
现有 Pipeline，不包含另一套处理逻辑。通用 CF-NetCDF 示例配置位于
`examples/configs/cf_netcdf_generic.json`；真实 CMAQ 适配尚未完成。

完整中文说明见 `docs/中文操作说明.md`。

新增“格式与数据结构转换”页面：表格与规则网格转换、二维 HDF5 映射、GeoTIFF 读写、GeoJSON/Shapefile ZIP 读写、矢量栅格化和分类栅格多边形化。范围和限制见 `docs/格式结构转换说明.md`。2026-09-30 全量测试 103 项通过，记录见 `output/teacher_revision_02/regression.xml`。

## 模型需求库

`src/ecodata/models/profiles` 包含 24 个独立模型配置，传统 SWAT 与 SWAT+
分开维护。CMAQ、WRF、SWAT/SWAT+、WASP 和 MODFLOW 6 为重点配置，其余为
官方资料可追溯的基础索引。除 WASP 标为 `standardized_preprocessing` 外，
其余 profile 保持 `requirements_index`；CALMET SURF.DAT、传统 SWAT 天气、
WPS SST 和 CMAQ 辅助场景等专项转换均明确标注范围和未验证边界，不声称
可直接运行完整模型。模型清单见 `docs/model_library_catalog.md`。

CALPUFF、SWAT+、WRF、CAMx、CMAQ 的标准化预处理边界见
`docs/five_model_standardized_preparation.md`。

## P3.2 多尺度峰值保真改进

P3.2 保留普通 QP 与原始峰值感知 QP，并新增自适应平滑 QP、
自适应平滑＋可信峰值软约束 QP。运行完整可复现实验：

```powershell
python examples/tune_p32_training_cv.py
python examples/run_p32_experiment.py
```

训练侧交叉验证记录和 48 小时正式评估输出位于 `output/p3_2`。
参考真值与独立验证站只用于最终评估，不参与权重计算或参数选择。
