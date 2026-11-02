# EcoData Engine 开源能力复用评估（2026-10-07）

## 已有依赖审计

现有平台已经直接使用 Xarray、Rasterio/GDAL 3.12.4、GeoPandas/PROJ、
cf-xarray、pint-xarray、xarray-regrid、GSTools、compliance-checker、
xskillscore、CVXPY/OSQP 等成熟库。xESMF 已有可选后端封装，但当前机器
缺少 ESMF/ESMPy，真实 xESMF 路径仍未执行。PseudoNetCDF 作为独立模型文件
读取验证工具使用，不把同一 Writer 的自检当作官方模型验证。

统一框架保持不变：`ModelSchema → ModelRequirementMatcher →
ProcessingPlanner → ModelAdapter → Validator`。本次只新增一个 Adapter。

## 候选复用评估

| 候选 | 可复用接口与方向 | 版本/环境成本 | 许可证与再分发 | 结论 |
|---|---|---|---|---|
| [GDAL](https://github.com/OSGeo/gdal) | C/C++、Python bindings及 `gdal_translate`/`ogr2ogr`；栅格与矢量读写、重投影、格式转换 | 原生库较重；当前 Rasterio/GeoPandas wheels 已间接使用 GDAL/PROJ | MIT/X 风格核心，驱动及第三方组件需随发行包核对 | 已复用；继续通过 Rasterio/GeoPandas 使用，不重复增加 `osgeo.gdal` 硬依赖 |
| [Xarray](https://github.com/pydata/xarray) | `open_dataset`/`to_netcdf` 和后端体系；标注多维数组的读写与计算 | 已安装 2026.7.0；NetCDF/HDF 后端已固定 | Apache-2.0，保留许可证和 NOTICE 要求 | 已直接接入，是统一 Dataset 核心；不重复封装另一套数组模型 |
| [xESMF](https://github.com/pangeo-data/xESMF) | `Regridder`；bilinear、nearest、conservative 等规则/曲线网格重网格 | Python 包轻，但真实运行依赖 ESMF/ESMPy；Windows 通常需要独立 Conda/WSL 环境 | MIT，分发时保留版权与许可文本；ESMF另行核对 | 可选直接依赖；已有封装，当前环境未执行，保留 xarray-regrid fallback |
| [PseudoNetCDF](https://github.com/barronh/PseudoNetCDF) | `pncopen` 读取及 `pncgen` 转换；覆盖部分 CMAQ I/O API、CAMx 等模型格式 | 当前验证环境 3.4.1；旧工具对 NumPy/Pandas 版本敏感 | LGPL-3.0，动态依赖和再分发需附许可证 | 作为独立读取验证器；不能据此宣称生成完整 CMAQ/CAMx 工程 |
| [SWAT+ Editor](https://github.com/swat-model/swatplus-editor) | 后端 `WriteFiles` 和 `fileio` 可写 SWAT+ 文本；输入依赖 Editor 项目 SQLite 数据库、`Project_config`、Peewee 和整套引用关系 | Python 3.10+ 后端加 Peewee/Flask；完整桌面还需 Node/Electron；接入成本高 | 仓库声明 SWAT+ 公共领域及免责条款；分发时保留声明 | 当前不直接嵌入。现有逐日天气和清单引用已独立实现并回读验证；接入完整 Editor 只会引入第二套项目数据库 |
| [FloPy](https://github.com/modflowpy/flopy) | `flopy.mf6` 创建、写出和重新加载 MODFLOW 6 simulation/model/packages；不包含 `mf6` 求解器 | 已安装并锁定 3.11.0；纯 Python，依赖 NumPy/Pandas/Matplotlib | USGS 公共领域，第三方组件另行遵循其声明 | **本次直接接入**：补齐范围明确的 MODFLOW 6 单层 DIS GWF 文本输入样例 |
| [WPS](https://github.com/wrf-model/WPS) | 官方 `geogrid.exe`、`ungrib.exe`、`metgrid.exe` 命令行；分别处理静态地理、GRIB 和气象插值 | Fortran/C、编译器、NetCDF、Jasper/PNG/Zlib，常配 MPI；不是 Python 库 | 仓库根目录未提供独立许可证文件，本次不假设再分发权；需按 NCAR/WRF 发布条款单独确认 | 适合外部官方程序调用，不适合作为便携版内置依赖；当前只参考官方输入/输出规范 |

## 已接通样例：NetCDF → MODFLOW 6（FloPy）

输入是用户明确提供的二维规则网格，必须包含：

- `top`、`bottom`、`initial_head`、`constant_head`，单位 m；
- `hydraulic_conductivity`、`recharge`，单位 m/day；
- `delr`、`delc`、`length_units`、`time_units`、`period_length`、`nstp` 属性。

路径：识别 NetCDF → Schema 字段/单位/维度检查 → 标准化 NetCDF → FloPy
生成 TDIS/IMS/DIS/IC/NPF/RCHA/CHD/OC → FloPy 独立加载路径回读 → 网格和
关键数组数值核对 → 平台 Validator 输出报告。

输出位于 `examples/open_source_reuse/modflow6_flopy/output/`。缺少定水头边界、
网格参数、单位或出现非法物理值时不生成 `mfsim.nam`。

## 验证状态

| 验证项 | 状态 |
|---|---|
| 标准化预处理支持 | 已验证 |
| 原生格式输出实现 | 已实现，并由 FloPy 重新加载和比较数组 |
| 官方软件读取验证 | 未验证（未调用 USGS `mf6` executable） |
| 官方模型运行验证 | 未验证 |

因此 MODFLOW 6 模型状态仍不能提升为“已验证原生适配”。FloPy 回读只能证明
输入结构可被 FloPy 解析，不证明数值模型收敛或工程物理设置正确。
