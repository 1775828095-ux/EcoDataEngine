# EcoData Engine 小规模真实空气质量工程场景测试

测试日期：2026-10-02。目的：验证四类真实环境数据进入已有后端后，识别、标准化、匹配、残差融合、科学数据输出与回读检查的工程链路。未运行真实环境模型，未新增研究算法，未改变整体架构。

## 1. 结论与验收边界

四份真实公开小样本已准备，完整后端流程已实际执行，生成含 PM25、temperature、wind、humidity、elevation 的 `model_input.nc`。必需变量全部为 float64，无 NaN，CRS、时间中心、小时平均边界、单位及来源说明保留。新增5项测试通过；全部回归结果见下文。

**不能据此认定已得到可正式驱动 CMAQ 等模型的输入。** 本案例是自定义五变量标准化科学数据要求，不是某个真实模型版本的原生输入规范。现有网页也尚不能一键把四类异构来源合并成该文件；本次通过工程案例脚本调用现有后端，未重做界面。

实际融合发现明显限制：原始简单残差融合有30/72个负浓度像元，最低 -47.5811899 ug/m3。最终候选输入显式施加非负边界，截为0；原始场单独保存，修正位置有质量标记。修正幅度较大，说明稀疏站点残差外推并不稳健，**不得将流程成功解释为融合精度或模型适用性已验证**。没有独立真值，不计算 MAE/RMSE。

## 2. 交付目录

```text
D:\codex\EcoDataEngine\examples\engineering_case_air_quality\
├── observation/pm25_station.csv
├── meteorology/era5_sample.nc
├── terrain/dem.tif
├── background/pm25_background.nc
├── _sources/                         原始公开响应，不删除
├── source_manifest.json              来源 URL、SHA256、字节数
├── README.md                         字段与使用说明
├── 运行工程测试.bat                  离线一键重跑
└── output/
    ├── model_input.nc                最终标准化候选输入
    ├── fused_pm25.nc                 未截断的原始融合场与背景、残差
    ├── input_analysis.json           数据识别报告
    ├── model_requirement.json        可编辑的统一 Schema 自定义要求
    └── engineering_report.json       处理计划、匹配、质量和限制
```

原始来源档案共8,881,839字节，约8.47 MiB；加输入、输出及说明约9 MB。仅下载区域与时段子集；未下载全球数据、完整大型 DEM、深度学习模型或新依赖。下载器设置单文件5 MiB、来源档案总量30 MiB上限。

## 3. 数据来源、格式及作用

| 来源/文件 | 大小与结构 | 变量、单位、时空信息 | 作用 |
|---|---|---|---|
| AirNow `pm25_station.csv` | 2站、16条记录，CSV | PM25，ug/m3；官方经纬度，EPSG:4326；2024-01-01至01-02的8个小时平均时段 | 提供观测残差 |
| NASA GEOS-CF `era5_sample.nc` | 48个小时、4×5网格，NetCDF | temperature K，u10m/v10m m/s，wind m/s，humidity %；0.5°隔点子采样，EPSG:4326 | 气象配套与网格/时间对齐 |
| USGS 3DEP `dem.tif` | 128×128、单波段Float32，GeoTIFF | elevation m，EPSG:3857，真实投影范围与 transform | 实际 CRS 转换与重采样 |
| NASA GEOS-CF `pm25_background.nc` | 8个时段、3×3网格，NetCDF | PM25，ug/m3，RH35条件；0.25°，EPSG:4326 | PM2.5背景与目标网格 |

区域：洛杉矶附近。背景网格中心经度 -118.25、-118.0、-117.75，纬度34.0、34.25、34.5。站点为 Glendora - Laurel 和 Los Angeles - N. Main Street，来自官方站点位置文件，不补造坐标。

气象使用用户允许的“ERA5或类似数据”中的 **GEOS-CF**；`era5_sample.nc` 沿用指定文件名，但文件 `actual_product`、README 和报告明确声明 **不是 ERA5**。GEOS 源通过 OPeNDAP 服务端截取、ASCII 数值传输并保存 NetCDF，保留原始响应与轴信息；没有用模拟值替代公开数据。

官方资料：

- [AirNow Hourly Data Fact Sheet](https://files.airnowtech.org/airnow/docs/HourlyDataFactSheet.pdf)：文件时间为 GMT 测量窗口开始；公开小时记录为 preliminary，不能作为监管级最终审核数据。观测和站点原始文件链接逐项记录在 `source_manifest.json`。
- [NASA GEOS-CF 数据访问](https://gmao.gsfc.nasa.gov/gmao-products/geos-cf/data-access_geos-cf/)与[GEOS-CF File Specification v1.3](https://gmao.gsfc.nasa.gov/pubs/docs/Knowland1446.pdf)：PM25_RH35_GCC为ug m-3，T2M为K，U10M/V10M为m s-1，RH为无量纲比例。
- [USGS 3DEP ImageServer](https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer)：请求裸地浮点高程，关闭默认 Hillshade。完整请求、服务元数据及原始TIFF已缓存。

## 4. 实际处理流程

1. **数据理解**：调用现有 `recognise_file` 读取四类输入并生成格式、结构、变量、字段类型、单位、CRS、范围、分辨率、缺失统计。报告补充实际 UTC 时间范围。CSV 的 PM25 无表头单位时，依据明确的 `unit` 列和官方说明确认，不凭变量名猜测。
2. **时间统一**：AirNow 的起止时刻完整保留，并将统一 `time` 设为小时窗口中心；GEOS 依据服务明确的轴原点与小时步长解码。气象48个小时中选择与背景相同的8段，逐段检查 `time_bnds` 完全一致。**不对每隔6小时的缺口做插值，也不把1小时平均冒称6小时平均。**
3. **空间统一**：目标采用背景3×3、0.25°网格。气象隔点场由既有 `supported_regrid`/`GridRegridder` 双线性重网格。DEM 用 Rasterio/PROJ 从EPSG:3857重投影至EPSG:4326并双线性重采样；使用实际transform，纬度与数值同时调整为递增。地形为静态二维，不人为制造逐时高程。
4. **单位与语义**：观测UG/M3规范记录为ug/m3，与背景单位相同；RH由来源明确的fraction×100转为%；温度通过既有模型适配单位换算K→degC。风先对齐u/v，再取向量模，不将其误称为小时平均风速。
5. **融合**：调用既有 `run_residual_task`，每个同一小时窗口分别执行背景+站点残差融合。两站不足以支持稳定线性散点插值，明确选用已有nearest残差策略；背景在站点处仍使用现有双线性采样。过滤只允许VALID且域内站点。气象、高程不是PM2.5来源，不与浓度混合。
6. **物理边界诊断**：先保存未修正 `fused_pm25.nc`，再对最终候选输入的负PM25显式截为0，保留 `PM25_bound_flag`（0未改、1截断）、修正数量及最大幅度。不是新增研究算法，也不修改原核心残差算法。
7. **需求匹配**：通过已有 `suggest_mapping`、`prepare_input` 匹配本案例自定义五字段要求，自动温度换算；缺少humidity的反例必须判定不满足。自定义配置使用统一 Model Schema，状态为标准化预处理，不标记已验证原生适配。
8. **导出验证**：使用已有 `export_structure` 和 `validate_export` 写NetCDF、重新读取，检查数值、维度、变量类型、单位、时间、CRS、缺失和元数据。额外检查五个必需变量全部有限、PM25非负。

### 最终输出结构与实测范围

| 变量 | 维度 | 类型/单位 | 实测min～max |
|---|---|---|---|
| PM25 | time×lat×lon（8×3×3） | float64，ug/m3 | 0～48.746679 |
| temperature | time×lat×lon | float64，degC | 2.859850～17.038420 |
| wind | time×lat×lon | float64，m/s | 0.596535～4.355654 |
| humidity | time×lat×lon | float64，% | 37.041810～97.895250 |
| elevation | lat×lon（3×3） | float64，m | 103.751162～1238.433148 |

另保留u10m、v10m、PM25_bound_flag及time_bnds。时间中心从2024-01-01 00:30到2024-01-02 18:30 UTC，8段，每段小时窗口长度1小时。全部必需变量有效比例100%，最终文件CRS为EPSG:4326。地理网格0.25°不是固定公里大小。

## 5. 发现并修复的问题

### 已修复：NetCDF质量检查对维度顺序过敏

同样的(time,lat,lon)变量在Dataset中不同的维度插入顺序，导出回读后逐行展开顺序可能改变。原质量检查将其误判为temperature数值丢失。本次只修正质量检查的比较遍历顺序，统一 `dim_order`，不改变转换数值或算法；新增针对该问题的回归测试。

### 已识别并明确处理：残差融合的负浓度

30/72个像元原始结果为负，最大修正幅度47.5811899 ug/m3。来源原值未修改；原始融合保留；最终候选文件显式截断并标记。**仍是方法适用性限制，并非已解决的精度问题**。本次不扩展研究算法，不将稀疏站点融合用于正式输入的物理可信度作保证。

### 服务端范围与可用性

USGS服务一次返回502，限定重试后下载成功。服务为保持投影像元比例，将请求经度范围略向外扩展；测试确认实际TIFF覆盖目标范围，使用实际transform，而不是强行把请求范围写回文件。缓存保证离线重复运行不依赖服务即时可用性。

## 6. 测试步骤与实际结果

项目根目录执行：

```powershell
.\.venv\Scripts\python.exe examples/run_engineering_case_air_quality.py
.\.venv\Scripts\python.exe -m pytest tests/test_engineering_air_quality.py -q
.\.venv\Scripts\python.exe -m pytest --junitxml=output/engineering_case_regression.xml
```

- 完整工程链路已真实运行并重复离线执行。实测处理约0.37秒（本机运行，不含网络下载、Python启动），不作为普遍性能保证。
- Windows一键脚本实际执行成功，出现DONE，退出码0。
- 新增5项测试：DAP子集解析/错误拒绝；真实结果回读与单位/时间/融合关系；实际DEM投影及高程；NetCDF维度插入顺序质量检查；缺失物理变量匹配拒绝。5项通过，无网络下载。
- 全回归最终复跑：214 passed，11 warnings，55.62秒。结果见JUnit `D:\codex\EcoDataEngine\output\engineering_case_regression.xml`。既有209项加本次5项；未覆盖原有实验产物。
- 已有警告为Rasterio仿射运算待弃用提示，以及测试故意限制QP迭代时的求解精度警告。不是本例下载或导出失败。

### 普通用户应怎么测试

优先双击案例目录的“运行工程测试.bat”，无需先填写CRS或单位。正常时看输出目录中的model_input.nc与engineering_report.json。报告里应有匹配ready=true、导出quality.passed=true，同时明确native适配=false、负值修正30格。

网页可分别上传四个输入检查识别，或加载生成的model_input.nc进行已有格式转换。**不要把四个异构原始文件上传后勾选普通多源加权融合**：它们不是同一变量的四个来源，也不是网页现有通用融合能够自动混合的对象。本次未新增一键异构拼接前端；完整场景通过上述已验证脚本执行。

失败原因：缺观测时间段、单位不明、CRS缺失、时间窗口不一致、必需字段缺失、输出NaN等均应停止并给出原因。不要填写猜测坐标或单位来绕过检查。这个案例输出成功，只意味着预处理文件结构和检查通过；原生模型格式、连续驱动时段及融合物理可信度仍需专项验证。

## 7. 本阶段完成范围

已验证：真实多源识别、明确单位确认、时间窗口对齐、实际不同CRS转换、不同网格对齐、简单残差融合、自定义输入匹配、浮点NetCDF输出及回读、原始数据追溯。未声称：ERA5下载、连续48小时驱动、融合精度提升、四源网页自动合并、原生环境模型适配或真实模型运行。到此停止，不扩展算法研究。
