# 转换后自动自查与信息损失检测完成报告

验收日期：2026-10-02。范围：增加独立质量检查，不新增研究算法、不改动核心转换/插值/优化算法。原始数据、旧实验输出保留。

## 1. 复用了什么

复用 `ValidationReport`、`TransformRecord`、现有 Pipeline、P1/P2 Validator、`dataset_quality`、`validate_export`、面积积分和 MAE/RMSE 函数。没有另建转换引擎。

新增 `evaluation/self_check.py` 的 `PostConversionValidator` 和 `ValidationThresholds`；在现有 schemas 中新增 `InformationLossRecord`，并给已有 ValidationReport 增加 PASS/WARN/FAIL 状态和 JSON 序列化。

检查链路：原始 Dataset 深拷贝及基线 → 现有 Pipeline → 独立前后对比 → 导出 → 重新读取实际文件 → 再检查 → 下载。覆盖 P1Pipeline、SequentialPipeline、P3Pipeline、ConversionTaskEngine、TrialResult 和通用模型准备流程。模型准备还按角色检查、对实际输出文件回读并再次运行模型需求匹配。

界面自动展示中文检查摘要、状态、详细报告，并提供以下三个文件；批量 ZIP 同时包含各任务的三个文件。无需用户另点“开始自查”。

- `before_profile.json`：记录数、变量/字段、dtype/语义、单位、时间范围/间隔/时间窗口、CRS、范围、分辨率、维度、缺失量/比例、min/max/mean/std/sum、分类集合、有效像元、来源、质量标记、字段及全局元数据。
- `after_profile.json`：成功导出时来自实际文件重新读取的数据，不是简单复制转换前统计。
- `validation_report.json`：对比、信息损失、阈值、错误/警告、模型需求复查以及文件回读检查。

失败文件无法读取时，报告明确标记 `after_profile_available=false`，提供空的实际输出 profile 和失败原因；不伪造回读成功。

## 2. 新增检查规则

| 任务 | 检查重点 |
|---|---|
| 同结构格式转换 | 记录、字段、逐值误差、缺失位置、时间、分类/文字/QC、单位、CRS、元数据。按相同坐标排序后比较，不误报倒序坐标。 |
| 时空重采样 | 统计变化、峰值/位置/极值区域比例、样本量、有效比例、缺失增加、时间窗口及空间覆盖、可比支持域上的面积/时间积分。 |
| 融合 | 修正幅度 min/max/mean/std/P50/P95、负值及明确物理范围、有效区域、站点到网格距离、站点凸包外推断范围。 |
| 结构转换 | 被舍弃的属性/几何/分类/元数据、CRS、范围变化；保留的几何还检查位置与拓扑，允许声明的重投影。 |
| 模型准备 | 角色级检查、变量/单位/维度/时间/空间/CRS及配置的物理范围复查、序列化后实际文件检查；不完整时禁止称为完整模型输入。 |

温度等单位换算先把比较值归一到相同单位，再计算峰值和数值差异；原始单位及统计仍保留在 before_profile。避免把 K→degC 当成峰值下降。投影网格的诊断和导出坐标使用 x/y，不把米误称为经纬度。

`InformationLossRecord` 包含：lost_fields、lost_metadata、changed_units、changed_resolution、changed_time_semantics、changed_crs、interpolation_applied、extrapolation_applied、clipped_values、filled_values、dropped_records、peak_loss、conservation_error、warnings。粗化的像元数量减少不被当成原始表格行删除；已有裁剪标记在纯格式回读时不会重复算作新增裁剪。

## 3. 无损与有损的区别

理论基本无损：保持结构及数值的 CSV/Excel→NetCDF、NetCDF↔NetCDF-compatible HDF5、不重采样的栅格↔NetCDF。仍须实际回读检查，不能只看后缀。

通常有损：聚合、插值、重网格、表格→网格、矢量→栅格、融合、切片和字段筛选。Shapefile 字段限制、CSV/Excel/普通 JSON 的单位/CRS/元数据表达限制，也可能带来损失；必须保存旁车报告。

单位仿射换算、字段改名及合法重投影不一定丢失物理信息，但仍记入报告。WARN 不等于“对所有模型都合适”，尤其峰值敏感任务必须看峰值损失。

## 4. PASS/WARN/FAIL

| 状态 | 判定与行为 |
|---|---|
| PASS | 所检查项没有发现不可接受损失；不等同科学精度已证明或所有模型可运行。 |
| WARN | 声明的结构变化、重采样/融合、峰值降低、裁剪/填补/外推、旁车元数据、原始非关键空列等；可下载，但必须根据模型用途判断。 |
| FAIL | 关键字段/单位/CRS/时间损坏，纯格式或仅字段/单位映射发生超容差数值改变，生成全 NaN/无穷值，关键变量全缺失，违反物理范围，需守恒任务超阈值，模型需求不完整等。禁止结果下载或继续传入带 FAIL 报告的 Pipeline；只允许诊断报告。 |

默认浮点容差：`abs(error) <= 1e-8 + 1e-6*abs(source)`；整数/布尔计数要求精确一致。逐值真实改变数与超容差数分别记录，float32 正常舍入不误判失败。

默认新增加缺失比例超过 **50 个百分点**判 FAIL；全数值为空或新生成变量全 NaN直接 FAIL。原始非关键列已全缺失且原样保留只 WARN；模型必需/用户所选关键列仍 FAIL。

需要守恒的 flux、cell_total、accumulation，或声明使用 conservative 的任务：支持域可比时，相对误差超过 `1e-6` 判 FAIL。源积分为零时用绝对容差。时间窗口、空间边界或 CRS 不一致时不生成误导性守恒指标，明确 WARN“不可比”。

峰值降低超过 10%默认 WARN；字段可显式声明 `max_peak_loss_percent`，超过其上限会 FAIL。未声明物理上限时，不擅自凭经验为任意环境变量设置范围；新出现负值先警告，明确 valid_min/valid_max 被违反则 FAIL。

无独立真值时 `accuracy.available=false`；不产生“精度成功率”。独立站指标只接受显式 validation/test 划分、质控/单位/变量及实际时间匹配；训练站不得用于独立精度。站点到网格中心距离是匹配距离，不是仪器真实定位误差。

## 5. 实际执行结果

所有新结果统一位于：`D:\codex\EcoDataEngine\output\information_loss_checks\`。

| 案例 | 真实结果 | 报告目录 |
|---|---|---|
| 水质控制 CSV→NetCDF | PASS；3条、8字段；数值误差0；年份/月为整数，连续变量float，类别字符串，mg/L及缺失位置保留。 | csv_to_netcdf |
| 用户真实水质 CSV→NetCDF | PASS；67,728条、13字段均保留；所有数值 max_abs_error/mean_abs_error/changed_value_count为0，缺失位置一致。 | real_water_csv_to_netcdf |
| 1 km→5 km conservative | WARN；400→16像元，均值10.19→10.19，最大值86→13.04；峰值降低84.8372093%，最大值位置位移0 m；面积积分4,076,000,000→4,076,000,000，守恒相对误差0。 | resample_1km_to_5km |
| 背景场＋站点残差 | WARN；9像元10→12，修正均值/最大幅度2 ug/m3；3个训练站匹配距离0 m；3个网格中心在站点凸包外；无独立验证站，未报告MAE/RMSE。 | background_station_fusion |
| 故意删除氨氮单位 | FAIL；诊断报告保存，未导出损坏的模型输入或结果文件。 | rejected_corrupt_conversion |

真实水质输入：`D:\codex\水环境数据复用包_2026-09-30\02_整理数据\manual_2021_2025_multianalyte_source.csv`。输出 NetCDF 为26,196,480字节，实际回读验证成功。其余小案例明确是**合成检查数据**，不冒充公开监测真值。

另执行已有真实 CMAQ 辅助工程案例，结果保存到 `examples/cmaq_case/output_information_loss_check/`，未覆盖旧实验。实际输出5项准备变量，保留背景场、原始融合、修正和30个截断标记；每个时段的融合自查写入 quality_report。配置的标准化预处理通过，**model_format_matched=false、native_model_input_verified=false、complete_model_input=false**，未宣称可直接运行原生 CMAQ。

### 实际 validation_report.json 摘录

以下摘自 resample_1km_to_5km 的实际报告，完整 before/after profile、检查类别、阈值和警告均在文件中：

```json
{
  "status": "WARN",
  "task_kind": "resampling",
  "information_loss": {
    "peak_loss": {
      "value": {
        "amplitude_percent": 84.83720930232558,
        "before_max": 86.0,
        "after_max": 13.040000000000001,
        "position_shift_m": 0.0
      }
    },
    "conservation_error": {
      "value": {
        "before_integral": 4076000000.0,
        "after_integral": 4076000000.0,
        "relative_error": 0.0,
        "absolute_error": 0.0,
        "zero_reference": false
      }
    }
  },
  "accuracy": {"available": false}
}
```

## 6. 自动测试与实际启动

全部回归 **283 passed，0 failed，0 errors，0 skipped**，耗时25.856秒。新增43个自查测试（含参数化），覆盖水质 CSV/科学 HDF5、真实算法重采样及融合、时间积分、整数精确性、单位归一、模型不完整、信息丢失、几何错位、错误文件及 FAIL 导出门禁、原始可选空列、Streamlit PASS/WARN/FAIL、批量 ZIP。

完整记录：`output/post_conversion_validation_regression.xml`。另有11条已有依赖警告：Rasterio Affine 待弃用及故意低迭代上限的 CVXPY 不精确解警告；对应失败处理测试通过，不伪装优化成功。

Streamlit 真实进程已在测试端口8517启动，健康端点返回HTTP 200及ok；界面上传、执行、状态和下载操作由 AppTest 实际执行。临时测试服务已停止，未改用户启动端口。

复现：在项目目录运行 `.venv\Scripts\python.exe examples\run_information_loss_checks.py`；模型案例运行 `examples\run_cmaq_case.py --output examples/cmaq_case/output_information_loss_check`。不联网、不新增算法、不删除原始文件。

## 7. 普通用户怎么看

1. 双击 `D:\codex\EcoData Engine 1.0\双击打开软件.bat`，使用原有“数据转换”或“模型数据准备”。
2. 原来的上传和目标设置不变，点“执行数据转换”。
3. 结果区自动出现 PASS/WARN；展开“自动质量检查”查看原因。FAIL只显示错误和诊断下载，不提供转换结果。
4. 单文件下载结果以及三个自查JSON；多文件点“下载全部结果(zip)”，其中包括各任务的自查和失败原因。
5. 可自行上传本次目录中的 `synthetic_water_qa.csv` 做NetCDF格式转换；上传 `synthetic_1km.nc`，目标5 km、conservative做重采样。
6. 融合上传 `synthetic_background.nc` 和 `synthetic_training_stations.csv`，主数据选background，处理变量value，勾选融合并选择“背景场 + 监测站残差融合”；时间/空间分辨率不改。当前页面默认残差linear、凸包外nearest兜底，离线对照案例显式nearest，两者在此常数修正案例均得到12；报告记录实际方法与外推区域。

## 8. 边界与限制

自查是独立代码读取前后 Dataset/文件，不是独立外部物理真值。它不能证明未知单位正确、未提供CRS正确、融合提高了科学精度，或证明所有原生模型输入可运行。

基线覆盖**识别模块实际读取的数据**；任意复杂HDF中未加载的其他dataset、被显式选择之外的数据，不在逐值审计范围内。当前数值统计会读取内存数组，适用于既有本地小规模处理上限，不声称完成海量数据流式审计。

CSV/JSON等不能完整自带科学元数据，必须保留报告。WARN只表示检查发现的信息变化已显式说明，不自动保证峰值敏感、严格守恒或完整原生模型任务可用；模型准备还须满足该模型Schema，并获得对应版本原生Adapter验证。本阶段完成后停止，不进入新算法研究。
