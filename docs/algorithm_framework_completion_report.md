# 智能跨尺度融合算法框架完成状态报告

日期：2026-10-13

## 已完成

- 新增插件式 `AlgorithmRegistry` 与 entry point 自动发现接口。
- 新增 `AlgorithmSpec/Context/Result/Compatibility` 统一契约。
- 新增目标尺度建议器，按连续量、分类量、总量/通量选择安全默认方法。
- 新增四层算法筛选：输入守门、候选计划、validation比较、物理质量门。
- 新增组合执行能力；现阶段支持“尺度统一 → 观测残差融合”的通用计划表达。
- 将现有 nearest、bilinear、conservative、IDW、残差融合、反方差融合、QP、FNO、
  UNO、Observation-guided UNO 包装或登记为插件；算法核心仍复用原实现。
- ZSSR、Thera、LIIF、DIIN、cubic 作为明确不可自动调用的候选登记。
- 新增7项自动测试，覆盖注册状态、语义尺度建议、自动筛除、重网格、IDW、
  validation选型和最终test拒绝。
- 真实京津冀 PM2.5 案例已通过 Registry → IDW插件 → 质量门 → NetCDF回读闭环。

## 真实案例结果

- 数据：2019-08-02 00:30 的 GEOS-CF 网格及固定40个训练站。
- 算法：`fusion.idw`，仅训练站参与；validation/test观测不用于插值。
- 输出：`output/algorithm_framework_real_case_20261013/pm25_idw_framework_output.nc`。
- 质量：文件可重新读取，变量和单位存在，数值有限且满足非负约束。
- 精度边界：没有独立高分辨率全场真值，报告明确 `accuracy_claimed=false`。

## 当前可执行插件

| 插件 | 后端 | 状态 |
|---|---|---|
| spatial.nearest / bilinear / conservative | xarray-regrid | available |
| fusion.idw | SciPy距离矩阵 | available |
| fusion.residual | 现有 ResidualFieldFusion | available |
| fusion.uncertainty_weighted | 现有反方差融合 | available |
| downscale.qp | 现有 CVXPY + OSQP P3.2 | available |
| neural.fno / multiscale_uno / observation_guided_uno | 现有 neuraloperator + PyTorch | available（仍需相容训练数据） |

`available` 表示后端和调用路径存在，不表示某个新数据集上已经证明精度更高。

## 尚未完成或未验证

- ZSSR：非商业学术许可证；PNG/图像路径尚未适配环境数据对象。
- Thera：需要独立 Linux/Python3.10/NVIDIA/JAX 环境；未下载自然图像权重。
- LIIF：未建立环境场训练集及反向聚合验证。
- DIIN：官方仓库目前没有代码和许可证，不能接入。
- cubic：尚未选择并验证统一后端，自动路由回退到 bilinear/nearest/conservative。
- 当前组合候选只搭建通用机制，没有把不兼容神经模型强行串联。
- 自动质量门已覆盖基本结构/物理检查；复杂守恒和信息损失仍由现有
  `PostConversionValidator` 在正式任务管线中执行。

## 复现

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_algorithm_framework.py -q --basetemp=.pytest_tmp_algorithm_framework
.\.venv\Scripts\python.exe tools\run_algorithm_framework_real_case.py
```

输出目录还包含 `algorithm_catalog.json`、`algorithm_screening.json` 和
`execution_report.json`，可审计每个候选为何可用或被拒绝。

## 回归结果

- 框架专项：7/7 通过。
- 全项目：1031项收集，**1027通过、4跳过、0失败、0错误**，运行约213秒。
- 完整机器可读结果：`output/algorithm_framework_real_case_20261013/full_regression.xml`。
- 跳过项保持原有可选依赖/条件状态，没有被冒充为通过。
