# Neural Reconstruction 开源集成与实测

## 1. 原有结构及本次范围

`fusion/neural_reconstruction.py` 已实现统一单位、背景上采样、多通道辅助变量、站点拆分、官方FNO残差重构和独立站点评价。
`observation_encoder.py` 原有点MLP和手写单头Attention；`interface/fusion_experiment_page.py` 调用同一核心并输出NetCDF/CSV/JSON。
本次没有修改普通转换、模型适配或其他融合算法。旧版FNO仍为默认值，原始实验输出保留。

## 2. 核查与复用选择

Stars为2026-10-09网页显示值，约数不代表精度、普及率或环境任务性能。

| 项目 | Stars | 论文/场景 | 实际复用位置与边界 |
|---|---:|---|---|
| [neuraloperator](https://github.com/neuraloperator/neuraloperator) | 约3.9k | [FNO](https://arxiv.org/abs/2010.08895)；[U-NO](https://arxiv.org/abs/2204.11127)，函数场/PDE基准 | 直接调用已安装1.0.2的FNO及UNO。UNO内部完成降/升尺度和跳跃连接；不自行写频谱层。MIT |
| [Senseiver](https://github.com/OrchardLANL/Senseiver) | 37 | [Senseiver论文](https://www.nature.com/articles/s42256-023-00746-x)，稀疏传感器重构连续场 | 借鉴位置/值编码及网格查询思路；没有复制源码，也不宣称完整Senseiver接入 |
| [PyTorch](https://github.com/pytorch/pytorch) | 约104k | [Attention Is All You Need](https://arxiv.org/abs/1706.03762)；[Perceiver IO](https://arxiv.org/abs/2107.14795)的跨注意力思路 | 直接调用`nn.MultiheadAttention`，格点背景/辅助特征作为query，观测token作为key/value。不是完整Perceiver复现 |
| [Weather Downscaling](https://github.com/NatLabRockies/neural-operators-weather-downscaling) | 6 | 仓库提供2025天气零样本降尺度论文及FNO/DFNO/DUNO等比较 | 用作降尺度评估参考，不引入SLURM训练框架或替换平台；BSD-3-Clause |
| [Focal Frequency Loss](https://github.com/EndlessSora/focal-frequency-loss) | 715 | ICCV2021，图像重构频率误差 | 直接调用官方pip库0.3.0；仅约束可观测粗场频谱，不假造高频真值。MIT |
| [PINO](https://github.com/neuraloperator/physics_informed) | 555 | Physics-Informed Neural Operator for Learning PDEs | 旧仓库已停止维护；仅参考约束思想，不安装旧实现。当前无污染输运方程/边界参数，不称PDE-PINO |

不是所有论文仓库都成熟/近期流行；具体选择以可调用官方API、维护成本及任务适配为准。

## 3. 实现及修改文件

新增`AttentionObservationEncoder`：MLP编码实际站点位置、数值及QA；PyTorch官方跨注意力融合背景和辅助格点特征。QA作为注意力log prior及观测loss权重；padding完全忽略，无观测时间片输出零embedding。

新版`backend='multiscale_uno'`：背景/辅助通道与观测embedding → 官方UNO（0.5、1、2尺度；skip）→ 背景残差修正。为减少重复模块，多尺度提取放在UNO内部，未另造一个FNO前置多尺度网络。

loss = 原有观测/粗场/梯度或TV/非负loss + 粗场Focal Frequency Loss + 训练观测上90%分位峰值加权误差。
峰值项是简单观测加权约束，不宣称来自上述论文的专有峰值算法。非负是软loss与最终裁剪，不是严格PDE物理一致性。
无高分辨率训练真值时，无法要求高频谱等于未知真值；此时频率loss只在粗网格计算，不保证补回未观测细节。

修改：`neural_reconstruction.py`、`observation_encoder.py`、`fusion_experiment_page.py`、`pyproject.toml`、`requirements-lock.txt`及本说明/依赖文档。
新增可复现实验`examples/run_neural_reconstruction_upgrade.py`；扩展现有测试文件，没有复制第三方网络源码。

## 4. 实际48小时对照（探索性、单种子）

12训练站、10独立验证站、480验证记录；种子20261009，两版40轮、相同数据和已有loss权重。高分辨率场仅传入`evaluation_truth`。没有用验证指标搜索参数或挑选运行。
本案例没有实际辅助变量，因此不能据此证明多变量辅助数据提高质量；辅助输入链路在单元测试中另行覆盖。

| 指标 | 原版FNO | UNO+Attention |
|---|---:|---:|
| 全场RMSE | 5.7086 | 4.8427 |
| 全场MAE | 3.7322 | 2.3419 |
| 独立站RMSE | 6.3064 | 6.6725 |
| 独立站MAE | 4.8090 | 4.8491 |
| bias | 2.5975 | 4.5281 |
| 每时刻全场最大值幅值MAE | 10.9346 | 3.6771 |
| 格点梯度MAE | 1.3787 | 1.4443 |
| area投影粗场一致性RMSE | 4.7877 | 3.7844 |
| 运行时间秒（本机单次） | 2.64 | 2.24 |

单位误差为ug m-3；梯度为格点差值而非每米梯度。area投影为既有近似粗场loss，不宣称精确地理面积守恒。
全场RMSE改善约15.2%、峰值幅值误差改善约66.4%；独立站RMSE恶化约5.8%，梯度误差也恶化。双线性全场RMSE为3.4359，仍优于两个网络。两版报告均为WARN。没有证明全面更优、泛化或正式同化。

完整输出：`output/neural_reconstruction_upgrade/`中的`fno.nc`、`multiscale_uno.nc`、各自站点CSV/JSON和`comparison.csv`；不覆盖`output/neural_reconstruction_demo/`。

## 5. 操作及复现

安装可选依赖：`.venv\Scripts\python -m pip install -e .[ml]`。
打开“融合重构实验” → 内置模拟数据 → 高级设置 → 重构版本“多尺度UNO + Cross Attention” → 运行；下载结果包。
命令复现：`.venv\Scripts\python examples/run_neural_reconstruction_upgrade.py`。
验证站位置和观测值不进入训练token/loss，评价真值只参与坐标定义与最后评价；测试修改验证值/评价真值后输出完全一致。

## 6. 限制

当前是单个场景在线拟合原型，不是跨场景训练完成的通用预训练operator，也没有独立时间外推验证。不支持自动处理非规则/投影网格、分类土地利用编码、学习不确定性或动态来源可靠性。
训练真值模式仍需另设独立时空训练集才能做严格全场泛化评价；本次只验证不使用训练真值的路径。
PyTorch/neuraloperator仍有上游多维索引弃用警告，不修改第三方安装源码。

## 7. 实际测试记录

- 重构+P2融合相关回归：12通过；包含原版和新版、辅助多通道、验证值/评价真值污染不改变输出、padding/点排列不变、无观测安全输出。
- 全量pytest：999通过、4跳过、0失败，128.81秒，205条警告。未把跳过项目计为通过。
- Streamlit AppTest：选择`multiscale_uno`、点击运行，exception/error均为空，显示10验证站、RMSE6.673、基线RMSE12.013、MAE4.849。
- 输出独立回读：h5netcdf读取新版输出，time=48、lat=24、lon=24、float32、单位ug m-3、EPSG:4326、全部有限值。
- 全回归命令：`.venv\Scripts\python -m pytest -o addopts='' -q -p no:cacheprovider --basetemp=D:\codex\ecodata_upgrade_regression_1009`。basetemp是本次测试临时目录，可换为本机新目录，业务代码不依赖该路径。

按`academic-paper-writing-skill`证据纪律保留单种子、模拟数据、负结果及未完成项；本报告不声称论文创新或统计显著提升。
