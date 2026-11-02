# GitHub算法调研与选择报告

检查日期：2026-10-13。Stars 是当日 GitHub API 快照，只用于衡量社区可见度，不作为算法优劣证据。

## 结论先行

- 首个接入选择 **Galileo nano**：MIT 许可、官方轻量权重随仓库提供、官方提供单文件编码器，当前 Windows/CPU 环境可直接验证。
- 本次接入范围是 **多源遥感特征提取**，不是融合结果预测，更不是降尺度精度证明。
- Thera 有完整源码与权重，但官方环境是 Linux、Python 3.10、NVIDIA GPU，且公开权重训练于自然图像 DIV2K；需要单独环境和环境变量适配实验。
- Sentinel-5P 自监督源码已公开，但仓库未发现 LICENSE 和固定依赖清单；暂不进入可分发主线。
- DIIN 论文存在，但官方仓库目前只有“Code coming soon”；按要求暂缓。
- Granite WxC 和 PRIMER 都是专用路线，不能当成通用气象/环境融合器。前者只优先覆盖官方已有权重的气温、风场任务；后者面向降水且算力需求很高。

## 官方实现核查表

| 算法 | 官方论文/仓库 | 源码、权重、许可证 | 论文实际支持 | 官方效果证据 | 当前接入判断 |
|---|---|---|---|---|---|
| Thera (TMLR 2025) | [论文](https://openreview.net/forum?id=GU8YOfmqyg) / [GitHub](https://github.com/prs-eth/thera) | 完整源码；公开多种权重；Apache-2.0；868 stars | 任意倍率单图超分辨率；权重以 DIV2K 自然图像训练 | 论文报告任意尺度图像基准优于既有隐式表示，并用神经热场处理像元观测/抗混叠 | **计划接入**。需 Linux+Py3.10+NVIDIA；先验证遥感影像，不能直接宣称适用 LAI/NPP/气温 |
| Sentinel-5P 自监督 SR (2026预印本) | [论文](https://arxiv.org/abs/2604.17652) / [GitHub](https://github.com/hyamomar/Sentinel-5P-Super-Resolution) | 有 supervised/self_supervised 代码；未发现公开权重；仓库未发现 LICENSE；5 stars | Sentinel-5P 高光谱观测；使用 SURE、等变约束、S5P 退化算子和 SNR 噪声信息 | 论文报告与监督方法可比，并用 EMIT 做外部结构验证 | **研究审查**。传感器依赖强，许可和可复现环境未明确，不能泛化成任意栅格自监督 |
| DIIN (IJCAI 2025) | [论文](https://www.ijcai.org/proceedings/2025/96) / [GitHub](https://github.com/Song-1205/DIIN) | 官方仓库只有 29 字节 README：“Code coming soon”；无权重、无许可证；0 stars | 任意尺度图像超分辨率；位置感知与邻域迭代交互 | 论文报告公开图像基准上的竞争结果 | **暂缓**。没有可运行官方实现，不编造替代版 |
| Galileo (ICML 2025) | [论文](https://proceedings.mlr.press/v267/tseng25a.html) / [GitHub](https://github.com/nasaharvest/galileo) | 完整源码；nano 权重随仓库提供；MIT；203 stars | S1/S2、NDVI、ERA5、TerraClimate、VIIRS、SRTM、Dynamic World、WorldCereal、LandScan/位置等官方分组的表征学习 | 论文在 11 个遥感基准/多类任务上报告强于专业模型的总体表现 | **已接入特征编码**。不能把编码器说成通用融合/降尺度模型 |
| Granite WxC | [GitHub](https://github.com/IBM/granite-wxc) | 完整源码；Hugging Face 提供特定权重；Apache-2.0；49 stars | 基于 Prithvi WxC 的气象降尺度；官方示例含 MERRA-2 2m气温 6×、ECCC u10/v10 风场 8× | 仓库与模型卡给出上述任务示例 | **后续专用接入**。需额外 Prithvi WxC 仓库/权重，不能直接处理所有 ERA5 变量 |
| PRIMER | [论文](https://www.nature.com/articles/s41467-025-67987-9) / [GitHub](https://github.com/sunmoumou1/PRIMER) | 有研究代码；未发现仓库 LICENSE；20 stars；官方配置为 CUDA 11.7、2×A100 40GB | ERA5/IMERG/雨量站等降水记录；坐标式扩散、偏差订正、超分与融合 | 论文报告降水融合、偏差订正和超分任务优于其比较基线 | **研究借鉴**。当前电脑不适合训练，许可证未明确，不能直接套到 PM2.5 |

## 当前项目环境与选择依据

| 项目 | 当前状态 |
|---|---|
| 操作系统 | Windows 11 |
| Python | 3.12.14 |
| PyTorch | 2.14.1 CPU，CUDA 不可用 |
| 现有统一框架 | `AlgorithmSpec`、`AlgorithmRegistry`、`AlgorithmSelector`、`UnifiedAlgorithmPipeline`、`AlgorithmQualityGate` 已存在并复用 |
| Galileo 官方版本 | commit `0f0b5b95ac81acef4b74cf4686dc877202f4541b` |
| 新增最小依赖 | `einops 0.8.2`；适配器不安装 Galileo 的完整数据下载/训练依赖 |

Galileo 官方完整包声明 Python `<3.12` 且固定 PyTorch 2.2.1；本次没有强装完整包，而是按官方 README 提供的 `single_file_galileo.py` 路径，在现有 Python 3.12/PyTorch CPU 中实测 nano 编码器。这只能证明单文件推理路径兼容，不能等同于官方完整训练环境已复现。

## 首个接入内容

- 插件 ID：`features.galileo_nano`
- 官方后端：原样调用 `external/galileo/single_file_galileo.py` 和官方 `encoder.pt`
- 当前严格支持的完整输入组：
  - SRTM：`elevation + slope`
  - Sentinel-2 RGB：`B2 + B3 + B4`
  - ERA5：`temperature_2m + total_precipitation_sum`（首版仅验证单时次接口）
- 不完整输入组会被兼容性检查拒绝，不用零填充伪装成已提供数据。
- 当前每次只允许一个完整官方输入组，避免未经测试的多组组合被误称为融合。
- 输出：空间特征 `galileo_embedding`；单位 `1`；保留 CRS 和来源；明确标记 `feature_extraction`。

## 验证结果

示例输入是 8×8、30m 的合成 DEM 高程/坡度，只用于接口冒烟测试，不代表真实科学效果。

| 检查 | 结果 |
|---|---|
| 官方 nano 权重加载 | PASS，约 1,037,856 参数 |
| Registry 调用 | PASS |
| 输入组守门 | PASS，缺 slope 时拒绝 |
| 输出结构 | PASS，4×4×128 float32 特征 |
| 有效数值 | PASS，无全 NaN/Inf |
| 单位与 CRS | PASS，units=1，EPSG:32650 保留 |
| 现有质量门 | PASS |
| 环境变量预测精度 | 未验证；本例没有高分辨率真值，也没有预测头 |
| 全量项目回归 | 1034 passed，4 skipped，0 failed，0 errors（133.954秒） |

输出目录：`output/algorithm_integration_galileo/`。

## 后续优先级

1. 用真实且属于 Galileo 官方变量组的数据验证 Galileo 多组输入；之后再设计独立预测头和验证集，不能直接拿 embedding 当融合结果。
2. 为 Thera 建立隔离的 Linux/WSL2 + Python 3.10 + NVIDIA 环境；先在匹配的影像数据上复现官方例子，再评估连续环境场适配。
3. Granite WxC 仅在官方支持的 MERRA-2 2m 气温或 ECCC 风场上做小样本验证。
4. Sentinel-5P 自监督等待许可证/环境清单明确，或取得作者许可后再接入。
5. PRIMER 先提炼质量/偏差机制并验证许可与算力；不在 CPU 环境强行训练。
6. DIIN 等待官方代码公开。
