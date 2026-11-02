# Observation-guided Neural Reconstruction：实现与实测

日期：2026-10-09。本阶段完成代码与实验框架，不宣称已经证明算法优化。没有新增网络家族、模型适配器或下载数据；普通转换、模型准备和原有融合流程未重构。

## 1. 完成了什么

- 保留 `fno`、`multiscale_uno`，新增可切换的 `observation_guided_uno` 实验配置，默认仍是原版FNO。
- 训练站观测loss支持QA、共同时间内的数据完整性、可选观测标准差。权重不使用验证站数值或参考场。
- 观测loss按训练轮次逐渐加强；增加训练站加权bias约束。
- 在原有PyTorch Cross Attention中增加局部观测残差通路和固定gate，复用同一个Attention模块；没有重新实现Transformer/FNO频谱层。
- 增加真实NetCDF/GeoTIFF背景、连续/分类辅助变量、CSV观测读取接口。投影栅格通过现有rasterio/GDAL重投影，不把投影x/y直接改叫lat/lon。
- 同一站点划分、时间、目标网格和评价器下比较六种方法：双线性、Ordinary Kriging、普通QP、FNO、原版UNO、观测引导UNO。
- 输出高分辨率NetCDF、独立站预测CSV、comparison.csv及JSON实验报告，记录来源SHA256、依赖版本、全部参数、种子、权重和训练损失。每次新建带时间戳的目录，不覆盖旧结果。

## 2. 原理与开源复用边界

训练可靠度：

`r_i = QA_i × completeness_station × clip(median_train(σ²)/σ_i², 0.05, 20)`。

`completeness`是训练站在背景共同时间内有效时间点的比例。没有提供σ时，误差精度因子为1；没有QA时按等可靠度，不解释为正式质控。
loss权重为 `r_i / mean_train(r)`；Attention的可靠度prior为 `r_i / max_train(r)`。σ是观测标准差，单位与value相同；单位转换只换算标准差尺度，去除温度等单位的零点偏移。非正/缺失σ、非法QA和重复站点/时间会被拒绝。

观测项：`8 × schedule(epoch) × Σ w_i |prediction_i-observation_i| / Σ w_i`。
默认40轮，前25%轮次线性warmup到完整权重；这只是预设训练日程，不是通过验证误差自动学习动态权重。
增加训练站加权有符号平均残差的平方，权重1。逆方差用于L1可靠度分配是明确的工程启发式，不宣称它是L1噪声模型下的最优似然权重，也不是预测不确定性估计。

观测残差：原有背景/辅助格点query的Cross Attention仍保留；追加仅用坐标query的同模块Attention，叠加训练站QA和距离prior，再以默认gate=0.5、局部覆盖因子加入embedding。局部尺度0.25是在归一化[-1,1]坐标上的固定参数。目的在于保留站点信息通路，不保证精度必然提升。

直接复用：

- [neuraloperator FNO/UNO](https://github.com/neuraloperator/neuraloperator)，本机1.0.2；[U-NO论文](https://arxiv.org/abs/2204.11127)。不修改第三方频谱层。
- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)，已有官方模块。不复制Senseiver/Perceiver网络。
- [Focal Frequency Loss](https://github.com/EndlessSora/focal-frequency-loss)，已有0.3.0；仍只约束可观测粗场频率，不用未知高频真值监督。
- [GSTools Ordinary Kriging](https://geostat-framework.readthedocs.io/projects/gstools/en/stable/api/gstools.krige.Ordinary.html)，复用已有SpatialFusionEngine；本机1.7系列。仅在本对照中启用训练站距离确定的Gaussian variogram长度上下界和10000次拟合上限，避免无界拟合不收敛；其他调用的默认行为不变。观测标准差平方传入cond_err。
- 现有CVXPY/OSQP普通约束QP、UnitEngine、结构Reader、rasterio、xarray和h5netcdf保持复用。

没有新增第三方依赖。本阶段保持上一版已经集成的频率/峰值/梯度/粗场一致性/非负约束。粗场约束是area投影软loss，不是严格地理面积守恒；最后显式裁剪负值并记录数量。不是污染输运PDE求解器、PINO完整复现或正式通用同化。

## 3. 文件与接口

| 文件 | 作用 |
|---|---|
| `src/ecodata/fusion/neural_reconstruction.py` | 权重、训练日程、bias项、现有网络调度、评价和元数据 |
| `src/ecodata/fusion/observation_encoder.py` | 同模块局部观测残差，不增加网络家族 |
| `src/ecodata/fusion/neural_inputs.py` | 真实栅格/科学数据、单位/日期确认、CSV数值字段映射 |
| `src/ecodata/fusion/neural_experiment.py` | 共用划分、六方法对照、评价、来源记录和输出回读 |
| `src/ecodata/engines/fusion.py` | 可选的训练距离variogram拟合边界；原默认不变 |
| `src/ecodata/interface/fusion_experiment_page.py` | 原实验入口内切换版本、真实上传、统一对照、下载 |
| `examples/run_observation_guided_experiment.py` | 固定参数的一键实验脚本 |
| `tests/test_neural_reconstruction.py`、`tests/test_observation_guided_experiment.py` | 原版兼容、权重、泄漏、真实读取接口、Attention、基线、回读 |

接口：`run_neural_reconstruction(..., config=NeuralReconstructionConfig(backend='observation_guided_uno'))`；`run_reconstruction_comparison(...)`；`read_experiment_grid(...)`；`read_experiment_stations(...)`。

CSV至少包含station_id、time、latitude、longitude、value、unit。上传页面支持手动映射PM25等实际数值列；缺少单位必须确认。qc_flag只接受VALID/INVALID/MISSING；quality_weight在(0,1]；observation_error为正标准差。已有split必须为train/validation/test，严格沿用，不重新分配；缺少split才按种子整站留出。

辅助气象连续变量双线性对齐，静态DEM广播到各时刻；分类土地利用必须明确声明categorical，采用最近邻和one-hot，最多32类。辅助场不覆盖目标区域、缺少背景时刻或存在未解释NaN时拒绝，不自动制造覆盖。默认目标网格是背景像元覆盖内4倍细化，不使用辅助场的大范围决定输出范围；边缘半像元内插值延伸也写入报告。NetCDF实际time bounds和变量单位保留，不能把粗场分辨率属性继续贴到细场。

## 4. 48小时实测结果

数据为原有P3模拟PM2.5：10km粗场、2.5km参考场、48时刻、12训练站、10独立验证站、480验证记录。所有神经方法40轮、种子20261009、学习率0.002、hidden_channels=16、observation_channels=8；其余完整参数在JSON。参考值只评价，参考坐标统一定义目标网格。未用验证指标搜索或调整参数。

最终输出：`output/observation_guided_experiments/20261009T145053261956Z/`。

| 方法 | 全场RMSE | 独立站RMSE | 独立站MAE | Bias | 全场峰值幅值MAE | 空间相关性 | 秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 双线性 | 3.4359 | 12.0126 | 8.2579 | -6.1438 | 30.7492 | 0.8309 | 0.85 |
| Ordinary Kriging | 118.5063 | 56.4363 | 30.6414 | 23.7351 | 341.6597 | 0.0660 | 6.16 |
| 普通QP | 1.8141 | 7.1784 | 5.0304 | -3.5292 | 4.2010 | 0.9553 | 1.78 |
| 原版FNO | 5.7086 | 6.3064 | 4.8090 | 2.5975 | 10.9346 | 0.8392 | 5.96 |
| 原版UNO+Attention | 4.8427 | 6.6725 | 4.8491 | 4.5281 | 3.6771 | 0.8640 | 4.92 |
| 观测引导UNO | 5.3455 | 7.1954 | 5.4866 | 4.3764 | 6.4077 | 0.8551 | 3.52 |

浓度误差单位ug/m3；运行时间为本机单次，含不同程度初次导入，不是公平性能基准。空间相关性为每时刻全场Pearson相关性的平均；全场峰值误差为每时刻最大值幅值差的平均，不是峰值位置误差。验证站峰值另有validation_peak_error，只比较各时刻留出站集合的最大值，不能当全域峰值。

**结论：新版没有同时改善三个目标。** 相对原版UNO，全场RMSE、独立站RMSE和全场峰值误差均变差；bias小幅下降，格点梯度MAE从1.4443降至1.3473。普通QP的全场重构最好，原版FNO的独立站RMSE最好。不得用本次结果声称新方法已胜出。

Kriging出现大范围异常外推，状态WARN；其默认无界拟合的首轮失败也保留在`20261009T143627905447Z`，没有删掉不利结果。最终训练距离有界策略是数值失败修复，不是根据验证RMSE挑选variogram。当前配置不适合作为这套合成数据的可靠Kriging参照；需要未来预注册训练内variogram诊断，不能据此宣称Kriging普遍差。

本案例所有训练QA/完整性相同且无观测误差列，可靠度权重实际均为1，因此不能证明异质QA/误差加权的实测贡献。本次对照也无实际辅助变量。权重公式和分类辅助输入由自动测试验证，效果仍需合适数据。

## 5. 已有真实数据的接口实验

只读取现有`examples/engineering_case_air_quality/`，不下载或修改原文件。

- 背景：NASA GEOS-CF PM25_RH35_GCC，3×3、8个小时平均时段；2024-01-01至2024-01-02 UTC，不是连续48小时。
- 观测：AirNow两个真实站、16记录，整站留出1站，8条独立记录；发布值仍为preliminary。
- 辅助：实际GEOS-CF temperature/humidity/wind及USGS 3DEP DEM。`era5_sample.nc`沿用旧名称，内容不是ERA5。DEM输入单位m根据原来源说明显式确认，投影EPSG:3857通过rasterio转换后使用。
- 原来源URL、元数据及SHA在原`source_manifest.json`；每次报告附输入文件SHA。GEOS-CF RH35条件与AirNow环境测量条件尚未做经验证湿度转换，两者不是完全等价物理条件。

| 神经方法 | 无辅助独立站RMSE | 加气象+DEM的RMSE | 加辅助MAE | 加辅助Bias |
|---|---:|---:|---:|---:|
| FNO | 10.2016 | 18.6214 | 15.3240 | 13.7631 |
| 原版UNO | 23.9226 | 20.1541 | 16.2979 | 15.3793 |
| 观测引导UNO | 23.2712 | 20.5935 | 16.8886 | 15.9649 |

无辅助输出：`output/observation_guided_experiments/20261009T145049741469Z/`。
加辅助输出：`output/observation_guided_experiments/20261009T145350214576Z/`。

两次都运行同样六种基线、同样划分与参数。Kriging因每时刻仅1个训练站而FAIL：原有GSTools门槛至少8站；没有伪造站点/variogram或换算法冒充通过。其他方法输出有限值，但均有训练观测范围外推WARN。无独立参考网格时，全场RMSE/空间相关性/全域峰值误差均为null，不伪造指标。

真实实验仅证明读取→字段映射→单位→共同时间→CRS/辅助对齐→训练→独立站预测→NetCDF/CSV/JSON输出能运行。1个训练站不够检验空间泛化、辅助变量作用或算法优越性。FNO加辅助反而变差，原版UNO略改善，新版仍未优于原版UNO。早期`20261009T144202439528Z`采用不同目标网格范围，保留但不混入以上同覆盖比较。

## 6. 普通用户怎么测试

打开现有开发版平台 → 左侧“融合重构实验” → 内置PM2.5样例 → “高级设置” → 重构版本选“Observation-guided UNO（实验版本）” → 点击“运行融合重构实验”。

需要统一对照时，在高级设置勾选“运行统一基线对照”，再运行，下载“全部对照结果”。ZIP包含六种成功计算的NetCDF、comparison.csv、validation_predictions.csv、experiment_report.json；FAIL方法只保留原因，不生成假结果。

自己的真实数据：切换“上传自己的数据”，上传背景NC/TIFF、监测CSV及可选辅助场。背景无时刻必须确认真实时刻，无单位必须确认单位；多变量科学文件选择实际变量，分类辅助声明分类。CSV没有value时选真实观测字段。不能默认留出20站而只有2站；应先明确实验划分，或在高级设置将留出站数设为1。若已有split则沿用，不受这个数字影响。

命令复现内置案例：

```powershell
.\.venv\Scripts\python.exe examples/run_observation_guided_experiment.py
```

真实背景和站点（不带辅助）的完整复现：

```powershell
.\.venv\Scripts\python.exe examples/run_observation_guided_experiment.py --background examples/engineering_case_air_quality/background/pm25_background.nc --stations examples/engineering_case_air_quality/observation/pm25_station.csv --station-value-column PM25 --validation-stations 1
```

辅助多变量文件可在页面多选temperature/humidity/wind并上传dem.tif、确认m；或通过read_experiment_grid显式选择后传auxiliaries。分类/多变量/缺单位的确认由页面提供，命令行不自动猜。

## 7. 质量与科研边界

- PASS表示方法运行和输出基本检查通过，不表示精度提升；WARN指出输出可能不稳定或不能证明全场精度；FAIL不给该方法生成结果。
- 输出执行有限值/非负/单位/独立站检查，保存后用h5netcdf重新读取并与内存数据比较。不是官方模型输入验证。
- 验证站值、QA、误差和参考值不进入训练权重/Attention/loss；自动测试将其改动后得到同样输出。没有在评价后调参或挑选更有利种子。
- 没有跨区域预训练模型、独立时间测试、多随机种子置信区间或正式同化。当前是场景内在线拟合原型，不保证随意查询坐标的零样本连续operator能力。
- 本阶段不把实验版设为默认、不提高科研或模型适配状态；也没有重新发布旧Portable包。
- 按academic-paper-writing-skill的科研证据纪律保留负结果、来源、独立划分和未完成项。

## 8. 测试记录

全量pytest实际执行：**1010通过、4跳过、0失败，126.65秒，226条警告**。4跳过为：两项旧工程最终输出尚未生成；真实xESMF缺少ESMF；EPA profile/PseudoNetCDF专用独立环境未配置。跳过不计作通过，不为了本阶段测试新下载或重跑无关工程。

新增覆盖训练QA/完整性/误差权重、非法σ和重复记录、NetCDF多变量确认、真实CSV字段映射、GeoTIFF翻转/重投影/单位/真实日期确认、投影科学文件time bounds保留、粗网格像元覆盖保持、分类辅助one-hot及时间不兼容提示、六方法相同网格/划分、导出回读和禁止覆盖旧结果。上游库弃用警告和QP既有求解精度警告未掩盖，未改第三方源码。

全回归命令：`.venv\Scripts\python.exe -m pytest -q -rs -o addopts='' -p no:cacheprovider --basetemp=D:\codex\ecodata_guided_final_regression_v2`；临时目录可替换为本机新目录，业务代码没有依赖该绝对路径。

Streamlit AppTest实际操作现有首页→融合重构实验→选择新版→运行→勾选统一对照→运行：两个流程exception/error均为0，对照表6行。单结果下载ZIP包含NC/独立站CSV/报告JSON，NC成功重新打开为float32，48×24×24；显示验证RMSE7.1954。没有把AppTest等同实际浏览器渲染或模型运行验证。
