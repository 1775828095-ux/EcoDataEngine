# P1-P2 dependency record

## 2026-10-09：Neural Reconstruction

直接调用[neuraloperator](https://github.com/neuraloperator/neuraloperator) 1.0.2的FNO/UNO（MIT），PyTorch 2.14.1的MultiheadAttention（BSD式许可证，保留上游声明），以及[Focal Frequency Loss](https://github.com/EndlessSora/focal-frequency-loss) 0.3.0（MIT，Jiang等ICCV2021）。没有复制网络源码。安装`.[ml]`即可；详细来源、适用边界与实测见`neural_reconstruction_upgrade.md`。

后续观测可靠度/局部Attention残差实验没有新增依赖，仍复用上述模块、现有GSTools、rasterio、UnitEngine和QP。新增的GSTools训练距离有界variogram拟合是可选项，原默认未改变。六方法统一对照、真实气象/DEM输入及未改善的实验结果见`observation_guided_neural_reconstruction.md`；不把模块接通称为科研效果已验证。

Verified in the P2 test environment on 2026-09-29.

| Project | Tested version | Purpose | License | Official project |
|---|---:|---|---|---|
| NumPy | 2.3.5 | arrays and deterministic linear interpolation | BSD-3-Clause | https://numpy.org |
| Pandas | 3.0.1 | CSV parsing, timestamps and grouped observations | BSD-3-Clause | https://pandas.pydata.org |
| Xarray | 2026.7.0 | unified Dataset representation and NetCDF interface | Apache-2.0 | https://xarray.dev |
| h5netcdf | 1.8.1 | explicit NetCDF read/write engine | BSD-3-Clause | https://h5netcdf.org |
| h5py | 3.16.0 | HDF5 backend used by h5netcdf | BSD-3-Clause | https://www.h5py.org |
| pytest | 9.1.1 | automated tests | MIT | https://pytest.org |
| SciPy | 1.18.1 | station mapping and residual interpolation | BSD-3-Clause | https://scipy.org |
| xarray-regrid | 0.4.2 | nearest, bilinear and conservative rectilinear regridding | Apache-2.0 | https://github.com/xarray-contrib/xarray-regrid |
| flox | 0.11.2 | required xarray-regrid dependency | Apache-2.0 | https://flox.readthedocs.io |
| numpy-groupies | 0.12.3 | required flox dependency | BSD-3-Clause | https://github.com/ml31415/numpy-groupies |
| toolz | 1.1.0 | required flox dependency | BSD-3-Clause | https://github.com/pytoolz/toolz |
| CVXPY | 1.9.3 | sparse convex quadratic-program construction | Apache-2.0 | https://www.cvxpy.org |
| OSQP | 1.1.3 | constrained downscaling QP solver | Apache-2.0 | https://osqp.org |
| Matplotlib | 3.11.2 | static comparison figures and UI previews | PSF-based | https://matplotlib.org |
| Streamlit | 1.64.0 | minimal local P3 test interface | Apache-2.0 | https://streamlit.io |
| psutil | 7.2.2 | process RSS peak sampling | BSD-3-Clause | https://psutil.readthedocs.io |
| PyMuPDF | 1.28.2 | rendered PDF collision QA | AGPL-3.0-or-later OR commercial | https://pymupdf.readthedocs.io |

Versions above are the actual versions used for the recorded P2 verification.
Runtime compatibility ranges are declared in `pyproject.toml`; this table is
not a lock file. The paragraph above is a historical P2 record. Current
integrations and citations are recorded below, not retroactively claimed for P2.

## 2026-10-02：成熟开源集成实测版本

所有集成使用安装依赖和官方 API；没有复制项目源码后改名。版本从实际安装 distribution metadata 读取，许可证核对对应安装包的 metadata/LICENSE。

| 项目 | 实测版本 | 许可证 | 当前用途和状态 | GitHub / 引用 |
|---|---|---|---|---|
| cf-xarray | 0.11.3 | Apache-2.0 | SemanticEngine：CF axes、coordinates、bounds、standard_names、grid_mapping；已实测 | [仓库](https://github.com/xarray-contrib/cf-xarray)、[软件 DOI](https://doi.org/10.5281/zenodo.4749735) |
| pint-xarray / Pint | 0.6.1 / 0.26.1 | Apache-2.0 / BSD-3-Clause | UnitEngine：维度相容性、量化/换算/反量化，温度偏移、浓度、速度、百分数；已实测 | [pint-xarray](https://github.com/xarray-contrib/pint-xarray)、[Pint](https://github.com/hgrecco/pint)；引用项目和实测版本，未核实论文 DOI，不杜撰 |
| xESMF | 0.9.2 | MIT | GridEngine 首选后端接口已接入；当前缺少 ESMF，**未执行真实 xESMF** | 用户旧仓库名 JiaweiZhuang/xESMF 对应现维护的 [pangeo-data/xESMF](https://github.com/pangeo-data/xESMF)；[官方软件引用入口](https://zenodo.org/badge/latestdoi/281126933) |
| GSTools | 1.7.0 | LGPL-3.0 | Ordinary / Universal / External Drift Kriging、训练 variogram 自动拟合、CondSRF；已实测 | [仓库](https://github.com/GeoStat-Framework/GSTools)、Müller et al. 2022, [10.5194/gmd-15-3161-2022](https://doi.org/10.5194/gmd-15-3161-2022) |
| compliance-checker | 6.1.0 | Apache-2.0 | ValidationEngine：实际输出文件 CF 1.8 / 适用时 ACDD 1.3 检查，保留原始详细报告；已实测 | [仓库](https://github.com/ioos/compliance-checker)、[官方调用示例](https://github.com/ioos/compliance-checker/blob/main/README.md)；未核实论文 DOI |
| xskillscore | 0.0.29 | Apache-2.0 | 明确独立真值才计算 RMSE / MAE / correlation；bias为明确 prediction-reference 均值；已实测 | [仓库](https://github.com/xarray-contrib/xskillscore)、[软件 DOI](https://doi.org/10.5281/zenodo.5173152) |
| PseudoNetCDF | 3.4.1 | LGPL-3.0，见安装包 LICENSE | CMAQAdapter 调用独立 reader，已读取官方旧版 ICON profile；**未验证新版原生 CMAQ NetCDF** | [仓库](https://github.com/barronh/PseudoNetCDF)、[CMAQ API](https://barronh.github.io/pseudonetcdf/api/PseudoNetCDF.cmaqfiles.html)；未核实论文 DOI |

### 安装和兼容性

- 主环境维持 NumPy 2.3.5、Pandas 3.0.1、SciPy 1.18.1，没有为了旧版模型工具降级平台。
- 当前验证环境使用 PseudoNetCDF 3.4.1；它属于 `verification` 可选依赖，不作为生产 Writer。原独立 `.venv-pnc` 已在项目清理时移除，现有测试直接调用主验证环境中的独立读取路径。
- 主依赖已写入 pyproject.toml。xESMF 是 optional extra，ESMPy/ESMF 需另外安装。[官方 Windows 安装限制](https://xesmf.readthedocs.io/en/stable/installation.html)指出通常使用 Linux/WSL 或独立 conda 环境。本阶段没有擅自安装 WSL、Docker 或大型编译工具。
- 当前 `import xesmf` 的真实失败原因是 `No module named 'ESMF'`。GridEngine 不会把 fallback 或模拟调用冒充 xESMF。
- 现有 xarray-regrid 0.4.2、rasterio/PROJ、CVXPY/OSQP、残差融合、反方差融合继续保留。安装 sparse 后 xarray-regrid 可能返回稀疏对象，封装层显式转为已有 Dataset/NetCDF 要求的数组；不改变权重算法。

### 仅研究架构，不作为运行依赖

- JEDI：按 [UFO observation operators / variable mapping](https://jcsda-jedi-docs.readthedocs-hosted.com/en/develop/inside/jedi-components/ufo/obsops.html) 和 [IODA 接口](https://jcsda-jedi-docs.readthedocs-hosted.com/en/latest/inside/jedi-components/ioda/interface.html) 借鉴状态、观测算子、变量映射与误差的分离。未安装，未核实整个多仓库系统的统一许可证，不假设一个许可证覆盖全部组件。
- [NCAR/DART](https://github.com/NCAR/DART)，Apache-2.0；参照[能力说明](https://docs.dart.ucar.edu/en/latest/guide/important-capabilities-dart.html)，不安装或调用。
- [nansencenter/DAPPER](https://github.com/nansencenter/DAPPER)，MIT、部分组件另有许可；参照其 model / observation / twin experiment 结构，不安装或调用。引用官方仓库，未核实本次对应软件 DOI。
- `AssimilationContext` / `AssimilationEngine Protocol` 仅定义 can_run / prepare / assimilate / validate。没有实现 EnKF、3DVAR、4DVAR 或正式通用同化。

### CMAQ 规范与样例范围

对照 [ICON](https://github.com/USEPA/CMAQ/blob/main/PREP/icon/README.md)、[BCON](https://github.com/USEPA/CMAQ/blob/main/PREP/bcon/README.md)、[MCIP](https://github.com/USEPA/CMAQ/blob/main/PREP/mcip/README.md)。官方 profile 下载地址、版本、字节数、SHA256 在 `examples/open_source_integration/cmaq_official/source_manifest.json`。

官方样例是 5.0.2 ASCII profile，不是 5.5 原生 I/O API NetCDF；不能用旧版 profile 读取通过来声称新版 ICON/BCON/MCIP 已适配。没有下载数 GB 基准包。原生网格、时间、TFLAG、化学机制及层结构闭环仍需对应版本官方小型文件验证。

## 2026-10-03：WASP 上游原始数据 Reader

| 项目 | 安装版本 | 许可证 | 使用范围 |
|---|---|---|---|
| [DOI-USGS/dataretrieval-python](https://github.com/DOI-USGS/dataretrieval-python) | 1.4.0 | CC0 / 美国政府公共领域，见官方 LICENSE.md | USGSRDBReader 直接调用 dataretrieval.rdb.read_rdb；不复制 RDB parser |
| [haydenth/ish_parser](https://github.com/haydenth/ish_parser) | 0.0.25 | MIT | NOAAISDReader 调用 ish_report.loads 读取当前案例的固定宽度 ISD 原文；保留原始记录与 QC，不自行实现字符位置规范 |
| [Vonng/isd](https://github.com/Vonng/isd) | 仅研究 main 分支，未安装 | MIT | 研究了 parser/isdh；该 Go hourly 实现面向 NOAA CSV，而当前案例是原始固定宽度 ISD gzip，未直接集成，也不把 ish-parser 冒充该项目 |
| pandas | 复用项目现有安装版本 | BSD-3-Clause | ExcelFile/read_excel(sheet_name=None)、多 sheet 评分、read_fwf；不增加复杂 Excel 依赖 |

上述 Reader 均使用安装依赖的 API。未核实专门的论文 DOI，不杜撰。ish-parser 的附加段质量码并非全部由 API 独立解码，原始附加段和原文保留，报告明确这一限制。WIF 没有新增 parser 或 writer。

## 2026-10-03：WASP 官方文本路径研究

- EPA WASP8 8.5：下载官方 Windows ZIP，作为外部验收程序，不加入 pip 依赖，不逆向二进制。[官方下载](https://www.epa.gov/hydrowq/wasp8-download)。来源哈希见 Narragansett official_adapter_reference/source_manifest.json。许可范围以 EPA 发行说明为准，未将其宣称为某种开源许可证。
- [respec/BASINS atcWASPInpWriter](https://github.com/respec/BASINS/tree/master/atcWASPInpWriter)：只参考公开 INP 文本布局，不复制/编译/集成完整 VB Builder。当前 Writer 为候选实现，未通过官方导入。尚未核实仓库统一许可证，不宣称已接入可再分发的 BASINS 库。
- Watershed/Steady State/Toxics 官方 ZIP 保留为参考。WASP 运行时原件不改，独立故障排查副本仅修改 consoleDefault 页面配置，不修改模型数据或算法。

## 2026-10-07：模型工具复用审计与 FloPy 接入

| 项目 | 实测版本 | 许可证 | 当前用途和状态 | 官方来源 |
|---|---:|---|---|---|
| FloPy | 3.11.0 | USGS public domain；第三方组件以各自声明为准 | 新增范围受限的 MODFLOW 6 单层结构网格 GWF 输入生成；已通过 FloPy 回读和数组核对，未运行官方 `mf6` | [GitHub](https://github.com/modflowpy/flopy)、[文档](https://flopy.readthedocs.io/) |

接入仅覆盖用户明确提供的 DIS、IC、NPF、RCHA、CHD、TDIS、IMS 和 OC
所需信息。缺少网格大小、时间参数或有效定水头边界时拒绝生成；不推断井、
河流、多层结构、工程映射或率定参数。四级状态单独记录在输出 manifest：
标准化预处理、原生格式输出、官方程序读取、官方模型运行。

## 2026-10-13：统一算法 Registry 与超分辨率候选审计

本轮没有复制或下载候选项目源码/权重。只有现有依赖真实执行通过的算法标记为
`available`；图像超分辨率候选只登记元数据和接入边界。

| 项目 | 官方实现与许可 | 当前状态 | 不能直接作为环境场算法的原因 |
|---|---|---|---|
| ZSSR | https://github.com/assafshocher/ZSSR；WIS/Yeda Academic Non-Commercial | `research_only` | 官方入口以 PNG 单图和内部训练为主；非商业许可限制分发/商用；尚未接入 NetCDF 单位、CRS、NoData 和反向聚合验证 |
| Thera | https://github.com/prs-eth/thera；Apache-2.0 | `dependency_missing` | 官方环境要求 Linux、Python 3.10、NVIDIA GPU/JAX；公开自然图像权重不能当作环境物理先验 |
| LIIF | https://github.com/yinboc/liif；BSD-3-Clause | `research_only` | 官方 PyTorch 1.6/DIV2K 路径和自然图像权重尚未完成环境变量语义、物理约束与独立真值验证 |
| DIIN | https://github.com/Song-1205/DIIN；未见许可证 | `research_only` | 官方仓库当前仅显示“Code coming soon”，无可执行代码，不能接入或再分发 |

已接入 Registry 的实际后端仍是项目现有 xarray-regrid、SciPy、GSTools、
CVXPY/OSQP、PyTorch/neuraloperator 及已有融合模块。Registry 只是统一调用和守门，
没有重新实现它们的算法核心。
