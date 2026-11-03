# 背景场与高斯过程残差修正接入

## 选择与范围

新增 `fusion.gp_residual`，复用已安装 scikit-learn GaussianProcessRegressor，BSD-3-Clause。
GitHub: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/gaussian_process/_gpr.py
官方说明: https://scikit-learn.org/stable/modules/gaussian_process.html
检索同时检查了 SciPy RBFInterpolator。优先采用GP，是为了补充背景与有噪站点的正则化协作及条件残差方差；不是声称GP更准确或其方差已经校准。
不新增依赖、不复制上游科学内核、不修改已有评分政策或放宽学习模型支持域。

## 数值机制

用训练角色的同一时刻站点观测减去背景在站点处的双线性采样，得到残差 r。
固定核 K = signal_std² × RBF(length_scale_m)，噪声方差为 noise_std²。
修正量为 K(query,stations) [K(stations,stations)+noise_std² I]^-1 r。
最终输出为背景在目标网格的插值加修正。零均值残差先验下，离站点足够远时修正趋于零；不代表域外精度保证。
三个参数必须显式提供，长度单位米、两个标准差与变量同单位。optimizer=None，不读取参考真值或现场调优。

## 接口与验收

通过默认注册表、submit_search、Profile冻结池、原数值执行器和质量门使用。
primary/background角色提供一个连续二维EPSG:4326背景；observations角色提供训练站表。
背景须有单一time坐标；站表须包含longitude、latitude、value、split、unit、variable、time。
variable须与背景physical_variable一致。拒绝非train标签、不同时间/单位/变量、重复/不足/共线站点、背景覆盖外采样。
复用已有station_inputs的米制投影；最多1000站以限制稠密GP成本。
参数：observation_crs、gp_length_scale_m、gp_signal_std、gp_noise_std。
背景和站点必须已具有可比较的物理定义，不能把GEOS特殊层浓度直接等同地面PM2.5。

产品保留background_value、residual_correction、residual_gp_std；后者仅为固定核下潜在残差条件标准差，不含背景、参数误差，不接入逆方差融合权重。
算法画像只声明实际可执行能力，不增加精度、非负、值域或守恒保证；现有评分器未知性质保持未知。
新增测试核对独立显式矩阵解、局部衰减、原输入保护、参考隔离、非法资源拒绝，以及公共Profile推荐、执行和产品回读。
合成接口检验不是环境效果研究；未运行新的真实数据优越性实验。

最终定向与回归：gp_residual_tests_release.xml为48项通过，gp_residual_fusion_regression.xml为7项通过，合计55项不同测试。
随后保留输出标量time坐标，新算法10项重跑通过（gp_residual_tests_time.xml），不累加成65项。
首轮公共入口测试夹具遗漏模型字段、输入封装等问题，以及注册表83→84的旧数量断言失败均保留在中间产物中；未通过放宽科学质量门修复。

## 后续工程任务与限制

候选用途：已完成物理定义校准的地面PM2.5背景 + 同期可用站点残差修正；保留原背景、原残差融合及纯站点插值对照。
新增的是可调用的协作算子，内部背景采样/残差/拟合/相加尚不是四个可独立搜索替换的公共节点。
未贯通专用station_covariates Profile入口，也未证明所有算法任意连接。
旧冻结候选池及旧学习模型不自动扩展；加入新算法需要新池版本和开发记录。真实任务前仍需审核校准背景的来源、时间支持和未使用评价数据。
