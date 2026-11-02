# 2026-10-06 持续开发记录

本批尚未完成五模型全部输入。输出解析与数值检查不是仿真验收；没有部署或运行模型。

## SWAT+62 显式对象连接

入口：首轮专项转换 → SWAT+ → 显式对象连接，CLI mode=connections。
ZIP含节点CSV、出流CSV、各类实际属性数据库和weather-sta.cli；JSON逐字段映射。
支持hru/aqu/ru/sdc，生成hru.con/aquifer.con/rout_unit.con/chandeg.con中实际存在对象的文件。
节点编号和属性库顺序须连续1..N，props是读取器数组下标；检查天气站名称、出流端点、每类水文比例和非循环拓扑。
面积转ha，高程转m，EPSG:4326或明确米制投影坐标转换到经纬度；静态资料没有时间字段。
需用户给真实节点/拓扑，不划分流域/HRU、不推导工程参数、不生成file.cio。当前仅cst/ovfl/rule=0；深层属性库引用和高级规则仍未实现。

官方依据：[62.0.0 hyd_read_connect.f90](https://github.com/swat-model/swatplus/blob/62.0.0/src/hyd_read_connect.f90)、[hyd_connect.f90](https://github.com/swat-model/swatplus/blob/62.0.0/src/hyd_connect.f90)。
examples/ai_round1/swat_plus/connections的official输入由[Osu_1hru](https://github.com/swat-model/swatplus/tree/62.0.0/refdata/Osu_1hru)已有输入反解，B2原文件头为旧Editor2.2/rev60.5.4，已按62读取代码核对；不能冒充原始GIS观测。
synthetic_projected是明确人工构造的不同列名/顺序、EPSG3857、m²/百分比样例。两组有独立逐字段/数值及引用检查，真实原始流域数据证据仍不足。官方文件仅用于测试；新writer独立实现，没有复制Fortran算法。

## WPS4.6 Lambert intermediate-v5

入口仍为WPS多变量气象场；单海温入口保持原样。mode=pressure_fields，显式projection和crs。
球形Lambert IPROJ3接受一维等间距x/y（m或km）、UTC时间及已有pressure/surface/soil变量映射；源CRS必须与声明球半径/真纬度/中央经线一致。
格距和半径写km，地理起点为第一个格心SWCORNER；不把格边当格心。坐标排序与单位换算外不插值、不重网格、不旋转风。风仍必须earth_relative。
配置样例见examples/ai_round1/wrf/lambert，两个均为合成数据（米/公里、不同列名/维序/倒序）。scipy FortranFile独立读取记录并与预设解析坐标和原始数值比较；float32温度绝对误差阈值预定3e-5 K。
官方依据：[WPS4.6 read_met_module.F](https://github.com/wrf-model/WPS/blob/v4.6.0/metgrid/src/read_met_module.F)。需实际源球形CRS，不允许WGS84椭球LCC冒充。
没有运行metgrid；需geo_em、匹配METGRID.TBL、其他实际必需变量及时间层；本功能不生成完整WRF工程。真实Lambert原始案例待补，不能称全面验证。

## 复测与依赖

运行run_tests.bat或python tools/run_regression.py。CLI通用形式：python -m ecodata.round1 --model swat_plus（或wrf） --input A文件 --config 配置JSON --output 新目录。
verification_tools是本地测试专用依赖geographiclib2.1、PseudoNetCDF3.4.1，保留各dist-info内许可证；不运行模型。requirements-verification.txt记录版本。既有运行时pip内部版本冲突未用来安装包；测试入口直接加载本地已核对的测试依赖。

## CALMET5.8.5 GEO.DAT

新增地理参数网格入口，mode=geography。NetCDF明确x/y格心、EPSG326xx/327xx、八字段映射：土地类别、地形m、粗糙度m、反照率、Bowen比、土壤热通量参数、人为热通量W/m²、叶面积指数。无时间维；无推算默认值。
GEO.DAT固定表头输出UTM/WGS-84、格边原点与格距km/f12.3（不能无损表达1m时拒绝），2..265方格轴，最多52类别，与5.8.5默认编译上限一致。北到南行、西到东列，六项地表参数全部IOPT=2显式网格。
格式依据：官方CALMET5.8.5 CODE/calmet.for READGE/FILLGEO及PARAMS.MET，已存本地官方源码证据。官方发布页 https://www.calpuff.org/calpuff/download/download.htm 。本实现没有复制模型计算代码，不部署或运行CALMET。
examples/ai_round1/calpuff/geography两组仅为合成测试（南北半球、不同维序、米/公里和百分比）。std库独立逐行固定列/数值解析，所有8字段与预设数值误差<=1e-12。真实完整地理参数A不足，因此标已实现但真实证据不足。仍需CALMET.INP网格和投影匹配；无控制/排放/受体工程。

## CAMx7.40 INITIAL三维浓度

mode=initial_concentrations，CSV完整单时刻layer×row×col；映射列名和0/1索引基准，机制CB6r5_CF3_SIMPLE_NoEvap。气体ppmV/ppbV/mol/mol转换到ppm，气溶胶质量浓度转ug/m3。LATLON/LCC复用已有网格校验。
必须给layer_count、layer_definition来源、required_species及明确带时区valid_until，源时刻和有效期标签转固定ITZON。官方ncf_readcnc调用ncf_get_tstep严格要求TFLAG<=开始时刻<ETFLAG，因此不使用两者相等的假成功文件，也不替用户延长有效期。仅导入初始快照，非时间插值或模拟。
examples/ai_round1/camx/initial两组为合成LATLON/LCC、2/3层、不同列名/索引/单位及倒序。独立SciPy读回全部浓度、维度、UTC跨年换算和时间范围。相对误差容差1e-7，绝对误差1e-10为预设float32测试阈值。
格式依据：[CAMx7.40 260801官方源码](https://www.camx.com/getmedia/CAMx_v7.40.src.260801.tgz)的IO_NCF/ncf_cncprep.f、ncf_readcnc.f、ncf_get_tstep.f、ncf_chk_griddef.f。原包约1.8MB，仅读取相关源文件和LICENSE；未把Ramboll模型计算代码复制到平台。其许可不是无条件MIT，原软件使用/再分发受自带协议约束。新writer按接口独立实现。
仍缺真实浓度A的独立案例；目标模型未实际读取。不提供气象、边界、CHEMPARAM或控制工程。

## CMAQ5.5 显式BNDARY3边界浓度

mode=boundary_concentrations，NetCDF包含源目标网格内x/y、每个边界点column/row和实际x/y、time×layer×perimeter各物种浓度；外圈厚度NTHIK=1。
[CMAS官方边界定义](https://www.cmascenter.org/ioapi/documentation/all_versions/html/DATATYPES.html#bndary)及[官方顺序图](https://www.cmascenter.org/ioapi/documentation/all_versions/html/BDY.jpg)：南边西到东、东边南到北、北边西到东、西边南到北，包含各自规定角点；并非简单顺时针。可重排原点索引，但不插值、填角或补浓度。
[BCON5.5 opn_bc_file.F](https://github.com/USEPA/CMAQ/blob/5.5/PREP/bcon/src/common/opn_bc_file.F)规定FTYPE2、网格、垂直及时间属性。当前支持VGTYP7，源属性VGTYP/VGTOP/VGLVLS须与配置相符；源层按1..N编号；至少2个真实UTC连续小时。机制cb6r3_ae7_aq气相/气溶胶质量物种映射，required_species不得缺项。不支持数浓度和表面积物种，不默认为其填值，因此不宣称该有限物种表覆盖所有机制需要。
两组examples/ai_round1/cmaq/boundary均为合成LATLON/LCC、2/3层和不同维序，手工按官方图列14点期望顺序，SciPy逐值核对并用PseudoNetCDF ioapi读回。相对误差1e-7/绝对1e-9的float32预设阈值。PseudoNetCDF退出时可能输出重复close的Not a valid ID警告，不影响已执行的数组读取；保留原日志。
这是用户提供现成外圈物理资料后的编码功能；不是从BC_PROFILE推导浓度或运行BCON。真实边界场原始数据案例未验证，MCIP网格/层、全机制和控制文件仍需项目提供。未运行BCON/CCTM。

## 同时修复

CMAQ共享hourly时间校验原使用asi8并假定纳秒，合法datetime64[s]会误报缺小时。现按timedelta64比较，秒/毫秒/微秒/纳秒四组回归；未更改时间或浓度值。未解码数值日期明确拒绝。

## WPS4.6 geogrid上游地形/土地分类瓦片

mode=geogrid_tiles；原始二维静态NetCDF明确变量和经纬轴，EPSG4326、规则经纬度。HGT_M源单位转换到m；LANDUSEF源必须是已经分类的整数码（单位1），明确类别范围、MMINLU及水/湖/冰/城市编码。两类各有标准/异构维序和单位的合成样例，不冒充真实地形或遥感原始数据。
输出geogrid_HGT_M.zip或geogrid_LANDUSEF.zip，内部index及00001-NX.00001-NY原始整数瓦片。known_x/y=1对应第一个源网格中心点；南到北行、西到东列；2/4字节有符号，大/小端显式选择。地形量化scale_factor及最大绝对误差须事先声明，超阈值拒绝；类别码无量化。缺测/缺格/时间维/单位冲突拒绝。ZIP元数据采用固定1980时间以保证重复输出字节一致，不作为数据观测日期。
[WPS4.6 source_data_module.F](https://github.com/wrf-model/WPS/blob/v4.6.0/geogrid/src/source_data_module.F)定义index；[read_geogrid.c](https://github.com/wrf-model/WPS/blob/v4.6.0/geogrid/src/read_geogrid.c)定义整数/字节序及float32缩放。测试用独立stdlib struct读回，核对index、参考格心、完整字节数和预设数值；真实独立地理案例仍待验证。
这是地理数据上游转换。用户仍须已有GEOGRID.TBL正确字段节/插值规则、目标域覆盖及其它地理数据；单独转换不运行geogrid，也不生成geo_em。后续登记工程入口的原生验证另列，不能把瓦片解析通过写成完整WRF地理域完成。

2026-10-26补充：LANDUSEF允许显式representation=fraction_layers、category_coordinate指定类别轴。输入必须为完整连续类别坐标的静态三维整数百分比（单位%），每格合计100%，每层0..100，scale_factor=1、允许量化误差=0；缺层、乱序、非完整合计拒绝，不归一或取最大类别。输出continuous、tile_z为类别数，保留完整逐层整数。真实NCAR MODIS21裁剪335993个混合源像元逐值无损，经geogrid→real→WRF三分钟及当前ZIP已有/新建两路线通过；输出7652个混合网格、21层比例。证据：项目夜间续跑_20261025/WRF真实比例土地利用与WRF土地利用当前ZIP两路线。仅登记范围，不支持任意新区域或分类体系。

## CALPUFF5.8.4 group17离散受体

mode=receptors：ZIP只包含CSV和用户已有实际CALPUFF控制文件。五字段映射id/x/y/elevation/height_agl；x/y转km，地面MSL海拔和地上高度转m；源坐标须与活动控制字段PMAP/IUTMZN/UTMHEM/DATUM匹配，当前支持WGS84或NAD27北半球UTM，不做基准转换。检查NX/NY/DGRIDKM/XORIGKM/YORIGKM覆盖、唯一id、缺项/单位/缺测/高度/原子组数，11000条默认编译上限。需事先声明原生float32米制允许误差。
[官方CALPUFF5.8.4](https://www.calpuff.org/calpuff/download/EPA_Files/CALPUFF_v5.8.4.zip) CODE/calpuff.for READCF IG17及cvdic、params.puf定义四值X数组和每条END。不是另造“受体.dat”文件：在原控制文件中只替换NREC和最后group17b，前文原样保持；未知后续活动记录明确拒绝。
examples/ai_round1/calpuff/receptors使用原官方CPUF.INP控制格式，两组受体均为合成：NAD27米制、映射列名的WGS84公里制/海拔公里。第二组控制基准也明确改为WGS84，仅用于格式测试，不能把旧示例气象视为匹配。独立split !活动子组读取，核对四值、计数、终止符、NREC之前原控制文件逐字不变。没有真实受体A/B2案例，证据不足；没有验证控制文件内其它物理参数和外部文件引用，更没有运行CALPUFF。
CALMET5.8.5与CALPUFF5.8.4分别在各自入口标明，不能混称同一格式版本。

两个地形入口另明确要求elevation_vertical_reference=MSL；无大地水准面改正功能，椭球高源需上游给有依据的海拔转换。

## SWAT+62 土地利用与已有HRU空间关联

mode=spatial_landuse，ZIP内HRU属性CSV（除lu_mgt的九字段）、已有HRU及地类GeoJSON面、真实引用数据库。JSON配置字段、属性名、source_crs/coordinate_unit=m、地类代码→landuse.lum名称和homogeneous_full_coverage策略。仅同一米制投影，固定几何运算相对容差1e-10（不是补缺口）；每HRU需完整单一地类覆盖，可由多个相邻同类面覆盖。混合、缺口、重叠、NaN、错误CRS、身份不一致、缺引用拒绝；不取多数、不拆HRU、不推导参数。复用已有export_hru_bundle，输出hru-data.hru和landuse_HRU_association.json，不提供file.cio。

目标依据：[hru_read.f90](https://github.com/swat-model/swatplus/blob/62.0.0/src/hru_read.f90)，lu_mgt需真实数据库名。examples/ai_round1/swat_plus/spatial_landuse两组几何均为合成，官方原有属性库只作引用格式。独立token行与手算5000m2核对；没有独立真实GIS A/B2。本批22项针对性测试及HRU/CLI/界面组合64项通过。时间静态，不涉及缺日补值。数据库内部深层引用/真实参数未认证。

## WPS4.6 已有v5场一致组包

mode=input_package，version=WPS-4.6-intermediate-v5-package。上传已有原生场ZIP，明确expected_times_utc、interval_seconds、required_levels（Pa等真实WPS层标签）、missing_values及prefix。检查五条Fortran记录/头/单位/层/格点数量、IPROJ0/3投影、网格完全一致、earth-relative风、连续时刻、所有必需字段和各时刻层集合。组包保留每场原生字节，不重网格、不补字段、不复制日/快照、不插值。输出PREFIX_日期_小时.wps；在WPS主机改名PREFIX:日期_小时后合入既有metgrid fg_name。Windows文件名不能含冒号，所以交付名保留.wps并说明改名步骤。

[官方read_met_module.F](https://github.com/wrf-model/WPS/blob/v4.6.0/metgrid/src/read_met_module.F)。examples/ai_round1/wrf/input_package两组原生记录按官方布局独立构造，均为合成已有格式场，不是原始观测。SciPy FortranFile独立核对温度解析式、日期/单位/层/投影及错误。相关76项回归通过。仍需geo_em、METGRID.TBL、完整项目驱动，不生成met_em/wrfinput，未运行metgrid。

## CAMx7.40 BOUNDARY 全数组输入

mode=boundary_concentrations，input_layout=full_grid_3d；字段time/layer/row/col与物种映射复用INITIAL。至少3x3网格，1..200层，明确start_time与valid_until，完整连续小时3600秒，至少两小时；不能缺前/后时刻、层、格或项目必需物种。气体转ppm，气溶胶转ug/m3；不做质量浓度到混合比的无依据换算。逐小时保留供给值，常规NetCDF3文件后端避免padding随机字节。

[官方源码包](https://www.camx.com/getmedia/CAMx_v7.40.src.260801.tgz) IO_NCF/ncf_bndprep.f、ncf_readbnd.f：CAMx_NAME=BOUNDARY，nf_get_vara_real读取全COL*ROW*LAY，再使用内部边缘行/列；并非CMAQ ghost perimeter。因此本profile要求完整三维供给，不能把仅边界表补成假内部值。examples/ai_round1/camx/boundary两组不同投影/格数/层数/字段单位/索引顺序均合成，独立SciPy检查全数组、侧边和时间区间；77项相关回归通过。无真实独立A/B2，标证据不足；未运行CAMx，气象、排放、机制控制仍需。

## SWAT+62 固定日历施肥/粪肥日程

mode=management_schedule，target_model_version=62.0.0，profile=calendar_fert_manure，automatic_operations=[]明确禁用自动规则。CSV映射schedule/order/operation/month/day/hu_sch/material/application/amount/amount_unit；操作映射仅fert/manu，hu_sch需源显式0（无量纲）。用户提供完整真实材料和施用数据库，粪肥org_min须有manure_om，其他粪肥深层项非null当前拒绝。用量转kg/ha，顺序每表1..N，日历递增，无跨年轮作/闰日推断；日期为事件表，不要求每日事件，更不补造不存在的措施。

官方62：[mgt_read_mgtops](https://github.com/swat-model/swatplus/blob/62.0.0/src/mgt_read_mgtops.f90)、[read_mgtops](https://github.com/swat-model/swatplus/blob/62.0.0/src/read_mgtops.f90)七操作字段；[mgt_sched](https://github.com/swat-model/swatplus/blob/62.0.0/src/mgt_sched.f90) fert/manu用量kg/ha。输出management.sch及实际材料/施用库，不生成file.cio；可核对已有landuse_file的管理名称，不擅自设置HRU管理。官方仓库LGPL2.1，writer为接口独立实现，官方小数据库保留来源，旧Editor头未冒称新工程。

examples/ai_round1/swat_plus/management两组合成日程，不同列名/顺序/日程数/kg-ha、g-m2、t-ha单位；独立Decimal读回用量与七字段、操作计数、引用库原字节一致。针对性+土地利用+CLI/界面63项通过。没有独立真实施肥计划A/B2，证据不足；未实现自动规则、作物播种收获、灌溉/轮作等，不是完整管理资料转换。

## CALPUFF5.8.4 常量点源group13

mode=constant_points，version=CALPUFF-5.8.4-group13，emission_time_policy=constant_supplied。ZIP含点源CSV及实际已有控制，十三字段明确映射：name/x/y/height/elevation/diameter/exit_velocity/exit_temperature/downwash/sigma_y/sigma_z/momentum_factor/platform_height；每个控制CSPEC物种都需name/column/unit，包括不排放物种的源显式0。检查控制物种标志、原group13完整子组计数、UTM/基准/网格范围及常量配置，拒绝已有变排放/缩放。烟囱高度/直径/温度需正，风速/sigma/质量率非负，无下洗/平台profile需源显式0；momentum_factor为0或1。坐标转km、高度m、速度m/s、温度K、排放g/s，IPTU=1，不解释吨/年或摩尔排放。坐标float32米制允许误差需事先声明，全部float32量化误差记报告。

依据：[官方5.8.4文件](https://www.calpuff.org/calpuff/download/EPA_Files/CALPUFF_v5.8.4.zip) CODE/CPUF.INP 13a–13d、calpuff.for READCF和params.puf（200点源、20物种默认上限）。X八参数后接控制物种顺序质量率；实数含小数点，包括下洗0.0。原控制NPT1之前和旧点源最后END之后保持，包括后续说明和控制组。不生成完整控制配置，不检查其它外部引用，不运行模型。

examples/ai_round1/calpuff/points两组均明确合成，部分数值取官方演示常数作格式参考；不是独立工程A/B2。不同基准、源字段顺序、点数、km/m、K/degC、g/s/kg-hour；独立delimiter解析每个X和物种顺序，手算质量率/单位及控制保留。相关52项通过；真实排放验证不足。其它排放、下洗和变排放仍未实现。
