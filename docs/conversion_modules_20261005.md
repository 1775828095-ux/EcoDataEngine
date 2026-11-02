# 本轮专用模块：输入前提、输出和复测

平台不部署或执行模型仿真。以下格式writer复用既有Pint单位引擎、NetCDF读取器、SST二进制记录封装和CLI/Streamlit入口。规则针对明确格式/变量，不针对样例地名或文件名写死。任意物理输入或工程拓扑仍须由用户提供。

## 通用入口

源码根目录：`C:/Users/ASUS/Documents/Codex/2026-10-02/wo/work/EcoDataEngine`，同步镜像`EcoDataEngine_Source`。

在根目录用配套Python启动：`_runtime/python.exe -m streamlit run src/ecodata/interface/app.py`。在“首轮专项转换”选择模型及下面列明的类型，分别上传**A源数据**和**config JSON**；缺项先补真实资料，不能填零冒充。执行后下载对应文件与转换报告。新入口已做开发AppTest连通，尚未由用户最终复测。

CLI统一为：`_runtime/python.exe -B -X utf8 -m ecodata.round1 --model MODEL --input A --config CONFIG --output DIRECTORY`。需`PYTHONPATH=源码根目录/src`；本轮所有实际命令在交付输出`actual_commands.ps1`，测试运行日志与XML另存。可选独立校验依赖：`PseudoNetCDF==3.4.1`、`geographiclib==2.1`；转换本身不依赖它们。

## CALPUFF流程 / CALMET5.8.5：高空探测

- 入口：CALPUFF→高空探测；CLI model=calpuff；mode=upper_air，version=CALMET-5.8.5-UP-2.0。
- A：CSV长表，一行一个探测层；可改列名/列顺序，配置station/time/level列和5个物理列。每站配置ID、1–4字符名称、经纬度和海拔。变量需要压力、MSL高度、温度、气象风向（真北起算来向）和风速；全部单位须明确。
- B：UP1.DAT、UP2.DAT…（UP.DAT2.0逗号数据格式），单位hPa、m、K、degree、m/s；层数按每次探测保存，2–99层。CSV可以多站、多次、不同层数；层序必须真实单调。源时刻需带UTC偏移，归UTC整点；给cadence就检查连续性，不填探测、不删坏层。
- NONE头不声称坐标基准，站点位置须在外部CALMET.INP正确配置；LL头需要明确datum。5.8.5原生代码将高度>=9998m视为缺测，因此本writer拒绝它，**不宣称覆盖任意模型层顶**。还需在CALMET.INP按报告配置IBYR、NUSTA、IDUSTA、UPDAT；探测时间必须覆盖目标处理期。工程数值检查范围在报告注明为暂定标准。
- 官方依据：[CALPUFF下载与许可](https://www.src.com/calpuff/download/download.htm)，本地既有官方CALMET5.8.5源的RDHDU、RDUP、QAYR4、PARAMS.MET；没有复制受许可限制的CALMET代码或二进制到平台。
- 真实A：`examples/ai_round1/calpuff/upper_air/A_NOAA_IGRA_72357_20260101.csv`，NOAA NCEI IGRA v2.2 USM00072357站，2026-01-01名义00/12UTC，各38层。精确保留压力>=500hPa的原始层，无缺测过滤、补值或插值；时间是名义探测小时，不是释放分钟。`raw_IGRA_20260101.txt`与source_manifest保存来源、散列、单位和完整性结果。[数据](https://www.ncei.noaa.gov/pub/data/igra/data/data-y2d/USM00072357-data-beg2026.txt.zip)、[IGRA格式](https://www.ncei.noaa.gov/pub/data/igra/igra2-readme.txt)、[NOAA使用政策](https://www.noaa.gov/information-technology/open-data-dissemination)。美国政府公开数据，保留来源且不暗示NOAA背书。
- 第二A：A_synthetic_two_station_UTF8.csv/config_synthetic_two_station.json，中文列名、多站、不同层数、hPa/km/km/h。明确合成，不冒充另一真实观测站。
- 验证：独立固定列/逐层解析、原始NOAA字段比例读回，检查头、时间、站点、层序及各变量；本案例无独立同源UP.DAT B2。只证明转换范围内格式/数值，不证明CALMET已读入。

## CALMET5.8.5：独立小时降水

- 入口CALPUFF→独立小时降水；mode=precipitation、version=CALMET-5.8.5-PRECIP-2.0。A为CSV，fields映射station/time/precipitation，station_ids明确输出顺序，时刻带UTC偏移，time_label=hour_end、timezone_west_hours明确固定标准时区。
- quantity=one_hour_amount接受完整前一小时累计深度，必须accumulation_seconds=3600、源单位为长度；或者hour_mean_rate接受已知小时平均率与长度/时间单位。前一小时累计mm转该小时平均mm/hour不改变结束时刻，不能拆分日累计、把瞬时雨量当小时率。支持源列名和顺序变化、多站，但缺小时/缺站/负值/NaN/缺测/布尔/单精度下溢均拒绝，显式0必须已有。
- B为PRECIP.DAT2.0 formatted NONE，每行year/Julian-day/hour/各站mm/hour，头包含起止时间、固定时区、站点数与ID顺序。暂定工程检查0–1000mm/hour，原生MXPS=700；不生成坐标基准或控制文件，已有CALMET.INP须正确配置IFORMP=2、NPSTA、IDPSTA顺序、IBTZ及站点位置。依据同一官方5.8.5 RDHD/RDP，不复制其程序。
- 真实A：`calpuff/precipitation/A_NOAA_USCRN_20210701.csv/config_NOAA.json`；NOAA USCRN Boulder14W，WBAN94075，2021-07-01 01UTC至07-02 00UTC完整24小时，P_CALC毫米总量26.3，最大12.7。原始精确记录raw_USCRN_20210701.txt与source_manifest提供无缺测证明。[原始站点文本](https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/2021/CRNH0203-2021-CO_Boulder_14_W.txt)、[官方字段说明](https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/README.txt)规定字段13为前60分钟累计mm、UTC结束时刻、P_CALC无单独QC标志。US政府公开数据，注明NOAA来源。错误下划线文件名404证据未用于A，改用官方目录真实文件名。
- 第二A：A_synthetic_two_station.csv/config_synthetic.json，中文列名、乱序、站点顺序[92,91]、mm/s→mm/hour、东八区跨闰日，独立读回为2024年Julian60日23时[3,0]和61日0时[1,2]。没有同案例B2，按官方读取结构独立解析并逐行与原始NOAA数据对比，不声称CALMET已读取。

## SWAT+62.0.0：HRU属性和已处理地形参数

- 入口：SWAT+→HRU属性与地形参数；mode=hru_attributes，target_model_version=62.0.0。
- A：ZIP内HRU源CSV +所需真实数据库；JSON设置table、10项fields映射、references按角色指向ZIP实际文件；可选topography CSV参数表配置。HRU必须是上游已划分对象，不从DEM、地类、土壤栅格自动推断。
- B：hru-data.hru，字段id/name/topo/hydro/soil/lu_mgt/soil_plant_init/surf_stor/snow/field；可选topography.hyd，字段name/slp/slp_len/lat_len/dist_cha/depos。坡度和沉积系数无量纲，长度m，支持明确%和km转换。沉积系数不是必须<=1的概率。
- HRU ID须已有且完整1..N，不重编号，不构造身份。必须引用真实topo/hydro/soil/landuse/soil_plant；snow/field允许显式null；non-null surf_stor暂拒绝。参考数据库只核对名称、最低记录结构及相关土层行数，**不认证数据库全部参数、内部引用或完整工程**。数据库作为用户已有工程输入，模块只生成hru-data.hru和可选topography.hyd，原数据库不重新生成、不作为转换成果。用户须保留ZIP内实际引用数据库。无时间轴/坐标字段；位置/面积/连接另在上游HRU工程中提供。
- 无file.cio、hru.con或虚构WGN/河道/含水层。ZIP禁止路径穿越、重复文件名、超大解压规模。
- 官方依据：[hru_read.f90](https://github.com/swat-model/swatplus/blob/62.0.0/src/hru_read.f90)、[HRU属性类型](https://github.com/swat-model/swatplus/blob/62.0.0/src/hru_data_module.f90)、[地形读取](https://github.com/swat-model/swatplus/blob/62.0.0/src/topohru_read.f90)；62.0.0仓库LGPL2.1；本轮自行实现writer，官方样例作为注明来源的格式参考。
- 样例1：`swat_plus/hru/A_official_HRU_bundle.zip/config_official.json`，官方[Ames_sub1](https://github.com/swat-model/swatplus/tree/62.0.0/refdata/Ames_sub1)12-HRU数据库独立解码为CSV。B2_official_hru-data.hru逐字段一致。**它是已处理参考输入重编码，不是从原始地理数据生成流域的证明**。引用数据库数值未全部认证，不能直接用于完整工程验收。
- 样例2：A_synthetic_custom_bundle.zip/config_synthetic.json，3个乱序HRU、自定义列名和两个地形对象，%/km独立手算Decimal读回；官方数据库仅引用夹具，不称真实项目参数。

## SWAT+62.0.0：植被非城市土地利用配置数据库

- 入口SWAT+→土地利用配置数据库；mode=landuse_database、target_model_version=62.0.0、profile=vegetated_nonurban。A为ZIP，14项字段映射的CSV和实际引用数据库；名称ASCII<=40字。B为landuse.lum（name/cal_group/plnt_com/mgt/cn2/cons_prac/urban/urb_ro/ov_mann/tile/sep/vfs/grww/bmp）及范围说明。
- 14项为名称/引用，无物理单位、时间、坐标或网格。需要实际plant.ini植物群落、management.sch管理操作、cntable.lum、cons_practice.lum、ovn_table.lum。核对引用名称、计数及子记录、CN0–100/USLE P0–1/糙率min<=mean<=max等基本检查；不生成被引用数据库，不认证全部植物参数、管理操作所引用的肥料/灌溉表。植物、CN、糙率必需；管理可显式null。urban/urb_ro及工程措施非null暂拒绝，不猜测城市或结构参数。
- 官方依据：[landuse_read.f90](https://github.com/swat-model/swatplus/blob/62.0.0/src/landuse_read.f90)、[landuse_data_module.f90](https://github.com/swat-model/swatplus/blob/62.0.0/src/landuse_data_module.f90)，LGPL2.1。
- `swat_plus/landuse/A_official_vegetated_bundle.zip/config_official.json`精确保留Ames12个HRU实际引用的两种地类cosy_lum和mntill_corn_lum，B2逐字段对比。官方tag62目录内原始文件头写Editor2.3.3/rev60.5.7；目标结构以实际62读取代码核对，不能由旧文件头推断完整工程兼容。
- 官方原表21项中20项非城市；其中4个未使用管理名称AGRR_rot/OATS_rot/SOYB_rot/WWHT_rot不在所给management.sch中。保存A_official_unresolved_negative_case.zip，测试要求明确拒绝；没有编造缺管理表或忽略坏引用。两项参考子集由已有HRU引用选择，不在writer中按地名或条目写死。
- 第二样例A_synthetic_custom_bundle.zip/config_synthetic.json，中文字段、3个合成地类名、反向条目、显式无管理；实际引用数据库仅参考夹具。独立拆分每行14字段读回。这只验证配置数据库映射/引用，**不是土地利用栅格处理、HRU划分、管理措施writer或完整流域工程**。

## WRF流程 / WPS4.6：等压、地面和指定土壤层场

- 入口：WRF→多变量气象场；mode=pressure_fields，version=WPS-4.6-intermediate-v5-fields。
- A：NetCDF，明确coordinates time/latitude/longitude/可选pressure，field→源变量映射；公历UTC整点，序列需interval_seconds且连续；规则一维经纬度、至少2×2网格。支持轴反向和源变量维序变化，但不重网格、不插值。配置单位不得覆盖冲突的源units。
- B：PREFIX_YYYY-MM-DD_HH.wps，WPS intermediate-v5大端Fortran记录，IPROJ0；WPS主机需重命名PREFIX:YYYY-MM-DD_HH并配置fg_name。地球半径6367.47km，和既有SST一致；不是任意投影支持。
- 等压字段TT(K)/UU、VV(m/s)/RH(%)/GHT(m)，压力Pa；只接受真实等压坐标，不sigma伪装。风成对、须明确earth_relative，输出is_wind_grid_rel=false，不做风旋转。地面PSFC/PMSL(Pa)、SKINTEMP/SST(K)、SOILHGT(m)、LANDSEA(0/1)、SNOW(kg/m²)、SNOWH(m)；PMSL用201300，其他地面200100层标。
- 土壤字段ST/SM按0–10、10–40、40–100、100–200、10–200cm输出对应标签；必须source_depth_cm吻合，源有层深属性也须一致。ST K；SM fraction且明确volumetric；kg/kg质量含水率拒绝。每层一个3D time/lat/lon变量，不改深度、不补缺层。
- 官方依据：[WPSv4.6.0 output.F](https://github.com/wrf-model/WPS/blob/v4.6.0/ungrib/src/output.F)、[读取代码](https://github.com/wrf-model/WPS/blob/v4.6.0/metgrid/src/read_met_module.F)、[METGRID.TBL.ARW](https://github.com/wrf-model/WPS/blob/v4.6.0/metgrid/METGRID.TBL.ARW)。本轮不分发WPS代码/二进制，仓库通用许可未确认，引用格式不等于可随意复制其代码。
- 真实等压A：`wrf/pressure/A_ncep_pressure.nc/config.json`，[NOAA PSL NCEP-NCAR](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html)，2024-01-01 00/06UTC，35–40N、105–110E，3×3、8个共同压力层、5变量720值均有限；只取所有变量共同实际层，不填RH缺少的高层。五个NCSS下载地址、散列与逐变量完整性记录保存于交付证据。
- 第二等压A：A_synthetic_mixed_fields.nc/config_synthetic.json，自定义坐标、转置维序、逆轴和混合TT/PSFC/LANDSEA。
- 真实土壤A：`wrf/soil/A_NOAA_topsoil_20240101.nc/config_NOAA.json`，[NOAA surface_gauss目录](https://psl.noaa.gov/thredds/catalog/Datasets/ncep.reanalysis/surface_gauss/catalog.html)，soilw.0-10cm.gauss.2024.nc和tmp.0-10cm.gauss.2024.nc，2时刻×2纬度×3经度，两变量24值。Gaussian全场不是等间距；此处只精确截取两个相邻纬度（40.9517/39.047）及3经度，无插值，不声称一般Gaussian网格转换。原来tmp请求误用tsoil的400响应已留证据，改用官方实际变量tmp后200；错误下载文件未用作A。
- 第二土壤A：A_synthetic_four_layers.nc/config_synthetic.json，四层、自定义坐标、逆轴、degC/%、闰日。
- NOAA公开数据政策保留来源、不背书；第二类明确合成。独立SciPy FortranFile检查每个头记录、XLVL、单位、经纬度、UTC、每层每时刻数值；float32预设相对舍入1e-7，温度绝对3e-5K。无该子域匹配B2，不认证metgrid已读取，更不代表WRF输入链完成。

## CAMx7.40：机制分种的地面网格排放

- 入口：CAMx→网格排放；mode=gridded_emissions、version=CAMx-7.40、mechanism=CB6r5_CF3_SIMPLE_NoEvap。
- A：CSV长表完整time×row×col，fields映射和row_col_base；每个源物种配置目标标签、gas/aerosol与单位。必须已分种且逐小时，不把NOx/PM25总量自动拆分。完整LATLON或球体LCC网格配置，行南→北、列西→东；固定标准时区ITZON（UTC以西小时），不夏令时。
- B：emissions.CAMx7.40.nc，NetCDF3经典格式，2D地面排放；气体mol/hour、气溶胶g/hour，保留逐小时起止TFLAG/ETFLAG；坐标/投影元数据按CAMx格式，不是一般NetCDF改名。
- 依据：[CAMx官方指南和版本资料](https://www.camx.com/download/)、[官方辅助工具](https://www.camx.com/download/support-software/)，7.40官方ncf_areaprep/ncf_chk_griddef/CHEMPARAM与camx2ncf_ems1.1；本轮没有复制其Fortran代码到平台。CAMx许可须按官方条件处理，不宣称可随意再分发二进制。
- 两组A均明确合成：`camx/emissions/A_synthetic_LATLON_hourly.csv/config_LATLON.json`（2小时2×3，NO+PEC，ITZON=-8，跨年）；A_synthetic_LCC_single_cell.csv/config_LCC.json（1小时单格，NO、自定义字段、base0、LCC）。独立SciPy NetCDF解析坐标/标志/每个值。**尚无合格真实小时分种清单，真实数据验证不足**；无B2，不认证CAMx已读取。
- 缺项/重复格/缺时刻/负排放/未知物种/质量摩尔含混/float32溢出下溢拒绝。无排放清单空间分配、时间分配、高架/点源输入或其他机制。

## CMAQ5.5：地面网格排放及对应GRIDDESC

- 入口：CMAQ→网格排放；mode=gridded_emissions、version=CMAQ-5.5、mechanism=cb6r3_ae7_aq。目录JSON只抽取官方物种标识作校验，不复制机制实现。
- A：已分种、已空间分配的NetCDF，axes、源单位、物种gas/aerosol_mass和cell_rate/surface_flux明确；浓度不能当排放。质量→摩尔需要明确正摩尔质量及其依据；面积通量→格点量需要真实格面积或明确WGS84测地面积规则。不推算空间/化学分配。
- 地面单层且vertical明示VGTYP7、VGTOP及两层界；只验证格式/范围，**实际项目必须与MCIP相同，样例垂直定义是演示配置而非实测气象场**。GDTYP1经纬度/2Lambert；格心与原点格距行列数相符，反轴与数据同步；Lambert需明确同一source_crs和球体半径，当前仅XCENT=P_GAM。不重采样、不造高架注入。
- hourly需真实连续UTC小时；static_inventory必须说明原始平均期与静态使用假设，无时间轴，I/OAPI SDATE/STIME/TSTEP/TFLAG=0；**不生成虚假小时序列**。
- B：emissions.CMAQ55.nc，I/OAPI GRDDED3，气体moles/s、颗粒质量g/s；GRIDDESC仅描述同一个已校验水平网格，与NC元数据匹配，非MCIP生成器。必须用CCTM/DESID映射匹配物种，不能据导出就认为CCTM接受了所有清单。
- 官方依据：[CMAQv5.5_2Oct2024输入说明](https://github.com/USEPA/CMAQ/blob/CMAQv5.5_2Oct2024/DOCS/Users_Guide/CMAQ_UG_ch04_model_inputs.md)、[GC物种文件](https://github.com/USEPA/CMAQ/blob/CMAQv5.5_2Oct2024/CCTM/src/MECHS/cb6r3_ae7_aq/GC_cb6r3_ae7_aq.nml)、[CMAS GRIDDESC](https://www.cmascenter.org/ioapi/documentation/all_versions/html/GRIDDESC.html)。EPA [license.md](https://github.com/USEPA/CMAQ/blob/CMAQv5.5_2Oct2024/license.md)为仓库复用条件，保留来源及其许可证证据；本轮主要是自有writer。
- 真实A：`cmaq/emissions/A_EDGAR_CO_2014_Paris.nc/config_EDGAR.json`，欧盟JRC [EDGARv8.1](https://edgar.jrc.ec.europa.eu/dataset_ap81)2014年CO TOTALS年度平均kg m-2 s-1，巴黎3×3（48.75–48.95N，2.25–2.45E），9个有效正值；许可/使用条件为公开数据并须来源署名，不擅自标成CC-BY。用HTTP Range只获取ZIP目录和2014最小成员（约15.8MB压缩），未下载843MB整包；精确子域，不插值填值。28.01g/mol是显式CO摩尔质量，验证转换而非清单排放真实性。
- 第二A：A_synthetic_LCC_two_hours.nc/config_synthetic.json，LCC逆y轴、跨年两小时、NO2kg/hour与ASO4Jg/hour、自定义坐标；明确合成。
- 独立SciPy NetCDF3解析和PseudoNetCDF读取，WGS84面积由**不同实现GeographicLib**独立计算后对比pyproj结果；预定rtol1e-7不放宽。第三方GRIDDESC读回两种网格与NetCDF参数一致。无同案例原生排放B2；验证数值和结构，不验证清单真实性、DESID工程匹配或CMAQ仿真。

## 验证工具及许可边界

[PseudoNetCDF3.4.1](https://github.com/barronh/pseudonetcdf)是可选独立读取工具，LGPL3许可见其发布包LICENSE，不嵌入平台生产writer；[GeographicLib2.1](https://geographiclib.sourceforge.io/Python/) MIT。SciPy读取器与netCDF4写入器独立；使用相同物理A的独立解析/手算不是匹配案例B2，不能冒称原生参考输入。非零float32下溢直接拒绝，不静默变0。

新增模块后需继续逐类开发的真实缺口见覆盖清单。平台两个源码目录同步一致；现有Windows便携ZIP未重新打包，本轮用户最终界面验收未执行。
