# SWAT+ 62.0.0 weather-wgn.cli

本功能转换已经具备的站点信息和12个月统计参数，不计算统计、不生成随机天气，不提供file.cio或其它流域工程。现有日天气NC功能不变。

依据：62.0.0 src/cli_wgnread.f90、climate_module.f90、cli_staread.f90、cli_initwgn.f90。第一行标题；每个站点一行`name latitude longitude elevation rain_yrs`，一行统计字段表头，随后1月至12月各14数值。原生文件没有month列；输入CSV必须明确month，转换后按月排序。WGN站点名是weather-sta.cli第二列的名称引用，不是另一个.wgn文件名。

|字段|含义|输出单位|
|---|---|---|
|tmpmx、tmpmn|月内日最高/最低温平均值|degC|
|tmpstdmx、tmpstdmn|日最高/最低温标准差|degC（温差）|
|pcpmm|平均月降水总量|mm|
|pcpstd|日降水统计标准差（按该版本字段约定）|mm/day|
|pcpskw|日降水偏度|1|
|pr_wd、pr_ww|干日/湿日后为湿日的条件概率|1|
|pcpd|平均月降水天数|day|
|rainhmx|该月最大0.5小时降水参数|mm|
|solarav|月平均日太阳辐射|MJ/m2/day|
|dewpt|原生双用途列：月平均露点，或明确声明的月平均相对湿度|degC，或湿度分数1|
|windav|月平均风速|m/s|

站点单位：经纬度degree、海拔m、rain_years为用于半小时最大降雨统计的记录年数year，不是整个气象序列的统一年数。经度要求±180，不猜测东/西经符号。站点名限制50字符ASCII，不截断。

输入CSV：station、month及全部14字段，字段名通过JSON fields映射。station必须与JSON stations[].id精确对应；units逐字段必填，station_units必填。来源source需url/institution/license/description和kind；climate_statistics还需跨至少两年的period_start_year/period_end_year及适当rain_years。此规则防止误用单年天气，不是转换器自动认证来源真实性。format_reference、synthetic_test会在报告中保留不同证据标签。missing_values需列出来源数字缺测标记，没有则明确[]；空值/NaN/无穷、缺月、重复月、缺列、未映射额外列等拒绝，不填零。

单位转换只支持已明确的有限路径：平均温度K或degF→degC；温度标准差K不改值、degF仅乘5/9，不加温度偏移；概率%→1；降水量m→mm；pcpstd m/day→mm/day；辐射kWh/m2/day→MJ/m2/day；风速km/h→m/s。W/m2不能直接当日能量；月平均日降水不能只改标签为月总量。

2026-10-21原生读取核查修复：官方62的cli_initwgn按站点判断：该列12个月全部位于[0,1]时按相对湿度分数读取；只要有一个月超出该范围，全部月份按露点°C读取。配置units.dewpt为`1`或`%`时明确选择湿度统计，输出分数；为`degC`、`K`、`degF`时保留露点语义，换算后若12个月全在[0,1]则拒绝，因为该原生格式不能无歧义表达。不能通过篡改一个月的值迫使模型切换。平台不计算湿度与露点之间的统计换算。报告humidity_semantics及逐字段输出单位保留实际含义。

依据：[cli_initwgn.f90](https://github.com/swat-model/swatplus/blob/62.0.0/src/cli_initwgn.f90)中idewpt判断和[cli_rhgen.f90](https://github.com/swat-model/swatplus/blob/62.0.0/src/cli_rhgen.f90)中的两条读取路径。

负温及负偏度允许；标准差/降水/湿日数/辐射/风速非负，概率0–1，最高温≥最低温；湿日数不能超过该月可能最多天数，二月允许闰年混合统计。模型cli_initwgn会调整部分偏度、湿日概率和湿日数；本程序保留原参数，报告提醒该行为，不实施模型初始化算法。

CLI：`python -m ecodata.round1 --model swat_plus --input examples/ai_round1/swat_plus/wgn/A_official_decoded_monthly.csv --config examples/ai_round1/swat_plus/wgn/config.json --output <新目录>`。可另加`--weather-station-index <已有weather-sta.cli>`，只检查WGN名称，不修改或打包该文件。无索引时明确“站点关联未验证”；有索引而WGN为null、拼写不同或缺少站点时拒绝，提示需在真实目标工程明确关联。本轮没有校验其它天气文件或空间代表性。

界面：首轮专项转换→SWAT+→WGN月统计→上传CSV→上传JSON→可选上传已有weather-sta.cli→执行→下载weather-wgn.cli或全部文件。输出为weather-wgn.cli、README_WGN.txt和conversion_report.json。选WGN却上传日天气配置会给出具体错误。

样例分层：A_official_decoded_monthly.csv由官方62.0.0/refdata/Ames_sub1/weather-wgn.cli独立按固定位置解码，B2_official_weather-wgn.cli保留原样。站点wgn001，37.48N、100.83W、887m、rain_years=10，12个月168值。来源只有参考格式/参数，不是本平台从长期原始气象算出的气候统计，不虚构统计时段。保留仓库LGPL2.1许可及来源链接；不宣称已核实全部第三方统计数据产权。

本轮独立真实统计数据验证不足：NOAA Normals和WorldClim没有已核对齐全的14项参数；Robit已有12月统计且有限，但第三方许可及部分单位未确认，不计为合格独立导入案例、不重新分发。官方样例重编码和人工合成回归不能替代独立真实数据验证。

参考对照阈值预先设为逐数值绝对误差≤1e-12（本轮文本编码工程标准）；单位换算测试同阈值。不运行SWAT+模型，不声称完整工程可运行。仍缺土壤、HRU、河道、含水层、土地利用、管理、连接和完整控制文件。现有Windows便携版ZIP未更新；使用最新源码运行。
