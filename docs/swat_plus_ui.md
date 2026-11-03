# SWAT+ 单站逐日天气专项转换

入口：首轮专项转换 → SWAT+：单站逐日天气（降水、温度、辐射、湿度、风速）。CLI 和界面共用 `export_uploaded_weather` 与 `SWATPlusWriter`。

旧的 `daymet_2021_complete.nc + config_ui.json` 仍只导出降水/温度，天气数值、表头、日期及实际天气索引保持一致；专项入口不再交付 file.cio，报告与清单相应更新。不会因 NC 中存在 solar_radiation 就静默改变旧配置的输出。

新增样例：`examples/ai_round1/swat_plus/power_2021/A_power_2021.nc` 是用户导入的原始天气 A（从公开 NASA POWER JSON 无损组织为 NC），同目录 `config_ui.json` 是本次配置；`NASA_POWER_2021_original.json` 保留上游原始响应，`source.json` 记录来源与准备方法。转换后的 station.* 是输出 B，不能再次当原始 NC 导入。

配置新增 `target_model_version: "62.0.0"` 与 `additional_weather: ["solar_radiation", "relative_humidity", "wind_speed"]`。可只选择其中一部分；未选择的字段明确列为未导出。基础字段仍为 precipitation、air_temperature_max/min，time 均需一维单站逐日。

输入单位：precipitation=mm/day，温度=degC；solar_radiation=MJ/m2/day（亦接受 MJ/m^2/day）或 kWh/m2/day（乘3.6）；relative_humidity=%（乘0.01）或1；wind_speed=m/s或km/h（除3.6）。不能仅换单位标签。W/m2缺少积分时段时拒绝；Daymet 白昼平均 srad 不可直接乘86400。风速高度不做隐式换算，源属性 measurement_height_m 写入报告。

要求：日期从start到end连续、有序、无重复；不能有NaN、-99或源填充值；所选变量缺列即报错，不补日期/数值。辐射和风速非负、湿度转换后0–1；最高温不得低于最低温。站点经纬度/海拔有限，与源属性（如果存在）一致，站点名不能含空白。

输出新增 station.slr、station.hmd、station.wnd、slr.cli、hmd.cli、wnd.cli；更新 weather-sta.cli 的实际天气引用，并在下载前检查所有交付 .cli 的文件引用。单变量文件三行头记录为标题、NBYR/TSTEP/LAT/LONG/ELEV、对应数值；日记录为年/年内日序/值。单位分别MJ/m²/日、0–1、m/s。pcp/tem及其索引保留。

完整六变量样例共下载16文件：5个站点天气文件、5个变量索引、weather-sta.cli、time.sim、README_weather_only.txt、manifest.json、validation_report.json、conversion_report.json。

NASA POWER样例为2021年365日；POWER Daily API v2.10.0，MERRA2/SYN1DEG/POWER，LST日标签；API geometry站点坐标39.679N、77.234W，海拔191m为产品返回值，不是测站测量。它是格点分析/卫星产品，不是地面实测。

版本边界：新增文本格式按SWAT+62.0.0的cli_smeas/hmeas/wmeas核对，Editor4.0.2支持rev62；本轮没有运行官方模型/Editor，不能称完整原生工程被接受。未生成weather-wgn.cli，没有12个月WGN统计及关联，也未生成HRU、河道、含水层。天气包不交付 file.cio。time.sim 仅保留日期控制片段，不是完整工程；weather-sta.cli 中 null 与 sim 是保留选项，sim 需要在目标工程中另配 WGN。存在性校验不能证明官方模型可运行。

CLI：`python -m ecodata.round1 --model swat_plus --input examples/ai_round1/swat_plus/power_2021/A_power_2021.nc --config examples/ai_round1/swat_plus/power_2021/config_ui.json --output <新的输出目录>`。

界面：上传A_power_2021.nc → 上传同目录config_ui.json → 检查配置 → 执行专项转换 → 下载全部转换文件。旧便携ZIP未更新，不能用来复测本轮新增变量。

2026-10-05 最小修复：仅改变专项 CLI/界面天气下载包的交付范围，保留通用 writer 和模板模式行为。输出请使用新目录，避免旧目录中残留先前的 file.cio。现有 Windows 便携版 ZIP 未更新，不包含本轮修复。
