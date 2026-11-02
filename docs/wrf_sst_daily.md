# WPS 4.6 逐日海温补充场

专项入口和CLI接受单快照、连续整点逐小时及连续整点逐日序列。逐日输出每个源UTC时刻一个SST_YYYY-MM-DD_HH.wps，另附README_daily_SST.txt及conversion_report.json。不填缺日、不插值、不复制成逐小时值；原有单快照固定场ZIP及逐小时二进制输出保持不变。

格式：WPS intermediate v5，IPROJ=0规则经纬度，SST/K/200100层，float32；文件头保留完整源时间。在WPS主机（通常Linux/WSL）重命名为SST:YYYY-MM-DD_HH，追加SST到实际namelist.wps的fg_name，不能把逐日序列作为constants_name。

metgrid时刻须与文件匹配。对本样例，start_date=2020-01-01_12:00:00，end_date=2020-01-03_12:00:00，interval_seconds=86400可覆盖所给3个时刻；这不是完整namelist。完整气象输入也必须覆盖这些时刻。若工程按小时或3/6小时处理，日海温不能单独覆盖中间时刻，需要用户另行选择有科学依据的时间处理方法或提供适合的高频海温；本转换器没有实施这种处理。

官方依据：WPS v4.6.0 [process_domain_module.F](https://github.com/wrf-model/WPS/blob/v4.6.0/metgrid/src/process_domain_module.F)的时间循环按interval_seconds推进（第141行），[read_met_module.F](https://github.com/wrf-model/WPS/blob/v4.6.0/metgrid/src/read_met_module.F)按fg_source:datestr构造文件名（第37行），并读取v5 HDATE头。没有“格式必须逐小时”的约束；这不等于任意运行配置都能使用日海温。

样例：examples/ai_round1/wrf/daily/A_sst_three_days.nc，配置config.json。NOAA OISST v2.1，2020年1月1–3日，每日12UTC；4×4海洋格点，纬度30.125–30.875N，经度300.125–300.875E，原单位degC。源为逐日分析产品，不能描述为瞬时海面观测。仅裁剪、拼接原始文件，没有插值或填补；source.json记录来源及复现步骤。

CLI：python -m ecodata.round1 --model wrf --input examples/ai_round1/wrf/daily/A_sst_three_days.nc --config examples/ai_round1/wrf/daily/config.json --output <新的输出目录>。

界面：首轮专项转换→WRF→上传A_sst_three_days.nc与config.json→执行→下载全部文件。报告的source_interval_seconds应为86400，time_processing说明无时间插值。

仍需要geo_em、适用METGRID.TBL、完整大气驱动、匹配域及有效海温覆盖（含插值邻域），并核对多个fg_name源的SST优先级。此4×4子域只用于格式/数值测试。独立解析通过不代表metgrid已实跑，更不代表WRF可运行；本轮没有运行metgrid、real或WRF。旧便携版ZIP未更新。
