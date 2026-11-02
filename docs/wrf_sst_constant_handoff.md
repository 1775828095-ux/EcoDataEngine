# WRF：WPS固定海温快照交接

目标接口是WPS4.6.0 metgrid，intermediate v5/IPROJ0/SST/K/200100层。不是wrfinput，不宣称某个WRF模拟版本已验证。原SST_YYYY-MM-DD_HH.wps仍保留；默认行为和既有小时连续性规则不变。

本轮新增可选JSON：`"wps_handoff":{"mode":"constant_snapshot"}`。必须只有一个源时刻，不自动平均或选帧。它表达将这一幅原始海温分析作为固定补充场，适用性须按实际模拟窗口决定；不生成小时海温，不更改原时间标签。

上传examples/ai_round1/wrf/A_sst_subset.nc与config_constant_snapshot.json，在“首轮专项转换”选择WRF。CLI也使用同一转换器。输出WPS_metgrid_SST_constant.zip含ECODATA_SST_CONSTANT、namelist.metgrid.fragment、manifest.json及README.txt。

在WPS工作目录解压包，把fragment的constants_name赋值合入已有&metgrid组；保留实际fg_name、其它constants及域配置。不要用片段替换完整namelist.wps。WPS读取固定文件的原名，不需Windows无法创建的冒号文件名。manifest记录原时间、网格范围、单位、SHA与未实跑标记。

已有NOAA OISST2.1样例为2020-01-01 12UTC、4×4海洋格点，仅足以测试格式/值。实际域所有水面及插值邻域必须被有效SST覆盖；不要让该小片段覆盖大域。若fg_name也含SST，需要在真实工程核对优先级及met_em中的SST/FLAG_SST。

官方依据：[WPS用户指南](https://www2.mmm.ucar.edu/wrf/site/documentation/users_guide/wps.html)的constants_name；[WPS4.6 process_domain_module.F](https://github.com/wrf-model/WPS/blob/v4.6.0/metgrid/src/process_domain_module.F)直接读取constants_name，并注明SST/SEAICE常量场。

仍缺完整三维气象/土壤驱动、geo_em地理域、完整namelist及对应WPS/WRF程序。本轮独立Fortran记录解析只能证明该包的文件结构/数据与配置一致，不能证明metgrid、real或WRF已经运行。
