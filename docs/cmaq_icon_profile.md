# CMAQ5.5 ICON/profile专项输入

进入首轮专项转换，选择CMAQ 5.5 ICON/profile；导入examples/ai_round1/cmaq/icon_profile/A_profile_long.csv及config.json，执行，下载IC_PROFILE_CMAQ55.csv和报告。

CLI：`python -m ecodata.round1 --model cmaq --input examples/ai_round1/cmaq/icon_profile/A_profile_long.csv --config examples/ai_round1/cmaq/icon_profile/config.json --output <输出目录>`。使用当前平台_runtime/python.exe；PYTHONPATH为src及当前目录。

这是ICON/profile的IC_PROFILE文件，不是CCTM I/O API初始浓度。现有PM2.5辅助场景不变，不能从PM25总量推测完整物种或垂直分配。

输入长表variable/unit/layer/value，44层。输出注释行、VNAME/UNITS/LAYER01..44表头和逐变量剖面；字符串加引号以保护含空格的单位。保留PRES/Pa、ZH和ZF/m及所有提供的物种，不猜测单位，不填零或截断负值。允许ppbV→ppmV、hPa→Pa、km→m；其它只有相同单位直接保留。严格检查记录完整、变量与配置一致、压力递减、层中心与层顶高度有序。物种/单位名称长度符合16字符约定。本轮范围限官方44层CB6R3M_AE7_KMTBR样例，禁止把旧CB05/CMAQ5.0.2模板与此格式混用。

真实样例来自EPA CMAQ5.5仓库；其剖面由CMAQ5.3 beta2模拟产生，为2016年37N/157W附近远洋年度均值，原作者/模拟版本不改写。A是独立解码长表，参考原CSV用于核对。它不是新观测，也不是任何日期的瞬时三维初始场；位置是样例来源，不是输出网格。

官方依据：https://github.com/USEPA/CMAQ/blob/5.5/PREP/icon/README.md 及该版本prof_icout.F（注释以#开头、按逗号计层、Fortran列表读变量/单位/各层值），prof_vinterp.F（ZH垂直插值）。设置ICTYPE=profile、IC_PROFILE=<输出CSV>，还需要GRIDDESC/GRID_NAME、MET_CRO_3D_FIN和已构建的ICON。ICON输出INIT_CONC_1才是下一步CCTM候选。目标机制、物种映射以及未提供物种须在ICON/CCTM工程核对；本轮没有合成或运行它们。

独立解析和源数据一致不证明真实ICON读取或CMAQ运行。本机未发现适用ICON程序/Fortran编译器。旧Windows便携ZIP未重打包，不含本修复。
