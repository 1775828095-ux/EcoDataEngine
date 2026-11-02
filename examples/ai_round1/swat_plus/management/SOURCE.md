# 合成固定日历管理数据
两组字段名、记录顺序、日程数、单位均不同；日程/用量为合成，官方数据库只作材料与施用方式格式引用。
kg/ha、g/m²、t/ha的解析换算；无估计措施。不存在独立真实管理A/B2。
官方读取：https://github.com/swat-model/swatplus/blob/62.0.0/src/mgt_read_mgtops.f90
字段：https://github.com/swat-model/swatplus/blob/62.0.0/src/read_mgtops.f90
用量依据：https://github.com/swat-model/swatplus/blob/62.0.0/src/mgt_sched.f90 fert/manu case，kg/ha。
数据库：https://github.com/swat-model/swatplus/tree/62.0.0/refdata/Osu_1hru
样例数据库头注明旧Editor2.2/rev60.5.4，只按62实际源码核对字段/引用，不冒称62完整工程。
重建脚本：make_management_examples.py；源链接/哈希见evidence/swat_management/sources.json。
