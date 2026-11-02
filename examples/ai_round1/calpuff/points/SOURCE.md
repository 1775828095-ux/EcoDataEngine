# CALPUFF5.8.4合成常量点源
官方控制格式来源：https://www.calpuff.org/calpuff/download/EPA_Files/CALPUFF_v5.8.4.zip CODE/CPUF.INP和calpuff.for READCF group13。
两组CSV与已有控制ZIP：不同列名/顺序、NAD27/WGS84、km/m、K/degC、g/s/kg-hour、1/2点源。合成字段均显式供给，包括初始sigma和二次物种0；不是填补缺项。WGS84控制基准为合成变体，不能认为配套原气象已匹配。
有些数值取官方演示常数作手算格式参考，不冒充真实工程原始观测A；没有独立真实A/B2。
输出CALPUFF.INP为已有控制group13的常量点源替换，其他组保持；不运行CALPUFF。
