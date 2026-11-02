# 合成WPS输入包，非真实独立数据
原始A本身是已有格式场的合成输入，不冒充原始观测。两组不同投影、网格形状、记录分组/顺序。
格式依据：https://github.com/wrf-model/WPS/blob/v4.6.0/metgrid/src/read_met_module.F
生成方法：make_wps_package_examples.py按官方五记录布局独立构造，温度=280+时间索引*10+字段层索引+格点索引 K。
本功能组包并保持原生场字节，不运行metgrid、不重网格、不生成气象值。合成案例不能证明真实工程数据全部正确。
