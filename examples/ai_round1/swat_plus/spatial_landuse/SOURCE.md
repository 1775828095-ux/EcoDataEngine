# 合成空间测试，非独立真实流域案例
两组100×50 m解析矩形；不同CRS、属性字段、HRU数、MultiPolygon/Polygon和反序结构。无插值、缺项填补。
引用数据库及属性来自官方SWAT+62仓库的既有输入，只作格式引用。
https://github.com/swat-model/swatplus/tree/62.0.0/refdata/Osu_1hru
官方读取代码：https://github.com/swat-model/swatplus/blob/62.0.0/src/hru_read.f90
许可沿用既有来源记录；新增几何为工程合成测试，不声称真实土地利用验证。
原始A是ZIP；config是JSON；format_reference是手选既有属性行，不是平台输出，不是同案例真实GIS的B2。
重新生成：python coverage_completion_20261006/make_spatial_examples.py
