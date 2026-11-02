# SWAT+ 62.0.0 分层土壤转换

输入CSV、单数据表Excel或Parquet。配置mode=soil，source_format必须匹配文件。每行是一种土壤的一层；fields将任意源列映射到示例配置中的21个必需语义字段。土壤级字段逐层重复且必须一致。每种土壤层号唯一连续1..N，支持多土壤、不同层数，不硬编码地点或土壤编号。

输出soils.sol、README_soils_only.txt、conversion_report.json。不生成file.cio、HRU、WGN或完整流域工程。上传缺必需参数的土壤检测表不会自动推算参数。空间位置与土壤名称/HRU之间的对应关系由工程提供；本格式不存时间或地理网格。目标hru-data.hru使用的土壤名称必须存在，转换报告明确标记该关联尚未上传验证。

## 单位与验证

zmx：最大根系深度mm；depth：各层底累计深度mm（depth_basis=bottom），不是层厚。bd：Mg/m³；awc：mm/mm；k：mm/h；cbn、clay、silt、sand、rock、cal：百分数；alb、anion_excl、crk：0..1；ec：dS/m；ph：pH；usle_k：已采用目标SWAT约定的USLE_K。不同USLE单位体系不自动猜测或换算。

复用现有Pint单位引擎，支持同维度深度、密度、导水率、电导率换算；明确区分百分数与0..1。拒绝未知单位、缺列、重复映射、NaN、无穷、缺测标记、重复层号、层深倒置、元数据冲突、无效土壤编号、水文双组未明确排水状态。黏粒+粉粒+砂粒合计100%的0.1百分点容差是预设工程一致性检查，不代表所有土壤测量精度。超出容差拒绝，不归一化。

62实际代码的粒级输入是百分数，依据soil_phys_init.f90中除以100的运算；soil_data_module部分注释写fraction不能单独作为单位依据。模型本身可能调整首层深度、alb、anion_excl；转换器保留原值并在相关情况下警告。

## 来源和证据边界

- 官方读取器：https://github.com/swat-model/swatplus/blob/62.0.0/src/soil_db_read.f90
- 字段与单位：https://github.com/swat-model/swatplus/blob/62.0.0/src/soil_data_module.f90
- 单位实际运算：https://github.com/swat-model/swatplus/blob/62.0.0/src/soil_phys_init.f90
- 原始参考文件：https://github.com/swat-model/swatplus/blob/62.0.0/refdata/Ames_sub1/soils.sol
- 官方版本：https://github.com/swat-model/swatplus/releases/tag/62.0.0
- 仓库LICENSE：GNU LGPL 2.1；副本随示例交付。未嵌入外部代码，按格式自主实现。

官方文件解码CSV是格式参考数据，不是上游实测土壤A。soil_05第四层粒级合计101%，完整参考CSV应拒绝；可复现地仅排除整种soil_05得到13种、52层子集，未修改其他数值。B2保留完整原文件。独立Decimal解析比对同名土壤及每层参数；合成多层/单位/列名测试检查泛用性，不能冒充真实多地区验证。尚缺独立、许可明确且所有参数齐全的上游实测数据验证。

## 用户操作

使用源码版：首轮专项转换 → SWAT+ → 分层土壤。原始上传表（本例为官方格式解码参考，不是实测A）：examples/ai_round1/swat_plus/soil/A_official_valid_subset.csv；配置config.json。执行后下载全部转换文件。应有soils.sol，13种土壤、52层，不应有file.cio；soil_01四层深度50/150/840/2000mm，首层bd1.26、k33.01mm/h、clay21/silt34/sand45%。B2_official_soils.sol仅作参考，不能上传为原始表。完整A_official_decoded_layers.csv用于复现拒绝soil_05。

CLI：python -m ecodata.round1 --model swat_plus --input A_official_valid_subset.csv --config config.json --output output

现有便携ZIP未更新。完整工程运行还需HRU、河网、地形、土地利用、管理、初始状态、连接和运行控制等合法一致文件。官方参考工程在未替换土壤的SWAT+62基线运行已在hyd_read_connect.f90:338失败，因此本轮没有实际模型读取/运行通过证据。
