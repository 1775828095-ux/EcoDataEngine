# 模型适配：当前数据转换为一类输入

## 操作

1. 上传文件，确认自动识别结果。
2. 输出目标选择“环境模型”，再选择模型。
3. 在“我要把当前数据转换成哪一种模型输入”中选一种，例如 AERMOD 的 source_parameters、receptors、meteorology 或 terrain。
4. 配置有逐字段要求时，系统按字段名、standard_name 和 aliases 推荐映射，用户确认未匹配字段。配置仅有输入类型索引时，选择该类型使用的现有变量，不虚构该类型的全部必需物理字段。
5. 确认输出格式以及本类输入的时间/空间要求，执行转换，下载按模型和输入类型命名的标准化文件及报告。

“模型完整性检查”位于独立折叠区。点击检查整套输入，只显示整个模型的匹配结果，不控制单类输入转换按钮。

## 实现边界

模型适配只使用所选输入的要求，其他输入和多文件依赖记录为未检查，不当作当前转换的缺失变量。逐字段配置可以在可编辑模型需求的 inputs 条目增加 fields 列表，包含 name、standard_name、aliases、unit、data_type、required 等。

有明确单位时，当前自动支持等价单位、g/L—mg/L—ug/L 与 degC—K 换算；未知或不兼容单位阻止转换，绝不只改单位标签。时间、空间及数据结构处理仍调用既有 Pipeline。本类输入没有明确固定分辨率时，不把其他输入或整套模型的分辨率强行套用，也不把 simulation-specific 当作有效频率字符串。

没有逐字段配置的模型输入只能输出标准化预处理数据，报告 field_schema_verified=false、can_generate_native_model_input=false。特别是 AERMOD 不同 source type 的必需字段及 AERMET/AERMAP 原生文件、WASP 分段与状态变量对应关系，不能仅靠通用 CSV/NetCDF 转换宣称完成验证。

## 真实数据演示

复现程序：examples/demo_selected_model_input.py，通过 Streamlit AppTest 实际执行页面操作，不是手工构造转换结果。

输入：已有水环境数据复用包中的 02_整理数据/manual_2021_2025_multianalyte_source.csv。

目标：WASP → water_quality_state → NetCDF。67,728 条记录、13 个字段。输出位置：output/p4/selected_model_input_demo/wasp_water_quality_state_standardized.nc 和 matching_report.json。独立回读逐字段检查数值、文字及单位一致。此次没有分段映射、没有宣称 WASP 原生运行输入完成。
