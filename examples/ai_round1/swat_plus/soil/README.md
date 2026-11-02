# 上传文件与参考输出
A_official_valid_subset.csv + config.json：可上传，13土壤52层。它是官方输入文件的独立解码格式参考，**不是上游实测原始数据**。
A_official_decoded_layers.csv：完整14土壤56层，soil_05粒级总和101%，用于错误拒绝测试。
B2_official_soils.sol：官方原生参考输出，不能当原始表上传。
subset.py/prepare.py在交付工作目录记录解码和完整性筛选方法；剔除整种soil_05，不填补、不改数值。
数据许可证见LICENSE_official_reference.txt（官方仓库LGPL2.1）；没有将任何外部转换代码并入平台。
