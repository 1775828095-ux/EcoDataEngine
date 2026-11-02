# SWAT+ 工程运行页面

见交付根目录《交付说明.md》。本页新增运行管理，不改变专用数据转换数值行为。

版本：SWAT+62.0.0 / 平台1.0.1.dev3+swatui.20261020。
来源：https://github.com/swat-model/swatplus/tree/62.0.0 。运行时许可证副本位于runtime/SWATplus_LICENSE.txt。

状态：上传、检查、日期设置、启动、停止、超时、日志和结果下载已实现。2026-10-20追加检查：运行器20项、四模型相关454项回归通过；页面开发测试实际调用62.0.0并完成官方Hubbard Brook示例的1964-01-01至01-03短运行，下载包含新success.fin与完成日志。

该例属于官方已制备工程，不是独立原始数据经平台转换的全链证明。原件file.cio/basin只有2项，与62.0.0要求的3项不符；测试副本在确认codes.bsn的carbon=0后，仅将新增碳参数文件位置显式设为null，并缩短运行日期，未改物理参数。平台现在会拦截此类项数错误，不自动改写或补造碳参数。原print.prt跳过前2年输出，三天短跑主要证明读取和推进成功，不用于水文结果评价；未代替用户复测。

证据：D:\EcoDataEngine\cases\SWATplus_20261020\UI_success_validation.json。供用户复测工程包：同目录official_example1_62_adapted_3days_UI_input.zip；明确属于适配后的工程输入，不能作为上游原始A。
