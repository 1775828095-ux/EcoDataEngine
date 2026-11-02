# CAMx 7.40 CB6r5 光解率表专项转换

专项页选择CAMx，导入examples/ai_round1/camx/A_photolysis_table.csv和config_photolysis.json，执行并下载photolysis.CAMx7.40.CB6r5.txt及报告。

CLI：`python -m ecodata.round1 --model camx --input examples/ai_round1/camx/A_photolysis_table.csv --config examples/ai_round1/camx/config_photolysis.json --output <输出目录>`。使用平台_runtime/python.exe；PYTHONPATH设为当前目录src和当前目录。

该输入是静态、晴空、条件查找表，不含经纬度/日期，不是时空气象网格，也不运行TUV。A来自官方2016-06-10 CB6r5查找表的独立解码CSV；本轮验证其重新编码，不验证上游辐射计算。

仅支持默认CAMx7.40尺寸：5臭氧柱×5反照率×3地形×11离地高度×23个CB6r5主光解反应×10天顶角。单位必须显式声明：臭氧柱DU（250..600），反照率无量纲，高度/地形km，光解率min^-1或s^-1。输出min^-1；秒率乘60。天顶角索引映射0,10,20,30,40,50,60,70,78,86度，不能把索引当角度。按臭氧/反照率/地形/高度及固定反应序写出，每行1空格+10个12字符数值。缺失、重复、非有限、负值、未知反应、错误尺寸/单位及额外字段拒绝，不填补或剪裁；CSV记录顺序可变。block_id不进入原生文件，其原始上下文映射保存在转换报告。

在已有CAMx.in中设置`Photolysis_Rates = 'photolysis.CAMx7.40.CB6r5.txt'`，匹配CAMx7.40 CB6r5 CHEMPARAM主光解反应序（23个；官方CB6r5样例ID为1,8,9,21,27,28,38,43,47,50,56,88,92,97,98,106,109,111,114,116,125,126,154）。不同化学机制或修改后的CHEMPARAM不在范围内。保留完整原项目配置，输出不替代CAMx.in。条件覆盖是否适合目标项目需另行评估。

格式与单位依据：[官方7.40源码](https://www.camx.com/getmedia/CAMx_v7.40.src.260801.tgz)中CAMx/readpht.f、readchm.f、kphoto.f、Includes/camx.prm.v7.40和Mod_src/chmstry.f；[官方小型输入包](https://www.camx.com/getmedia/v7.40.specific.inputs.260801.tgz)中的tuv.do_CB6r5.20160610及CHEMPARAM。软件许可保留examples/ai_round1/camx/CAMx_LICENSE.txt；样例无单独数据许可证声明。

文件独立解析与数值转换检查不代表实际CAMx读取或模拟运行。仍缺全套域/网格控制、3D气象、排放、全机制初始/边界场及相应其它输入。旧便携ZIP未重打包，不包含此修复。
