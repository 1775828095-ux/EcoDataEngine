# 本轮参考资料与工具通知

- SWAT+62.0.0官方参考输入（包括样例ZIP中的引用数据库和B2片段）来自 `https://github.com/swat-model/swatplus/tree/62.0.0/refdata/Ames_sub1`，原项目LGPL2.1文本附在 `licenses/SWAT_REFERENCE_LGPL21.txt`。本轮CSV/ZIP为注明范围的独立解码/子集，官方记录值未为了通过检查改动；源文件头Editor2.3.3/rev60.5.7如实记录。自行实现的writer没有复制Fortran实现。
- CMAQ物种标识目录引用 `https://github.com/USEPA/CMAQ/tree/CMAQv5.5_2Oct2024/CCTM/src/MECHS/cb6r3_ae7_aq` 的GC/AE文件，官方许可证附在 `licenses/CMAQ_EPA_license.md`。只用于指定机制物种名校验，未复制CCTM算法；未据标识自动猜测摩尔质量。
- NOAA IGRA、USCRN和PSL公开资料保留来源和请求/散列信息，不暗示美国政府/NOAA背书。JRC EDGAR公开数据按发布机构的署名条件使用，不擅自标注CC-BY；源网址与使用条件在相应样例配置和模块指南。
- WPS/CALMET/CAMx仅核对官方格式/读入结构，本轮源码增量包未包含这些程序的代码或可执行文件，其安装/分发须另遵守对应许可。本轮没有安装或运行它们。
- PseudoNetCDF3.4.1（LGPL3）和GeographicLib2.1（MIT）是交付工作目录中可选独立验证工具，不嵌入生产writer；源码增量ZIP不包含vendor，其许可证随原包保留。SciPy、netCDF4、Pint、pyproj等已有依赖不在本轮重新打包。
