# CAMx7.40合成边界全数组案例
两组LATLON/LCC、4×3/3×4、2/3层、不同索引起点、列名、列顺序、行顺序及混合比/质量浓度单位。
公式a=10+100*t+10*k+j*NX+i；O3=a ppbV，PSO4=a*0.1 ug/m3；源单位可换ppmV、mg/m3，全部格点均明确供给。
2024-02-29T23Z至2024-03-01T01Z，两小时真实格式但数值为合成、不是观测。
官方格式来源：https://www.camx.com/getmedia/CAMx_v7.40.src.260801.tgz
IO_NCF/ncf_bndprep.f、ncf_readbnd.f；读取COL×ROW×LAY全数组，边界在网格内侧边行列，非CMAQ ghost perimeter。
源码只用作格式依据、不部署、不复制模型代码。新增合成数值CC0，官方源码许可见已有evidence/LICENSE。
无独立真实A/B2，证据限于格式和合成转换通用性；未运行CAMx。
