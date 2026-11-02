"""Diagrams label actual preprocessing, reuse and missing raw-build capability."""
import streamlit as st


def render_flow(model,route,mode=None):
    existing=route=='existing'
    labels={
      'swat_plus':('原始DEM/土壤/土地利用/出口及参数','QSWAT+划分 → Editor生成TxtInOut','SWAT+原生运行'),
      'wrf':('原始FNL GRIB与登记地理资料','geogrid → ungrib → metgrid → real','WRF原生积分'),
      'calpuff':('原始气象观测＋预栅格化地理资料','MAKEGEO/气象输入重建；MM4等继续复用','CALMET → CALPUFF'),
      'camx':('原始排放清单/独立化学场：构建链未实现','化学成品复用；气象可选原始GFS构建','CAMx原生积分'),
      'cmaq':('原始排放清单/独立化学场：构建链未实现','化学成品重组；气象可选原始GFS构建','CCTM原生积分')}
    raw,pre,native=labels[model]
    source='已有完整工程；保留配套资料' if existing else raw
    preparation='核对版本、引用、网格/对象、时段；接入工程副本' if existing else pre
    if model=='cmaq' and mode in ('icon_profile','bcon_profile'):
        preparation=('ICON/profile' if mode=='icon_profile' else 'BCON/profile')+'原生预处理 → 接入O3；其余224物种保留'
    dashed=not existing and model in ('camx','cmaq')
    graph='digraph G { rankdir=LR; node [shape=box,style="rounded,filled",fillcolor="#e9f3ff",fontname="Microsoft YaHei"]; '
    for name,label in [('a','本次数据＋配置'),('b','转换与检查'),('c','下载转换ZIP'),('d',source),('e',preparation),('f',native),('g','读取原生输出 → 数值检查 → 图表/下载')]:graph+=f'{name} [label="{label}"]; '
    graph+='a -> b -> c; c -> e; d -> e -> f -> g; '
    if dashed:graph+='d [style="rounded,dashed",label="'+raw+'（未实现）"]; d -> e [style=dashed]; '
    graph+='}'
    st.graphviz_chart(graph)
    st.caption('图中运行和出图是后续执行步骤；需本次任务成功记录才能判定跑通。虚线为尚未实现的原始构建链。')
