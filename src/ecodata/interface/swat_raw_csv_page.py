"""Raw two-table weather upload with explicit source selections."""
import io,json,hashlib,zipfile
import streamlit as st
from ecodata.core.exceptions import SchemaError
from ecodata.models.adapters.swat_plus_raw_csv import read_csv,convert_csv


def render():
    st.info('原始CSV原样上传。页面内明确筛选和对应日期，不修改源文件、不补值；完整选中记录与质量标记保留在溯源报告。')
    rain=st.file_uploader('原始降水CSV',type=['csv'],key='raw_rain')
    temp=st.file_uploader('原始温度CSV',type=['csv'],key='raw_temperature')
    if not(rain and temp):return
    try:
        selection={}
        for role,upload,label,variables in [('rain',rain,'降水',{'precipitation':'降水数值字段'}),('temperature',temp,'温度',{'air_temperature_max':'最高温字段','air_temperature_min':'最低温字段'})]:
            frame=read_csv(upload.getvalue());cols=list(frame.columns)
            st.markdown('#### '+label+'原始表')
            st.dataframe(frame.head(10))
            date=st.selectbox(label+'日期字段',cols,index=None,key=role+'_date')
            obj=st.selectbox(label+'流域或站点字段',cols,index=None,key=role+'_object')
            if obj is None:return
            value=st.selectbox(label+'流域或站点值',list(frame[obj].unique()),index=None,key=role+'_value')
            mapping={name:st.selectbox(title,cols,index=None,key=role+'_'+name) for name,title in variables.items()}
            unit=st.selectbox(label+'单位（依据原始说明确认）',['mm/day'] if role=='rain' else ['degC'],index=None,key=role+'_unit')
            selection[role]=dict(date_field=date,object_field=obj,object_value=value,values=mapping,unit=unit)
        import datetime
        start=st.date_input('选择开始日期',value=None,min_value=datetime.date(1,1,1),max_value=datetime.date(9999,12,31),key='csv_start')
        end=st.date_input('选择结束日期',value=None,min_value=datetime.date(1,1,1),max_value=datetime.date(9999,12,31),key='csv_end')
        name=st.text_input('输出天气站名称',key='csv_station')
        lat=st.number_input('目标站纬度',value=None,min_value=-90.,max_value=90.,format='%.8f',key='csv_lat')
        lon=st.number_input('目标站经度',value=None,min_value=-180.,max_value=180.,format='%.8f',key='csv_lon')
        elev=st.number_input('目标站高程（m）',value=None,format='%.4f',key='csv_elev')
        location_source=st.text_input('位置与高程的资料依据',key='csv_location_source')
        confirmed=st.checkbox('确认单位、位置依据和两处观测的地理适用性；查看原始质量标记，不把缺测补为0',key='csv_confirm')
        config=dict(project_name='raw_csv_weather',start=str(start),end=str(end),station=dict(name=name,latitude=lat,longitude=lon,elevation=elev),target_model_version='62.0.0')
        key=hashlib.sha256(rain.getvalue()+temp.getvalue()+json.dumps([selection,config,location_source,confirmed],sort_keys=True).encode()).hexdigest()
        if st.session_state.get('csv_key')!=key:st.session_state.pop('csv_result',None)
        st.session_state['csv_key']=key
        ready=confirmed and start and end and name and location_source and all(v is not None for v in (lat,lon,elev)) and all(s['date_field'] and s['object_value'] is not None and s['unit'] and all(s['values'].values()) for s in selection.values())
        if st.button('检查原始CSV并转换天气',disabled=not ready,key='csv_convert'):
            selection['location_source']=location_source
            st.session_state['csv_result']=convert_csv(rain.getvalue(),temp.getvalue(),selection,config)
        if result:=st.session_state.get('csv_result'):
            files,report,nc=result
            st.success('原始CSV筛选、逐日对应与天气转换完成；尚未运行模型。')
            st.json(report)
            buffer=io.BytesIO()
            with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
                for n,b in files.items():z.writestr(n,b)
            st.download_button('下载全部转换文件',buffer.getvalue(),'swat_plus_raw_csv_converted.zip')
            from ecodata.interface.round1_page import render_project_continuation
            render_project_continuation('swat_plus',None,buffer.getvalue(),nc,json.dumps(config).encode(),key)
    except (ValueError,KeyError,UnicodeError,SchemaError) as exc:
        st.session_state.pop('csv_result',None)
        st.error('原始CSV转换被拒绝：'+str(exc))
