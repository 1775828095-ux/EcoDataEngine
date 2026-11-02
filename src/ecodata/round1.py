"""Reproducible scoped converters used identically by CLI and Streamlit."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import pandas as pd
from ecodata.core.exceptions import SchemaError


def convert(model, content, config, *, weather_station_index=None):
    if not isinstance(config,dict):raise SchemaError('配置JSON必须为对象')
    modes={'calpuff':{None,'upper_air','precipitation','geography','receptors','constant_points'},'swat':{None},'swat_plus':{None,'soil','wgn','hru_attributes','landuse_database','aquifer_parameters','connections','spatial_landuse','management_schedule'},'wrf':{None,'pressure_fields','geogrid_tiles','input_package'},'camx':{None,'gridded_emissions','initial_concentrations','boundary_concentrations'},'cmaq':{None,'gridded_emissions','bcon_profile','boundary_concentrations'}}
    mode=config.get('mode')
    if model not in modes or not isinstance(mode,(str,type(None))) or mode not in modes[model]:
        raise SchemaError(f'模型{model}不支持转换mode={mode!r}；请使用对应专项配置，不回退到其他格式')
    if weather_station_index is not None and not (model=='swat_plus' and config.get('mode')=='wgn'):
        raise SchemaError('weather-sta.cli引用检查仅适用SWAT+ mode=wgn')
    if model=='calpuff' and mode=='constant_points':
        from ecodata.models.adapters.calpuff_point_sources import export_points
        files,report=export_points(content,config)
    elif model=='calpuff' and mode=='receptors':
        from ecodata.models.adapters.calpuff_receptors import export_receptors
        files,report=export_receptors(content,config)
    elif model=='calpuff' and mode=='geography':
        from ecodata.io.structures import read_structure
        from ecodata.models.adapters.calmet_geography import export_geography
        files,report=export_geography(read_structure('source.nc',content),config)
    elif model=='cmaq' and mode=='boundary_concentrations':
        from ecodata.io.structures import read_structure
        from ecodata.models.adapters.cmaq_boundary import export_boundary
        files,report=export_boundary(read_structure('source.nc',content),config)
    elif model=='cmaq' and config.get('mode')=='bcon_profile':
        from ecodata.models.adapters.cmaq_bcon_profile import export_bcon_profile
        files,report=export_bcon_profile(content,config)
    elif model=='cmaq' and config.get('mode')=='gridded_emissions':
        from ecodata.io.structures import read_structure
        from ecodata.models.adapters.cmaq_gridded_emissions import export_gridded_emissions
        files,report=export_gridded_emissions(read_structure('source.nc',content),config)
    elif model in {'calpuff','swat','camx','cmaq'}:
        encoding=config.get('encoding','utf-8-sig')
        if encoding not in {'utf-8','utf-8-sig','gb18030'}:raise SchemaError('CSV仅支持明确UTF-8/UTF-8 BOM/GB18030编码')
        try:
            # CSV reader must not silently mangle duplicate columns.
            import csv
            decoded=content.decode(encoding,errors='strict')
            header=next(csv.reader(io.StringIO(decoded)))
            if len(header)!=len(set(header)):raise SchemaError('CSV表头存在重复字段')
            frame=pd.read_csv(io.StringIO(decoded),keep_default_na=False,na_values=[''],
                              **({'float_precision':'round_trip'} if model=='cmaq' else {}))
        except (UnicodeError,ValueError,StopIteration) as exc:raise SchemaError(f'CSV读取失败：{exc}') from exc
        if model=='calpuff':
            if config.get('mode')=='upper_air':
                from ecodata.models.adapters.calmet_upper_air import export_upper_air
                files,report=export_upper_air(frame,config)
            elif config.get('mode')=='precipitation':
                from ecodata.models.adapters.calmet_precipitation import export_precipitation
                files,report=export_precipitation(frame,config)
            else:
                from ecodata.models.adapters.calmet_surface import export_surface
                files,report=export_surface(frame,config)
        elif model=='swat':
            from ecodata.models.adapters.swat_weather import export_weather
            files,report=export_weather(frame,config)
        elif model=='camx':
            if mode=='boundary_concentrations':
                from ecodata.models.adapters.camx_boundary import export_boundary
                files,report=export_boundary(frame,config)
            elif mode=='initial_concentrations':
                from ecodata.models.adapters.camx_initial import export_initial
                files,report=export_initial(frame,config)
            elif config.get('mode')=='gridded_emissions':
                from ecodata.models.adapters.camx_emissions import export_gridded_emissions
                files,report=export_gridded_emissions(frame,config)
            else:
                from ecodata.models.adapters.camx_photolysis import export_photolysis
                files,report=export_photolysis(frame,config)
        else:
            from ecodata.models.adapters.cmaq_icon_profile import export_icon_profile
            files,report=export_icon_profile(frame,config)
    elif model=='wrf' and mode=='input_package':
        from ecodata.models.adapters.wps_input_package import export_package
        files,report=export_package(content,config)
    elif model=='wrf':
        from ecodata.io.structures import read_structure
        from ecodata.models.adapters.wps_sst import export_sst
        dataset=read_structure('source.nc',content)
        if mode=='geogrid_tiles':
            from ecodata.models.adapters.wps_geography import export_geography
            files,report=export_geography(dataset,config)
        elif config.get('mode')=='pressure_fields':
            from ecodata.models.adapters.wps_fields import export_pressure
            files,report=export_pressure(dataset,config)
        else:
            files,report=export_sst(dataset,config)
    elif model=='swat_plus':
        if config.get('mode')=='management_schedule':
            from ecodata.models.adapters.swat_plus_management import export_management
            files,report=export_management(content,config)
        elif config.get('mode')=='spatial_landuse':
            from ecodata.models.adapters.swat_plus_spatial_landuse import export_spatial_landuse
            files,report=export_spatial_landuse(content,config)
        elif config.get('mode')=='connections':
            from ecodata.models.adapters.swat_plus_connections import export_connections
            files,report=export_connections(content,config)
        elif config.get('mode')=='aquifer_parameters':
            from ecodata.models.adapters.swat_plus_aquifer import export_aquifer_bundle
            files,report=export_aquifer_bundle(content,config)
        elif config.get('mode')=='landuse_database':
            from ecodata.models.adapters.swat_plus_landuse import export_landuse
            files,report=export_landuse(content,config)
        elif config.get('mode')=='hru_attributes':
            from ecodata.models.adapters.swat_plus_hru import export_hru_bundle
            files,report=export_hru_bundle(content,config)
        elif config.get('mode')=='soil':
            from ecodata.models.adapters.swat_plus_soil import export_soils
            files,report=export_soils(content,config)
        elif config.get('mode')=='wgn':
            from ecodata.models.adapters.swat_plus_wgn import export_wgn
            files,report=export_wgn(content,config,weather_station_index)
        else:
            from ecodata.models.adapters.swat_plus_upload import export_uploaded_weather
            files,report=export_uploaded_weather(content,config)
    else:raise SchemaError('未知专项模型；支持范围见docs/conversion_coverage_status.md')
    report.update(source_sha256=hashlib.sha256(content).hexdigest(),
                  configuration=config,output_sha256={n:hashlib.sha256(b).hexdigest() for n,b in files.items()},
                  evidence_level='conversion candidate; independent checks are in test report, not self-certified model acceptance')
    if weather_station_index is not None:
        report['weather_station_index_sha256']=hashlib.sha256(weather_station_index).hexdigest()
    files['conversion_report.json']=json.dumps(report,ensure_ascii=False,indent=2).encode('utf-8')
    return files,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,choices=['calpuff','swat','swat_plus','wrf','camx','cmaq'])
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--weather-station-index',type=Path,help='WGN模式可选：只核对现有weather-sta.cli的WGN名称引用，不修改文件')
    args=parser.parse_args()
    try:
        files,report=convert(args.model,args.input.read_bytes(),json.loads(args.config.read_text(encoding='utf-8-sig')),
                             weather_station_index=args.weather_station_index.read_bytes() if args.weather_station_index else None)
    except (SchemaError,ValueError,KeyError) as exc:
        parser.exit(2,f'Conversion rejected: {exc}\n')
    args.output.mkdir(parents=True,exist_ok=True)
    for name,data in files.items():(args.output/name).write_bytes(data)
    print(json.dumps({k:v for k,v in report.items() if k not in {'configuration','output_sha256'}},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
